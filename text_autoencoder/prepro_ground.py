#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse
import pdb
import json
import subprocess as sp
import os
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer
from transformers import BertTokenizer
from functools import partial
import random
import numpy as np


from multiprocessing import Pool
bert_toker = BertTokenizer.from_pretrained('bert-base-uncased')
gpt2_toker = GPT2Tokenizer.from_pretrained('gpt2')


class Feature:
    def __init__(self, bert_ids=None, gpt2_ids=None, post=None, pre_ids=None):
        self.bert_ids = bert_ids
        self.gpt2_ids = gpt2_ids
        self.raw_text = post
        self.cond = pre_ids


def get_feature(line, maxlen, raw_pre=False, pre_tokenizer=None):
    pre = line[0]
    post = line[1].strip()
    bert_ids = bert_toker.encode(post)[:maxlen]
    gpt2_ids = gpt2_toker.encode(post)[:maxlen]
    if raw_pre:
        pre_ids = pre
    else:
        if pre_tokenizer is None:
            pre_tokenizer = bert_toker
        pre_ids = pre_tokenizer.encode(pre)[:maxlen]
    feature = vars(Feature(bert_ids, gpt2_ids, post, pre_ids))
    # print(rating)
    return feature

# Define the split ratio
split_ratio = {"train": 0.8, "dev": 0.1, "test": 0.1}

def distributed_main(args):
    files = [os.path.join(args.corpus, fname) for fname in os.listdir(args.corpus) if fname.endswith('.txt')]
    pool = Pool()
    dir_path = os.path.join(args.corpus, 'parsed_raw_pre')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


    for file in tqdm(files, total=len(files)):
        print(f"Processing {file}")
        with open(file, "r", encoding="utf-8") as reader:
            chunk = []
            block = []
            cnt = 0
            for line in reader:
                cnt += 1
                sentences = line.strip()
                if len(sentences.split('\t'))!=2: 
                    continue
                pre, post = sentences.split('\t')
                if len(pre)<10 or len(post)<10: continue
                if not sentences:
                    if block:
                        chunk.append(block)
                        block = []
                else:
                    block.append([pre, post])
                    if args.bidirection:
                        block.append([post, pre])
            # save last chunk
            if block:
                chunk.append(block)
            if chunk:
                # feature = get_feature(chunk[0], maxlen=args.maxlen)
                print(f"In total {len(chunk[0])}/{cnt}")
                data = chunk[0]
                np.random.shuffle(data)
                # Calculate the split indices
                train_split = int(split_ratio["train"] * len(data))
                dev_split = train_split + int(split_ratio["dev"] * len(data))

                # Split the data into train, dev, and test sets
                dataset = {}
                dataset["train"] = data[:train_split]
                dataset["dev"] = data[train_split:dev_split]
                dataset["test"] = data[dev_split:]
                
                for split_name in split_ratio.keys():
                    if len(dataset[split_name]) != 0:
                        feature = pool.map(partial(get_feature, maxlen=args.maxlen, raw_pre=True), dataset[split_name], len(dataset[split_name])//16)
                        split_path = os.path.join(os.path.dirname(file), 'parsed_raw_pre'+f'{"_ae" if args.bidirection else ""}', split_name)
                        if not os.path.exists(split_path):
                            os.mkdir(split_path)
                        torch.save(feature, os.path.join(split_path, f'{os.path.basename(file)[:-4]}.pt'))
                        # Save pre and post sentences separately
                        with open(os.path.join(split_path, f'source_{split_name}.txt'), 'w', encoding='utf-8') as source_file:
                            source_file.write('\n'.join([x[0] for x in dataset[split_name]]))
                        with open(os.path.join(split_path, f'target_{split_name}.txt'), 'w', encoding='utf-8') as target_file:
                            target_file.write('\n'.join([x[1] for x in dataset[split_name]]))
                        

    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='file name of training corpus')
    parser.add_argument('--maxlen', type=int, default=512, 
                        help='file name of training corpus')
    parser.add_argument('--bidirection', action='store_true', 
                        help='for ae training')
    args = parser.parse_args()

    if os.path.isdir(args.corpus):
        distributed_main(args)


