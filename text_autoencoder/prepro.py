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


class Feature:
    def __init__(self, bert_ids=None, gpt2_ids=None, raw_text=None, cond=None):
        self.bert_ids = bert_ids
        self.gpt2_ids = gpt2_ids
        self.raw_text = raw_text
        self.cond = cond



from multiprocessing import Pool
bert_toker = BertTokenizer.from_pretrained('bert-base-uncased')
gpt2_toker = GPT2Tokenizer.from_pretrained('gpt2')


def get_feature(line, maxlen):
    raw_text = line.strip()
    bert_ids = bert_toker.encode(raw_text)[:maxlen]
    gpt2_ids = gpt2_toker.encode(raw_text)[:maxlen]
    feature = vars(Feature(bert_ids, gpt2_ids, raw_text))
    # print(rating)
    return feature


# Define the split ratio
split_ratio = {"train": 0.8, "dev": 0.1, "test": 0.1}

def split_sentences(text):
    # split the text based on any sentence ending characters
    split_text = re.split(r'(?<=[.!?])[\s]+', text)
    result = []
    for sentence in split_text:
        ending = re.search(r'[.!?]+$', sentence)
        if ending:
            ending = ending.group()
            result.append(sentence)
        else:
            result.append(sentence)
    return result

def distributed_main(args):
    files = [os.path.join(args.corpus, fname) for fname in os.listdir(args.corpus) if fname.endswith('.json')]
    pool = Pool()
    for file in tqdm(files, total=len(files)):
        print(f"Processing {file}")
        with open(file, "r", encoding="utf-8") as reader:
            chunk = []
            block = []
            for line in reader:
                parsed_line = json.loads(line)
                text = parsed_line['text']
    
                if not text.strip():
                    if block:
                        chunk.append(block)
                        block = []
                else:
                    block.append(text)
            # save last chunk
            if block:
                chunk.append(block)
            if chunk:
                # feature = get_feature(chunk[0], maxlen=args.maxlen)
                print(f"In total {len(chunk[0])}")
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
                        feature = pool.map(partial(get_feature, maxlen=args.maxlen), dataset[split_name], len(dataset[split_name])//4)
                        if not os.path.exists(os.path.join(os.path.dirname(file), 'parsed', split_name)):
                            os.makedirs(os.path.join(os.path.dirname(file), 'parsed', split_name))
                        torch.save(feature, os.path.join(os.path.dirname(file), 'parsed', split_name, f'{os.path.basename(file)[:-4]}{split_name}.pt'))

    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='file name of training corpus')
    parser.add_argument('--maxlen', type=int, default=256, 
                        help='file name of training corpus')
    args = parser.parse_args()

    if os.path.isdir(args.corpus):
        distributed_main(args)


