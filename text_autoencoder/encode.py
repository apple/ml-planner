#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from transformers import BertModel, BertTokenizer
import torch
import json
import argparse
from torch.nn.utils.rnn import pad_sequence
from pbart.data_utils import FeatureDataset
from torch import multiprocessing as mp
import os
import tqdm


def parse_args(): 
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_path", type=str, default='../models/story_ae_5e-5_cnt/epoch0-step199999-ppl2.454-BERT.pkl')
    parser.add_argument("--batch_size_per_process", type=int, default=4096)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--encoder_max_seq_len", type=int, default=128)
    parser.add_argument("--use_all_gpus", action='store_true')
    return parser.parse_args()

def encode_file(ifile, ofile, model, device, batch_size, encoder_max_seq_len, use_all_gpus):
    chunk = torch.load(ifile)
    paragraphs = FeatureDataset(chunk)
    mw = model.encoder.layer[-1].output.LayerNorm.weight
    mb = model.encoder.layer[-1].output.LayerNorm.bias
    if use_all_gpus:
        print ("using %d gpus"%torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.eval()
    print (ifile, "=>", ofile)
    res = []
    data, lens = [], []
    num_paragraphs = len(paragraphs)
    for i in tqdm.tqdm(range(num_paragraphs)):
        cur_data, cur_leng = paragraphs[i]
        data.extend(cur_data)
        lens.append(cur_leng)
        if i!=num_paragraphs-1 and len(data) < batch_size:
            continue
        with torch.no_grad():
            input_ids_bert = pad_sequence([ torch.tensor(x[:encoder_max_seq_len]) for x in data], batch_first=True, padding_value=0)
            input_ids_bert = input_ids_bert.to(device)
            encoded_layers, _ = model(input_ids_bert, attention_mask=torch.ne(input_ids_bert, 0))
            encoded = encoded_layers[:, 0, :] # bsz x dim
            encoded = (encoded - mb) / mw
            ret = encoded.detach().cpu()
        assert ret.size(0) == len(data) == sum(lens)
        cur = 0
        for leng in lens:
            res.append(ret[cur:cur+leng])
            cur += leng
        data = []
        lens = []
    torch.save(res, ofile)

def main(local_rank, args):
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", state_dict=torch.load(args.model_path))
    model = model.to(device)
    model.eval()
    files = [ (os.path.join(args.input_dir, fname), os.path.join(args.output_dir, fname[:-3]+'.encoded.pt')) for fname in os.listdir(args.input_dir) if fname.endswith('.pt')]
    files.sort()
    for ifile, ofile in files[local_rank::args.num_processes]:
        encode_file(ifile, ofile, model, device, args.batch_size_per_process, args.encoder_max_seq_len, args.use_all_gpus)

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.num_processes == 1:
        main(0, args)
        exit(0)
    mp.spawn(main, args=(args,), nprocs=args.num_processes)
