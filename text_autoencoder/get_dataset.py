#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from autoencoder.autoencoder_utils import DistributedBucketingDataLoader, eval_gpt2, cal_entropy, calc_self_bleu
from interpolation import parse_args, load_model
import tqdm
import os
import re

def eval_dataset(args):
    args, model, logger = load_model(args, suffix = "dev_set_eval.txt")
    test_set = []

    eval_dataloader = DistributedBucketingDataLoader(args.dev_pt_dir, args.batch_size_per_gpu, args.sentence_len, shuffle=False)
    test_set = []
    total_size = 0
    labels = []
    num_sample = args.max_size
    for batch in eval_dataloader:
        input_ids_dec = batch[1].cuda()
        if args.cond_feature:
            input_ids_dec_cond = batch[3].cuda()
            cond = model.decode(input_ids_dec_cond, tokenizer='enc')
            if args.remove_dash_n:
                cond = [re.sub('\n', ' ', s) for s in cond]
            labels.extend(cond)
        else:
            labels.extend(batch[3])
        model.decoder.tokenizer.pad_token_id = 0
        input_sentence = model.decode(input_ids_dec, tokenizer='dec')
        if args.remove_dash_n:
            input_sentence = [re.sub('\n', ' ', s) for s in input_sentence]
        input_sentence = [re.sub('!', '', s) for s in input_sentence]
        test_set.extend(input_sentence)
        total_size += batch[0].shape[0]
        if total_size >= num_sample:
            break
    test_set = test_set[:num_sample]
    #### write to file ####
    with open(os.path.join(args.dev_pt_dir, 'prefix.txt'), "w") as f_pre, open(os.path.join(args.dev_pt_dir, 'human.txt'), "w") as f_human:
        for t, c in zip(test_set, labels):
            f_pre.write(f"{c}\n")
            f_human.write(f"{t}\n")
        exit()


if __name__ == "__main__":
    parser = parse_args()
    parser.add_argument("--cond_feature", action = 'store_true')
    parser.add_argument("--remove_dash_n", action = 'store_true')
    args = parser.parse_args()
    eval_dataset(args)