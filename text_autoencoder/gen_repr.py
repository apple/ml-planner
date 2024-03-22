#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from autoencoder.autoencoder_utils import DistributedBucketingDataLoader, eval_model_loss, generate_hidden
from interpolation import parse_args, load_model
from autoencoder.noiser import *
import os
import numpy as np
import re
from utils import boolean_string



def gen_hiddens(args, max_size = None, output_data = False):
    args, model, _ = load_model(args, suffix="repr.txt")
    if max_size is None:
        max_size = args.max_size
    train_dataloader = DistributedBucketingDataLoader(args.train_pt_dir, args.batch_size_per_gpu, args.sentence_len, shuffle=False)
    device = torch.device("cuda")
    noisers = {'bert':noise_bert, 'bart':noise_bart, 'none':noise_none, 'sub': noise_sub, 'sub_u':noise_sub_uniform}
    task_id = args.task_id if args.task_id else os.path.basename(args.load_ckpt)
    print(task_id)
    # logger.info('='*35+task_id+'='*35)
    noise_ratio = 0.0
    # logger.info('-'*80)
    # logger.info(f"noise level at {noise_ratio}")
    args.noiser_ratio = noise_ratio
    noiser = noisers[args.noiser](model.encoder.tokenizer, mlm_probability=args.noiser_ratio)
    hiddens, conds = generate_hidden(model, train_dataloader, noiser, device, cond_model=args.cond_model, max_size=max_size)
    if output_data:
        save_dataset(args, train_dataloader, model)
    return hiddens, conds

def save_dataset(args, dataloader, model):
    test_set = []
    total_size = 0
    labels = []
    num_sample = args.max_size
    for batch in dataloader:
        input_ids_dec = batch[1].cuda()
        if args.cond_feature:
            input_ids_dec_cond = batch[3].cuda()
            cond = model.decode(input_ids_dec_cond, tokenizer='enc')
            cond = [re.sub('\n', ' ', s) for s in cond]
            labels.extend(cond)
        else:
            labels.extend(batch[3])
        input_sentence = model.decode(input_ids_dec, tokenizer='dec') #, skip_special_tokens=True
        # import pdb;pdb.set_trace()
        input_sentence = [re.sub('\n', ' ', s) for s in input_sentence]
        input_sentence = [s.rstrip("!") for s in input_sentence]
        test_set.extend(input_sentence)
        total_size += batch[0].shape[0]
        if total_size >= num_sample:
            break
    test_set = test_set[:num_sample]
    #### write to file ####
    with open(os.path.join(args.train_pt_dir, 'prefix.txt'), "w") as f_pre, open(os.path.join(args.train_pt_dir, 'human.txt'), "w") as f_human:
        for t, c in zip(test_set, labels):
            f_pre.write(f"{c}\n")
            f_human.write(f"{t}\n")

    


def gen_repr(args):
    hiddens, conds = gen_hiddens(args, output_data=True)
    task_id = args.task_id if args.task_id else os.path.basename(args.load_ckpt)
    # torch.save(hiddens, os.path.join(args.save_dir, f"{args.max_size}_repr.pt"))
    if args.max_size == -1:
        args.max_size = None
    filename = f"{args.max_size}_repr_{task_id}.npz" if args.max_size else f"repr_{task_id}.npz"
    assert(len(hiddens)==len(conds))
    print(f"Total {len(hiddens)} reprs before filtering")
    hiddens_filtered, conds_filtered = [], []
    for i, h in enumerate(hiddens):
        if h.shape == hiddens[0].shape:
            hiddens_filtered.append(h)
            conds_filtered.append(conds[i])
    hiddens = hiddens_filtered
    conds = conds_filtered
    assert(len(hiddens)==len(conds))
    print(f"Total {len(hiddens)} reprs after filtering")
    with open(os.path.join(args.train_pt_dir, filename), "wb") as f:
        if isinstance(hiddens[0], torch.Tensor):
            hiddens=[h.cpu().numpy() for h in hiddens]
        if isinstance(conds[0], torch.Tensor):
            conds=[c.cpu().numpy() for c in conds]
        np.savez(f, hiddens=np.array(hiddens), conds=np.array(conds))

if __name__ == "__main__":
    parser = parse_args()
    parser.add_argument('--cond_model', default=None, choices=('zh', 'de', 't5', None), help='path to the conditional model checkpoint')
    parser.add_argument("--cond_feature", type=boolean_string, default=True)
    args = parser.parse_args()
    # for fname in test_models:
    gen_repr(args)