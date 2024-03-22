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
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import set_seed


def gen_hiddens(args, local_rank):
    args, model, _ = load_model(args, suffix="repr.txt")
    train_dataloader = DistributedBucketingDataLoader(args.train_pt_dir, args.batch_size_per_gpu, args.sentence_len, rank=local_rank, num_replica=args.world_size, shuffle=False)
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
    hiddens, conds = generate_hidden(model, train_dataloader, noiser, device, max_size=args.max_size)
    return hiddens, conds


def gen_repr(args, local_rank):
    set_seed(args.seed+local_rank)
    hiddens, conds = gen_hiddens(args, local_rank)
    task_id = args.task_id if args.task_id else os.path.basename(args.load_ckpt)
    # torch.save(hiddens, os.path.join(args.save_dir, f"{args.max_size}_repr.pt"))
    filename = f"{args.max_size}_repr_{task_id}.npz" if args.max_size else f"repr_{task_id}.npz"
    assert(len(hiddens)==len(conds))
    # if args.world_size == 1 or (dist.get_rank() == 0):
    print(f"Rank {local_rank}: Total {len(hiddens)} reprs before filtering")
    hiddens_filtered, conds_filtered = [], []
    for i, h in enumerate(hiddens):
        if h.shape == hiddens[0].shape:
            hiddens_filtered.append(h)
            conds_filtered.append(conds[i])
    hiddens = hiddens_filtered
    conds = conds_filtered
    assert(len(hiddens)==len(conds))
    print(f"Rank {local_rank}: Total {len(hiddens)} reprs after filtering")
    if args.world_size == 1 or (dist.get_rank() == 0):
        # Use all_reduce to get the vectors from each GPU
        dist.all_reduce(hiddens, op=dist.reduce_op.SUM)

        # Concatenate the vectors
        concat_vec = torch.cat(vec_list)

        with open(os.path.join(args.train_pt_dir, filename), "wb") as f:
            np.savez(f, hiddens=np.array(hiddens), conds=np.array(conds))

def init_processes(local_rank, args, backend='nccl'):
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank+local_rank, world_size=args.world_size)
    gen_repr(args, local_rank)

    

# Use all_reduce to get the vectors from each GPU
dist.all_reduce(vec_list, op=dist.reduce_op.SUM)

# Concatenate the vectors
concat_vec = torch.cat(vec_list)

if __name__ == "__main__":
    parser = parse_args()
    # distributed
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--MASTER_ADDR', type=str, default='localhost')
    parser.add_argument('--MASTER_PORT', type=str, default='29505')
    parser.add_argument('--start_rank', type=int, default=0)
    args = parser.parse_args()
    # for fname in test_models:
    if args.world_size == 1:
        gen_repr(args, 0)
        exit(0)
    mp.spawn(init_processes, args=(args,), nprocs=args.gpus)