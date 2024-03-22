#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from autoencoder.autoencoder_utils import DistributedBucketingDataLoader, eval_model_loss
from interpolation import parse_args, load_model
from autoencoder.noiser import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def eval_chkpt(args):
    args, model, logger = load_model(args, suffix="eval.txt")
    eval_dataloader = DistributedBucketingDataLoader(args.dev_pt_dir, args.batch_size_per_gpu, args.sentence_len, shuffle=False)
    device = torch.device("cuda")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
    noisers = {'bert':noise_bert, 'bart':noise_bart, 'none':noise_none, 'sub': noise_sub, 'sub_u':noise_sub_uniform}
    task_id = args.task_id if args.task_id else args.load_ckpt
    print(task_id)
    logger.info('='*35+task_id+'='*35)
    noise_level = 0.3 #args.noiser_ratio
    for noise_ratio in [noise_level, 0.0]:
        logger.info('-'*80)
        logger.info(f"noise level at {noise_ratio}")
        args.noiser_ratio = noise_ratio
        noiser = noisers[args.noiser](model.encoder.tokenizer, mlm_probability=args.noiser_ratio)
        eval_model_loss(model, gpt_model, gpt_tokenizer, args.ae_step, eval_dataloader, noiser, device, logger, max_valid_size=args.valid_size, onlypart=True)

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    eval_chkpt(args)