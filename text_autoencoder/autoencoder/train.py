#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse
import logging
import os, random
import pickle
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from datetime import datetime
from utils import set_seed, average_gradients, boolean_string
from autoencoder_utils import DistributedBucketingDataLoader, eval_model_loss
from autoencoder import AutoEncoder, VAE, encoderModels, decoderModels
from optim import Adamax
from noiser import *
from tensorboardX import SummaryWriter
import tqdm
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer
#########################################################################
# Prepare Parser
##########################################################################

def parse_args():
    parser = argparse.ArgumentParser(description='text auto-encoder model')
    parser.add_argument("--seed", type=int, default=88)
    # parser.add_argument("--sentence_len", type=int, default=128)
    # parser.add_argument("--noiser", type=str, choices=['bart', 'bert', 'none'], default='bart')

    # learning
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument("--enc_lr", type=float, default=None)
    parser.add_argument("--dec_lr", type=float, default=None)
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs for train')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--valid_size', type=int, default=256, help='size for validation set')   
    parser.add_argument('--lr_decay_interval', type=int, default=50,
                        help='how many epochs to wait before decrease learning rate')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout ratio')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    

    # logging
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='how many steps to wait before logging training status')
    parser.add_argument('--val_interval', type=int, default=1000,
                        help='how many steps to wait before testing')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='how many epochs to wait before saving')
    parser.add_argument('--save_dir', type=str, default='models', help='where to save the snapshot')
    # data
    # parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument("--train_pt_dir", type=str, default='./data-bin/trip/pts/train_pt_dir')
    parser.add_argument("--dev_pt_dir", type=str, default='./data-bin/trip/pts/dev_pt_dir')
    # parser.add_argument('--shuffle', default=False, help='shuffle data every epoch')
    parser.add_argument('--sentence_len', type=int, default=61, help='how many tokens in a sentence')
    # model
    parser.add_argument('--enc_model', type=str, default='bert-large', choices=['bert-large-uncased', 'google/flan-t5-xl', 't5-large', 't5-3b', 'conv','bertconv'], help='encoder model')
    parser.add_argument('--dec_model', type=str, default='gpt2-medium', choices=['gpt2-medium', 'gpt2-large', 'gpt2-xl', 'deconv', 'deconformer'], help='decoder model')
    parser.add_argument('--latent_size', type=int, default=768, help='size of latent variable')
    parser.add_argument('--n_head', type=int, default=16, help='size of latent variable')
    
    parser.add_argument("--load_enc", type=boolean_string, default=True)
    parser.add_argument("--load_dec", type=boolean_string, default=True)
    parser.add_argument("--share_gpts", type=boolean_string, default=True)

    parser.add_argument('--num_feature', type=int, default=1,
                        help='number of feature')
    # conv deconv model
    parser.add_argument('--out_layer', type=str, default='lm_head', choices=['pred_token', 'pred_emb', 'lm_head'], help='deconvolution last layer choice')
    parser.add_argument('--reg_layer', type=str, default='ln', choices=['bn', 'ln', 'none'], help='regularization layer')
    parser.add_argument('--embed_dim', type=int, default=768, help='number of embedding dimension')
    parser.add_argument('--filter_size', type=int, default=600, help='filter size of convolution')
    parser.add_argument('--filter_shape', type=int, default=5,
                        help='filter shape to use for convolution')
    # parser.add_argument('--last_filter_shape', type=int, default=-1,
    #                     help='filter shape to use for last convolution')
    parser.add_argument('--tau', type=float, default=0.01, help='temperature parameter')
    parser.add_argument('--num_layer', type=int, choices=[2, 3, 4, 5, 6], default=3, help='number of layers')
    # option
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='debug', help='experiment name')
    # noise
    parser.add_argument("--noiser", type=str, choices=['bart', 'bert', 'sub', 'none', 'sub_u'], default='none')
    parser.add_argument("--noiser_ratio", type=float, default=0.3, help='sub noise ratio')
    parser.add_argument("--h_noiser", type=str, choices=['normal', 'none', 'vae'], default='none')
    parser.add_argument("--h_noiser_ratio", type=float, default=0.3, help='hidden noise ratio')
    parser.add_argument("--h_tanh", action='store_true', help='hidden maps to [-1,1]')
    parser.add_argument("--ground", action='store_true', help='grounded generation mode')

    # distributed
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--MASTER_ADDR', type=str, default='localhost')
    parser.add_argument('--MASTER_PORT', type=str, default='29505')
    parser.add_argument('--start_rank', type=int, default=0)
    return parser.parse_args()

def main(args, local_rank):
    logging.basicConfig(
                    filename=os.path.join(args.save_dir, "log.txt"),
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)



    if args.world_size == 1 or (dist.get_rank() == 0):
        writer = SummaryWriter(os.path.join(args.save_dir, 'train'))
        writer_eval = SummaryWriter(os.path.join(args.save_dir,'eval')) 
        logger.info("arguments")
        logger.info(args)
    #########################################################################
    # Prepare Model and Optimizer
    ##########################################################################
    set_seed(args.seed)

    if args.h_noiser == 'vae':
        AE_class = VAE
    else:
        AE_class = AutoEncoder

    if args.resume_ckpt is not None:
        model = AE_class.from_pretrained(encoderModels(args), decoderModels(args), '', args)
    else:
        model = AE_class(encoderModels(args), decoderModels(args), args)

    logger.info(f"Encoder Model Params: {model.encoder.model_size}, decoder Model Params: {model.decoder.model_size}")
    model = model.to(device)

    with open(os.path.join(args.save_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    optimizer_grouped_parameters = []

    if args.enc_lr > 0.:
        param_optimizer = list(model.named_enc_parameters())
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']   # no decay for bias and LayerNorm (ln)
        optimizer_grouped_parameters_ = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01, 'lr':args.enc_lr},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr':args.enc_lr}
            
        ]
        optimizer_grouped_parameters = optimizer_grouped_parameters + optimizer_grouped_parameters_
    if args.dec_lr > 0.:
        param_optimizer = list(model.named_dec_parameters())
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']   # no decay for bias and LayerNorm (ln)
        optimizer_grouped_parameters_ = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01, 'lr':args.dec_lr},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr':args.dec_lr}
            
        ]
        optimizer_grouped_parameters = optimizer_grouped_parameters + optimizer_grouped_parameters_
 
    optimizer = Adamax(optimizer_grouped_parameters, args.lr, max_grad_norm=1.0)

    #########################################################################
    # Prepare Data Set
    ##########################################################################
    set_seed(args.seed+local_rank)
    train_dataloader = DistributedBucketingDataLoader(args.train_pt_dir, args.batch_size_per_gpu, args.sentence_len, rank=local_rank, num_replica=args.world_size)
    if args.world_size == 1 or (dist.get_rank() == 0):
        eval_dataloader = DistributedBucketingDataLoader(args.dev_pt_dir, 2*args.batch_size_per_gpu, args.sentence_len, shuffle=False)
    
    
    #########################################################################
    # For evaluation
    ##########################################################################

    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')


    #########################################################################
    # Training !
    ##########################################################################
    noiser = {'bert':noise_bert, 'bart':noise_bart, 'none':noise_none, 'sub': noise_sub, 'sub_u':noise_sub_uniform}[args.noiser](model.encoder.tokenizer, mlm_probability=args.noiser_ratio)
    global_step, step = 0, 0
    tr_loss, tr_correct, tr_tokens = 0., 0., 0
    # logger.info('start training')
    tokenizer_name = model.decoder.tokenizer.__class__.__name__
    for epoch in range(1, args.epochs+1):
        model.train()
        for batch in tqdm.tqdm(train_dataloader, desc=f"training_epoch{epoch}"):
            if args.ground:
                input_ids_enc, input_ids_dec = batch[3], batch[1]
            else:
                input_ids_enc, input_ids_dec = batch[0], batch[1]

            # # A hack for now    
            # if model.encoder.__class__.__name__ != 'BertEncoder':
            #     bpe = BertTokenizer.from_pretrained('bert-base-uncased')
            #     input_text = [bpe.decode([s for s in sent if s not in (101, 102, 0)]) for sent in input_ids_enc]
            #     encoded_input = model.encoder.tokenizer.batch_encode_plus(input_text, pad_to_max_length=True, truncation=True)
            #     input_ids_enc, att_mask_enc = torch.Tensor(encoded_input['input_ids']), torch.Tensor(encoded_input['attention_mask'])


            if global_step == 0:
                model.save(args.save_dir, 'epoch%d-step%d'%(epoch, global_step))

            input_ids_enc = noiser.noise(input_ids_enc)
            input_ids_dec_cls = {'BertTokenizer': input_ids_enc, 'GPT2Tokenizer': input_ids_dec}
            batch = (input_ids_enc, input_ids_dec_cls[tokenizer_name], ) + batch[2:]
            batch = tuple(t.to(device) for t in batch[:3])
            input_ids_enc, input_ids_dec, lm_labels = batch

            loss, correct, ntokens, h = model(input_ids_enc, input_ids_dec=input_ids_dec, lm_labels=lm_labels)
            tr_loss += loss.item() * ntokens
            tr_correct += correct.item()
            tr_tokens += ntokens.item()
            step += 1
            loss.backward()
            if not (step % args.gradient_accumulation_steps == -1 % args.gradient_accumulation_steps):
                continue

            if args.world_size > 1:
                average_gradients(model)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1




            if args.world_size == 1 or (dist.get_rank() == 0):
                # tensorboard
                writer.add_scalar('loss/loss', loss.item() , global_step)
                writer.add_scalar('loss/LR', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('hidden/h_mean', h.mean().item() , global_step)
                writer.add_scalar('hidden/h_std', h.std().item() , global_step)
                writer.add_scalar('hidden/positive_ratio', (h>0).sum()/h.shape.numel(), global_step)

                if global_step % args.val_interval == -1 % args.val_interval:
                    model.eval()
                    eval_loss, eval_ppl, eval_mid_ppl, _, rouge_l, bleu = eval_model_loss(model, gpt_model, gpt_tokenizer, global_step, eval_dataloader, noiser, device, logger, max_valid_size=args.valid_size, onlypart=True, ground=args.ground)



                    noisers = {'bert':noise_bert, 'bart':noise_bart, 'none':noise_none, 'sub': noise_sub, 'sub_u':noise_sub_uniform}
                    noise_ratio = 0.0
                    current_noiser = noisers[args.noiser](model.encoder.tokenizer, mlm_probability=noise_ratio)
                    _, _, _, _, _, clean_bleu = eval_model_loss(model, gpt_model, gpt_tokenizer, global_step, eval_dataloader, current_noiser, device, logger, max_valid_size=args.valid_size, onlypart=True, ground=args.ground)

                    noise_ratio = 0.3
                    current_noiser = noisers[args.noiser](model.encoder.tokenizer, mlm_probability=noise_ratio)
                    _, _, _, _, _, robust_bleu = eval_model_loss(model, gpt_model, gpt_tokenizer, global_step, eval_dataloader, current_noiser, device, logger, max_valid_size=args.valid_size, onlypart=True, ground=args.ground)




                    writer_eval.add_scalar('loss/loss', eval_loss, global_step)
                    writer_eval.add_scalar('metric/int_ppl', eval_mid_ppl, global_step)
                    writer_eval.add_scalar('metric/rouge', rouge_l, global_step)
                    writer_eval.add_scalar('metric/bleu', bleu, global_step)
                    writer_eval.add_scalar('metric/clean_bleu', clean_bleu, global_step)
                    writer_eval.add_scalar('metric/robust_bleu', robust_bleu, global_step)
                    
                    
                    if global_step % args.save_interval == 0 % args.save_interval:
                        model.save(args.save_dir, 'epoch%d-step%d-ppl%.3f-bleu%.1f'%(epoch, global_step, eval_ppl, bleu))
                    model.train()

                if global_step % args.log_interval == -1 % args.log_interval:
                    ppl = torch.exp((tr_loss/tr_tokens).clone().detach()).item()
                    logger.info('Training:')
                    logger.info('Epoch: {}, '
                        'Steps: {}, '
                        'Corr: {}, ' 
                        'Loss: {}, PPL: {}'.format(epoch, global_step, tr_correct/tr_tokens, loss.item(), ppl))
                    
                    rand_id = torch.randint(input_ids_enc.shape[0], (1,))
                    input_sentence = model.decode(input_ids_enc[rand_id,:])
                    input_sentence_corrupted = model.decode(input_ids_enc[rand_id,:])
                    predict_sentence = model.generate_from(h[rand_id,:])
                    
                    logger.info("Input Sentence:")
                    logger.info(input_sentence[0].strip())
                    logger.info("Corrupted Sentence:")
                    logger.info(input_sentence_corrupted[0].strip())
                    logger.info("Output Sentence:")
                    logger.info(predict_sentence[0].strip())
                    
                    
                    tr_loss, tr_correct, tr_tokens = 0., 0., 0
    
   
    if args.world_size == 1 or (dist.get_rank() == 0):
        writer.close()
        writer_eval.close()
    
def init_processes(local_rank, args, backend='nccl'):
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank+local_rank, world_size=args.world_size)
    main(args, local_rank)

if __name__ == "__main__":
    args = parse_args()
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    args.save_dir = os.path.join(args.save_dir, f"{args.exp_name}_{date}")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.enc_lr is None:
        args.enc_lr = args.lr
    if args.dec_lr is None:
        args.dec_lr = args.lr
    args.batch_size_per_gpu = int(args.batch_size / args.world_size / args.gradient_accumulation_steps)
    if args.world_size == 1:
        main(args, 0)
        exit(0)
    mp.spawn(init_processes, args=(args,), nprocs=args.gpus)
