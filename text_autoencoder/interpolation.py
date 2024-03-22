#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from autoencoder.autoencoder import AutoEncoder,VAE, encoderModels, decoderModels
import torch
import argparse
import logging
import os, sys
import glob
import re
import json
import tqdm
from utils import boolean_string
from autoencoder.autoencoder_utils import DistributedBucketingDataLoader, eval_gpt2

def parse_args():
    parser = argparse.ArgumentParser(description='text auto-encoder model')
    parser.add_argument("--train_pt_dir", type=str, default='./data-bin/trip/pts/train_pt_dir')
    parser.add_argument("--dev_pt_dir", type=str, default='./data-bin/trip/pts/dev_pt_dir')
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--task_id", type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='models', help='where to save the snapshot')
    parser.add_argument('--batch_size_per_gpu', type=int, default=16, help='batch size for training')
    parser.add_argument('--ae_step', type=int, default=100, help='iteration number of the AE')

    parser.add_argument('--lower', action='store_true', help='lowercase the generation')
    parser.add_argument('--user_input', action='store_true', help='user given input')

    
    parser.add_argument("--share_gpts", type=boolean_string, default=True)
    parser.add_argument("--enc_model", type=str, default='bert-base-uncased')
    parser.add_argument("--dec_model", type=str, default='gpt2')
    parser.add_argument("--load_enc", type=boolean_string, default=True)
    parser.add_argument("--load_dec", type=boolean_string, default=True)

    parser.add_argument('--valid_size', type=int, default=128, help='size for validation set')   
    parser.add_argument('--max_size', type=int, default=None, help='size for training set') 
    parser.add_argument('--sentence_len', type=int, default=61, help='how many tokens in a sentence')    
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout ratio')
    parser.add_argument('--latent_size', type=int, default=768, help='size of latent variable')
    parser.add_argument('--num_feature', type=int, default=16,
                        help='number of feature')
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

    parser.add_argument("--noiser", type=str, choices=['bart', 'bert', 'sub', 'none'], default='sub')
    parser.add_argument("--noiser_ratio", type=float, default=0.3, help='sub noise ratio')
    parser.add_argument("--h_noiser", type=str, choices=['normal', 'none', 'vae'], default='none')
    parser.add_argument("--h_noiser_ratio", type=float, default=0.3, help='hidden noise ratio')
    parser.add_argument("--h_tanh", action='store_true', help='hidden maps to [-1,1]')    
    parser.add_argument("--ground", action='store_true', help='grounded generation mode')
    return parser



def load_model(args, load_ae_model=True , suffix = "log.txt"):
    # args.save_dir = os.path.join(args.save_dir, f"{args.exp_name}_{date}")
    args.save_dir = args.load_ckpt if args.load_ckpt else os.path.join(args.save_dir, args.task_id)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logging.basicConfig(
                    filename=os.path.join(args.save_dir, suffix),
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    if args.load_ckpt:
        logger.info(f"load from checkpoint {args.load_ckpt}")
    else:
        raise Exception

    files = glob.glob(f"{args.load_ckpt}/*.pkl")
    model_prefix = re.sub(r'(.*)-[A-Z|0-9]+.pkl', r'\1' ,files[0])
    exception_list = ["valid_size","batch_size_per_gpu","save_dir","sentence_len","train_pt_dir","dev_pt_dir"]
    try:    
        with open(f"{args.load_ckpt}/commandline_args.txt", 'r') as f:
            # args.__dict__ = json.load(f)
            load_dict = json.load(f)
            for k in args.__dict__.keys():
                if k in load_dict and k not in exception_list:
                    args.__dict__[k] = load_dict[k]
            # todo
        logger.info(f"load from saved config: {args}")
    except:
        model_names = set([re.sub(r'.*-([A-Z|0-9]+).pkl', r'\1' ,f) for f in files])
        enc_name_dict = {'CNN':'conv', 'BERT':'bert', 'BERTCNN':'bertconv'}
        dec_name_dict = {'DCNN':'deconv', 'GPT2':'gpt2', 'DCF':'deconformer'}
        for k,v in enc_name_dict.items():
            if k in model_names:
                args.enc_model = v
        for k,v in dec_name_dict.items():
            if k in model_names:
                args.dec_model = v
        logger.info("load from input config")
    args.resume_ckpt = model_prefix
    model = None
    if load_ae_model:
        if args.h_noiser == 'vae':
            AE_class = VAE
        else:
            AE_class = AutoEncoder
        model = AE_class.from_pretrained(encoderModels(args), decoderModels(args), '', args)
        model = model.cuda()
        model.eval()
    return args, model, logger

def interpolate(args):
    args, model, logger = load_model(args)

    test_set = []
    if args.user_input:
        sentence_1 = input('sentence 1')
        sentence_2 = input('sentence 2')
        test_set.append([sentence_1, sentence_2])


    eval_dataloader = DistributedBucketingDataLoader(args.dev_pt_dir, 2, args.sentence_len, shuffle=False)
    for batch in eval_dataloader:
        input_ids_bert = batch[0].cuda()
        input_sentence = model.decode(input_ids_bert)
        test_set.append(input_sentence)
        if args.valid_size and len(test_set) >= args.valid_size:
            break

    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

    task_id = args.task_id if args.task_id else args.load_ckpt
    logger.info('='*35+task_id+'='*35)
    cnt = 0
    only_ppl = False
    total_nll, total_token = 0.0, 0
    total_nll_data, total_token_data = 0.0, 0
    for sentence_1, sentence_2 in tqdm.tqdm(test_set, desc=f"interpolation"):
        if cnt > 5: only_ppl = True
        cnt += 1
        nll, n_token = _interpolate(model, logger, gpt_model, gpt_tokenizer, sentence_1, sentence_2, only_ppl, args.lower)
        total_nll += nll * n_token
        total_token += n_token
        for s in [sentence_1, sentence_2]:
            nll_data, n_token_data = eval_gpt2(gpt_model, gpt_tokenizer, s)
            total_nll_data += nll_data * n_token_data
            total_token_data += n_token_data       

    logger.info(f"average data PPL: {torch.exp(total_nll_data/total_token_data)}")
    logger.info(f"average interpolation PPL: {torch.exp(total_nll/total_token)}")


def _interpolate(model, logger, gpt_model, gpt_tokenizer, sentence_1, sentence_2, only_ppl= False, lower = False):

    h_0 = model.encode(sentence_1)
    h_N = model.encode(sentence_2)
    if not only_ppl:
        logger.info('-'*80)
        logger.info('-'*80)
        logger.info(sentence_1)
        logger.info('-'*80)
        num_of_samples = 5
        with torch.no_grad():
            for i in range(num_of_samples+1):
                w_N = i/float(num_of_samples)
                w_0 = 1 - w_N
                history = h_0*w_0+h_N*w_N
                resp = model.generate_from(history)[0]
                if lower: resp = resp.lower()
                logger.info(f"{i}:{resp}")
        logger.info('-'*80)
        logger.info(sentence_2)
    history = (h_0+h_N)/2
    resp = model.generate_from(history)[0]
    if resp == '': 
        print("Warning, generated response is an empty string!", file=sys.stderr)
        resp = "None"
    if lower: resp = resp.lower()
    nll, n_token = eval_gpt2(gpt_model, gpt_tokenizer, resp)
    return nll, n_token

 

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    # for fname in test_models:
    interpolate(args)