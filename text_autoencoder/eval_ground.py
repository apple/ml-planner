#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from autoencoder.autoencoder_utils import eval_gpt2, cal_entropy, calc_self_bleu, calc_bleu, calc_rouge, cal_most_freq
import tqdm
import os
import glob
import argparse
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from bert_score import score
import numpy as np

def eval_ground(args):
    
    if args.ref != 'none':
        with open(args.ref) as f:
            ref_text = [x.strip() for x in f.readlines()]

    eval_files = glob.glob(os.path.join(args.eval, "*.txt"))
    results = []

    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

    for filepath in tqdm.tqdm(eval_files):

        # Read in the suffix text from the current file
        with open(filepath) as f:
            suffix_text = [x.strip() for x in f.readlines()]
            if args.by_len:
                # gen_split = [gpt_tokenizer.tokenize(s) for s in suffix_text]
                gen_split = [s.lstrip().rstrip().lower() for s in suffix_text]
                gen_split = [s.split(" ") for s in gen_split] 
                ent_pos, dist_pos, most_freqs = [], [], []
                for n in range(1,5):
                    pos = [" ".join(s[:n]) for s in gen_split]
                    ent, dist = cal_entropy(pos)
                    most_freqs.append(cal_most_freq(pos)[n-1])
                    ent_pos.append(ent[n-1])
                    dist_pos.append(dist[n-1])

                print(f'{os.path.basename(filepath)[:-4]}\t Dist: {" ".join(map("{:.2f}".format, dist_pos))}\n')
                print(f'{os.path.basename(filepath)[:-4]}\t Ent: {" ".join(map("{:.2f}".format, ent_pos))}\n')
                print("\n")
                for mf in most_freqs:
                    print(', '.join(f'{k}: {v:.2%}' for k, v in mf.items()))
                print("\n")
                continue


            ent, dist = cal_entropy(suffix_text)
            selfbleu = calc_self_bleu(suffix_text[:1000])
            # selfbleu = 0.0

            if args.ref != 'none':
                _, _, rouge_l = calc_rouge(ref_text, suffix_text)
                bleu = calc_bleu(ref_text, suffix_text)
                _, _, F1 = score(suffix_text, ref_text, lang='en', verbose=True, model_type="microsoft/deberta-xlarge-mnli")
                bertscore = F1.mean().cpu().numpy()

 

            tot_nll, tot_gpt_tokens = 0,0
            for g in suffix_text:
                nll, n_gpt_tokens = eval_gpt2(gpt_model, gpt_tokenizer, g)
                tot_nll += nll.item() * n_gpt_tokens
                tot_gpt_tokens += n_gpt_tokens
            nll_avg = tot_nll/tot_gpt_tokens
            ppl = torch.exp(torch.tensor(nll_avg))
            
            length = np.mean([len(s.split(" ")) for s in suffix_text])
  
            if args.ref != 'none':
                relevance = [f"{bleu:.2f}", f"{rouge_l:.3f}", f"{bertscore:.2f}"]
            else:
                relevance = []
            results.append([os.path.basename(filepath)[:-4], f"{ppl:.3f}", f"{dist[0]:.2f}/{ent[0]:.3f}", f"{selfbleu:.2f}", f"{1-dist[3]:.2%}"] + relevance + [f"{length:.2f}"])


    if args.ref != 'none':
        relevance = ["BLEU"] + ["ROUGE"] + ["BertScore"]
    else:
        relevance = []

    df = pd.DataFrame(results, columns=["file"] + ["PPL"] + ["Diversity"] + ["Self_bleu"] + ["Repetition"] + relevance + ["len"] )
    df = df.sort_values('file')
    df.to_csv(os.path.join(args.eval, "summary.tsv"), sep="\t")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="none", help="ref file path")
    parser.add_argument("--eval", help="Evaluation folder path")
    parser.add_argument("--by_len", action = 'store_true')
    args = parser.parse_args()
    eval_ground(args)