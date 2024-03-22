#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from autoencoder.autoencoder_utils import DistributedBucketingDataLoader, eval_gpt2, cal_entropy, calc_bleu, calc_self_bleu
from interpolation import parse_args, load_model
from autoencoder.noiser import *
import os
import numpy as np
import tqdm
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from text_autoencoder.baseline.classifier_text import BERTClassifier
import glob
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset



def gen_from_repr(args):
    args, model, _ = load_model(args, suffix="gen.txt")
    model.cuda()
    model.eval()
    device = torch.device("cuda")
    # noisers = {'bert':noise_bert, 'bart':noise_bart, 'none':noise_none, 'sub': noise_sub, 'sub_u':noise_sub_uniform}
    task_id = args.task_id if args.task_id else args.load_ckpt
    print(task_id)
    print(args.repr_file)
    args.repr_file = glob.glob(args.repr_file)[0]
    with open(args.repr_file, "rb") as f:
        reprs = np.load(f)['arr_0']  
    if args.data_transformation:
        reprs = np.arctanh(reprs)
    reprs = torch.from_numpy(reprs[:,:,:,:args.num_feature,:])
    text = []
    if args.sample_size:
        reprs = reprs[:,:args.sample_size,:,:,:]
    batch_size = 256
    num_batches = (reprs.shape[1] - 1)  // batch_size + 1

    with torch.no_grad():
        for r in tqdm.tqdm(reprs, desc=f"gen from repr"):
            batches = [r[i*batch_size : (i+1)*batch_size,:,:,:] for i in range(num_batches)]
            gen_text = []
            for b in batches:
                input_hidden = b.view(-1, args.num_feature, reprs.shape[-1]).to(device)
                # input_hidden = torch.tanh(input_hidden)
                gen_text.extend(model.generate_from(input_hidden.permute(0,2,1), beam_width=args.beam_size, top_k = args.top_k, sample=(args.top_k>1), skip_special_tokens = args.by_len))
            text.append(gen_text)  # 6, 768, 16

    print(f"Total {len(r)} samples, {len(reprs)} steps")
    text = np.array(text).transpose() # 250*6
    gen = defaultdict(list)
    if args.forward:
        num_sample = len(r) #5000 #len(r)
        if args.dev_pt_dir.endswith(".txt"):
            with open(args.dev_pt_dir, "r") as f:
                test_set = f.readlines()
        else: # eval pt
            args.dev_pt_dir = os.path.dirname(args.dev_pt_dir)
            eval_dataloader = DistributedBucketingDataLoader(args.dev_pt_dir, args.batch_size_per_gpu, args.sentence_len, shuffle=False)
            test_set = []
            total_size = 0
            labels = []
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
                input_sentence = model.decode(input_ids_dec, tokenizer='dec')
                if args.remove_dash_n:
                    input_sentence = [re.sub('\n', ' ', s) for s in input_sentence]
                test_set.extend(input_sentence)
                total_size += batch[0].shape[0]
                if total_size >= num_sample:
                    break
        test_set = test_set[:num_sample]
        #### write to file ####
        # with open(os.path.join(args.dev_pt_dir, 'review.txt'), "w") as f:
        #     for t, c in zip(test_set, labels):
        #         f.write(f"Prefix:\n {c}\nSuffix:\n {t}\n")
        #     exit()
    filename = args.repr_file[:-4] + f".top{args.top_k}.txt" if args.beam_size == -1 else args.repr_file[:-4] + f".beam{args.beam_size}.txt"
    with open(filename, "w") as f:
        for i, t in enumerate(text):
            if args.forward and args.dev_pt_dir:
                f.write(f"Sample {i}: {test_set[i]}\n")
            else:
                f.write(f"Sample {i}:\n")
            for j, s in enumerate(t):
                if args.remove_dash_n:
                    s=re.sub('\n', ' ', s)
                if args.forward:
                    f.write(f"Time {j/(len(reprs)-1):.2f}: {s}\n")
                else:
                    f.write(f"Time {1-(j+1)/len(reprs):.2f}: {s}\n")
                gen[j].append(s)
            f.write("\n")

        # eval PPL under GPT2
        gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')


        if args.forward:
            new_gen = defaultdict(list)
            # if '[UNK]' not in t
            new_test_set = [t for t in test_set if len(t.rstrip().strip())>=10]
            for j in range(len(reprs)):
                for i, g in enumerate(gen[j]):
                    if len(test_set[i].rstrip().strip())<10: continue
                    new_gen[j].append(g)
            test_set = new_test_set
            gen = new_gen

            for j in range(len(reprs)):
                tot_nll, tot_gpt_tokens = 0,0
                assert (len(gen[j])==len(test_set))
                for i, g in enumerate(gen[j]):
                    nll, n_gpt_tokens = eval_gpt2(gpt_model, gpt_tokenizer, g)
                    tot_nll += nll.item() * n_gpt_tokens
                    tot_gpt_tokens += n_gpt_tokens
                nll_avg = tot_nll/tot_gpt_tokens
                ppl = torch.exp(torch.tensor(nll_avg))
                f.write(f"PPL at t={j/(len(reprs)-1):.2f}: {ppl}\n")
            f.write("\n")
            print(f"{len(test_set)} samples after filtering")
            if args.dev_pt_dir:
                alpha_sq = np.loadtxt(args.repr_file[:-4] + ".alpha_sq.csv")
                bleu_all = []
                for j in range(len(reprs)):
                    bleu = calc_bleu([g for g in gen[j]], test_set)
                    bleu_all.append(bleu)
                    f.write(f"BLEU at t={j/(len(reprs)-1):.2f}: {bleu}\n")
                f.write(f"AuBLEU (by step): {sum(bleu_all)/len(reprs)}\n")
                def auroc(pairs):
                    sorted_pairs = sorted(pairs, key=lambda x: x[0])
                    auroc = 0
                    for i in range(1, len(sorted_pairs)):
                        x1, y1 = sorted_pairs[i-1]
                        x2, y2 = sorted_pairs[i]
                        height_avg = (y2 + y1) / 2
                        auroc += height_avg * (x2 - x1)
                    return auroc
                au_bleu = auroc(zip(alpha_sq, bleu_all))

                f.write(f"AuBLEU (by alpha_sq): {au_bleu}\n")

        else:
            # sample
            new_gen = defaultdict(list)
            for j in range(len(reprs)):
                for g in gen[j]:
                    if len(g.strip())==0: continue
                    new_gen[j].append(g)
            gen = new_gen
            if args.by_len:
                gen_split = [gpt_tokenizer.tokenize(s) for s in gen[len(reprs)-1]]
                from itertools import zip_longest
                padding_value = "PAD"
                padded_gen_split = [list(x) for x in zip_longest(*gen_split, fillvalue=padding_value)]
                ent_pos, dist_pos = [], []
                for pos in padded_gen_split:
                    ent, dist = cal_entropy(pos)
                    ent_pos.append(ent[0])
                    dist_pos.append(dist[0])
                f.write(f'Diversity over Len: Dist: {" ".join(map("{:.2f}".format, dist_pos))}\n')
                f.write(f'Entropy over Len: Ent: {" ".join(map("{:.2f}".format, ent_pos))}\n')
                f.write("\n")
  


            for j in range(len(reprs)):
                ent, dist = cal_entropy(gen[j])
                f.write(f'Diversity at t={1-(j+1)/len(reprs):.2f}: Dist: {" ".join(map("{:.2f}".format, dist))}, Ent {" ".join(map("{:.2f}".format, ent))}\n')
            
            # self_bleu = calc_self_bleu(gen[len(reprs)-1])
            f.write(f'Diversity: {dist[0]:.2f}/{ent[0]:.2f}/{1-dist[3]:.2f}\n')
            f.write("\n")
            tot_nll, tot_gpt_tokens = 0,0
            # import pdb;pdb.set_trace()
            for g in gen[len(reprs)-1]:
                nll, n_gpt_tokens = eval_gpt2(gpt_model, gpt_tokenizer, g)
                tot_nll += nll.item() * n_gpt_tokens
                tot_gpt_tokens += n_gpt_tokens
            nll_avg = tot_nll/tot_gpt_tokens
            ppl = torch.exp(torch.tensor(nll_avg))
            f.write(f"Gen PPL: {ppl:.3f}\n")
            f.write("\n")

            if args.cond is not None and args.cond != -1:
                # eval acc via BertClassier
                classifier = BERTClassifier.from_pretrained("", args.classifier_path)
                classifier = classifier.cuda()
                classifier.eval()
                label = torch.LongTensor([args.cond]*len(gen[len(reprs)-1]))
                inputs = classifier.tokenizer.batch_encode_plus(gen[len(reprs)-1], pad_to_max_length=True)
                input_ids = torch.LongTensor(inputs['input_ids'])
                dataset = TensorDataset(input_ids, label)
                dataloader = DataLoader(dataset, batch_size=32)
                correct = 0
                total_predictions = []
                for batch in dataloader:
                
                    batch = tuple(t.to(device) for t in batch)

                    _, corr, prediction = classifier(batch[0], batch[1])
                    correct += corr
                    total_predictions.extend(prediction.tolist())

                f.write(f"Gen ACC: {correct / len(gen[len(reprs) - 1])}\n")
                f.write(f"Predictions: {total_predictions}\n")


if __name__ == "__main__":
    parser = parse_args()
    parser.add_argument("--repr_file", type=str, default=None)
    parser.add_argument("--data_transformation", action = 'store_true')
    parser.add_argument('--sample_size', type=int, default=None, help='sample size')
    parser.add_argument("--remove_dash_n", action = 'store_true')
    parser.add_argument("--forward", action = 'store_true')
    parser.add_argument("--classifier_path", type=str, default="models/classifier/epoch50-step664999-acc0.962")
    parser.add_argument('--cond', type=int, default=None, help='conditional value')
    parser.add_argument('--beam_size', type=int, default=-1, help='sample size')  
    parser.add_argument("--top_k", type = int, default = 1)  
    parser.add_argument("--by_len", action = 'store_true')
    parser.add_argument("--cond_feature", action = 'store_true')
    args = parser.parse_args()
    # for fname in test_models:
    gen_from_repr(args)