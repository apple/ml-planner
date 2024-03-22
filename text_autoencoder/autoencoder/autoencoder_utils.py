#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import os
import torch 
import json
import math
import numpy as np
import random
from collections import defaultdict
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, GPT2Tokenizer
from sumeval.metrics.rouge import RougeCalculator
from sumeval.metrics.bleu import BLEUCalculator
from transformers import MarianMTModel, MarianTokenizer, T5ForConditionalGeneration, T5Tokenizer, BertModel, BertTokenizer

from generation_utils import EOS_ID
import tqdm


class Feature:
    def __init__(self, bert_ids, gpt2_ids, raw_text, cond=None):
        self.input_ids_bert = bert_ids
        self.input_ids_dec = [EOS_ID] + gpt2_ids
        self.lm_labels = gpt2_ids + [EOS_ID]
        if cond is not None:
            self.cond = cond

class BucketSampler(Sampler):
    """
    this sampler will sort data by sequence length
    """
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
class FeatureDataset(Dataset):
    """ pytorch dataset for GPT2 training """

    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        feat_dict = self.features[i]
        feat = Feature(**feat_dict)
        return feat

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features):
        input_ids_bert = pad_sequence([torch.tensor(f.input_ids_bert)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        input_ids_dec = pad_sequence([torch.tensor(f.input_ids_dec, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=0)
        lm_labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)
        if not hasattr(features[0], 'cond'):
            cond = [None for f in features]
        else:
            if isinstance(features[0].cond, int) or isinstance(features[0].cond, str):
                cond = [f.cond for f in features]
            else: #cont feature
                cond = pad_sequence([torch.tensor(f.cond)
                               for f in features],
                              batch_first=True, padding_value=0)

        return (input_ids_bert, input_ids_dec, lm_labels, cond)


class BucketingDataLoader(object):
    """ this loads pt chunks and then convert to mini-batch loader"""
    def __init__(self, pt_name, batch_size, max_seq_len,
                 bucket=100, shuffle=True,
                 rank=0, num_replica=1):

        self.pt_name = pt_name
        self.batch_size = batch_size
        self.max_len = max_seq_len
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.rank = rank
        self.num_replica = num_replica


    def __iter__(self):
        chunk = torch.load(self.pt_name)[self.rank::self.num_replica]
        # discard long examples
        trunc_chunk = []
        lens = []
        total = len(chunk)
        for feat in chunk:
            if len(feat['gpt2_ids'])+2 > tokenizer_gpt2.max_len_single_sentence or len(feat['bert_ids']) > tokenizer_bert.max_len_single_sentence:
                continue
            tot_len = len(feat['gpt2_ids'])
            if tot_len > self.max_len:
                feat['gpt2_ids'] = feat['gpt2_ids'][:self.max_len]
            if len(feat['bert_ids']) > self.max_len:
                feat['bert_ids'] = feat['bert_ids'][:self.max_len]
            trunc_chunk.append(feat)
            lens.append(tot_len)
        print (f"{self.pt_name}: rank {self.rank} has chunks {len(trunc_chunk)}/{total}, batch_size: {self.batch_size}")
        print (f"{self.pt_name}: rank {self.rank} has chunks {len(trunc_chunk)}/{total}, total iterations: {len(trunc_chunk)//self.batch_size}")
        dataset = FeatureDataset(trunc_chunk)
        sampler = BucketSampler(lens, self.bucket_size, self.batch_size,
                                droplast=True, shuffle=self.shuffle)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=0,  # can test multi-worker
                            collate_fn=FeatureDataset.collate)
        yield from loader

    def __len__(self):
        raise NotImplementedError()

class DistributedBucketingDataLoader(object):
    """ distributed version """
    def __init__(self, db_dir, *args, **kwargs):
        self.db_dir = db_dir
        self.args = args
        self.kwargs = kwargs

    def _get_files(self):
        files = [os.path.join(self.db_dir, fname) for fname in os.listdir(self.db_dir) if fname.endswith('.pt')]
        files.sort()
        if not ('shuffle' in self.kwargs and self.kwargs['shuffle'] == False):
            random.shuffle(files)
        return files

    def __iter__(self):
        for db_name in self._get_files():
            loader = BucketingDataLoader(db_name, *self.args, **self.kwargs)
            yield from loader

class InfiniteDistributedBucketingDataLoader(DistributedBucketingDataLoader):
    def __init__(self, db_dir, *args, **kwargs):
        super().__init__(db_dir, *args, **kwargs)
        self.iterator = iter(self)

    def __iter__(self):
        while True:
            for db_name in self._get_files():
                loader = BucketingDataLoader(db_name, *self.args, **self.kwargs)
                yield from loader

    def __next__(self):
        while True:
            try:
                return next(self.iterator)
            except StopIteration:
                self.iterator = iter(self)

class TextDataset(Dataset):
    def __init__(self, prefix_path, tokenizer, max_length=16, device=torch.device("cpu")):
        # Load the prefixes from a file
        with open(prefix_path, "r") as f:
            prefixes = [line.strip() for line in f]
        self.input_ids = [tokenizer.encode(prefix, padding='max_length', max_length=max_length, return_tensors="pt").to(device) for prefix in prefixes]

    def __getitem__(self, idx):
        return 0, 0, 0, self.input_ids[idx]

    def __len__(self):
        return len(self.input_ids)

class TextDataLoader(object):
    def __init__(self, prefix_path, tokenizer, batch_size, max_length=256, device=torch.device("cpu")):
        self.dataset = TextDataset(prefix_path, tokenizer, max_length, device)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

    def __iter__(self):
        return iter(self.loader)


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    rep_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score


def cal_most_freq(generated, k=5):
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    top_k_counter = [defaultdict(int), defaultdict(int),
                    defaultdict(int), defaultdict(int)]
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        sorted_counter = sorted(counter[n].items(), key=lambda item: item[1], reverse=True)
        for key, value in sorted_counter[:k]:
            top_k_counter[n][key] = float(value/total)
    return top_k_counter

def eval_model_loss(model, gpt_model, gpt_tokenizer, ae_step, eval_dataloader, noiser, device, logger, max_valid_size = None, onlypart=False, ground=False):
    if onlypart:
        print ('WARNING! Only part of the dev data evaluated!')
    tot_loss, tot_correct, tot_tokens = 0., 0, 0
    tot_gpt_tokens = 0
    tot_nll_mid = 0
    input_sentence, input_sentence_corrupted, predict_sentence = [], [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_dataloader, desc=f"validation_epoch{ae_step}"):
            if ground:
                input_ids_bert, input_ids_dec = batch[3], batch[1]
            else:
                input_ids_bert, input_ids_dec = batch[0], batch[1]
            input_ids_enc = noiser.noise(input_ids_bert)
            tokenizer_name = model.decoder.tokenizer.__class__.__name__
            input_ids_dec_cls = {'BertTokenizer': input_ids_bert, 'GPT2Tokenizer': input_ids_dec}
            batch = (input_ids_enc, input_ids_dec_cls[tokenizer_name], ) + batch[2:]
            batch = tuple(t.to(device) for t in batch[:3])
            input_ids_enc, input_ids_dec, lm_labels = batch

            loss, correct, ntokens, h = model(input_ids_enc, input_ids_dec, lm_labels)
            tot_loss += loss.item() * ntokens
            tot_correct += correct.item()
            tot_tokens += ntokens.item()

            h_all = model.encoder_mean(input_ids_bert[:2,:].to(device))
            h_0, h_N = h_all[0], h_all[1]
            h_mid = (h_0+h_N)/2
            resp = model.generate_from(h_mid.unsqueeze(0))[0]
            nll_mid, n_gpt_tokens = eval_gpt2(gpt_model, gpt_tokenizer, resp)
            tot_nll_mid += nll_mid.item() * n_gpt_tokens
            tot_gpt_tokens += n_gpt_tokens


            if tot_tokens > 128 * 256 * 1024 and onlypart:
                break

            input_sentence += model.decode(input_ids_bert)
            input_sentence_corrupted += model.decode(input_ids_enc)
            predict_sentence += [x.lower() for x in model.generate_from(h)]
            if max_valid_size and len(input_sentence) >= max_valid_size:
                break
                    

        loss = tot_loss/tot_tokens
        nll_mid_avg = tot_nll_mid/tot_gpt_tokens
        ppl = torch.exp(torch.tensor(loss))
        mid_ppl = torch.exp(torch.tensor(nll_mid_avg))
        acc = tot_correct / tot_tokens
        input_sentence = [t.strip() for t in input_sentence]
        predict_sentence = [t.strip() for t in predict_sentence]



        _, _, rouge_l = calc_rouge(input_sentence, predict_sentence)
        bleu = calc_bleu(input_sentence, predict_sentence)
        # self_bleu = calc_self_bleu(predict_sentence)
    batch_size = input_ids_bert.shape[0]
    logger.info('Validation:')
    logger.info('Steps: {}, '
                'Loss: {}, '
                'PPL: {}, '
                'Acc: {}, '
                'Int_PPL: {}, '
                'Rouge: {}, '
                'Robust BLEU: {}, '.format(ae_step, loss.item(), ppl.item(), acc, mid_ppl.item(), rouge_l, bleu))
    
    rand_id = torch.randint(batch_size, (1,))[0]
    logger.info("Input Sentence:")
    logger.info(input_sentence[rand_id].strip())
    logger.info("Corrupted Sentence:")
    logger.info(input_sentence_corrupted[rand_id].strip())
    logger.info("Output Sentence:")
    logger.info(predict_sentence[rand_id].strip())

        
    return loss.item(), ppl.item(), mid_ppl.item(), acc, rouge_l, bleu

def load_cond_model(model):
    if model is None or model == "None":
        return None
    elif model.startswith('Helsinki-NLP'):
        assert model in ['Helsinki-NLP/opus-mt-zh-en', 'Helsinki-NLP/opus-mt-en-de']
        cond_model = MarianMTModel.from_pretrained(model).encoder
        cond_tokenizer = MarianTokenizer.from_pretrained(model)
    elif model.startswith("t5"):
        # check if this is a valid t5 model. 
        assert model in ['t5-small', 't5-base', 't5-large']
        cond_model = T5ForConditionalGeneration.from_pretrained(model).encoder
        cond_tokenizer = T5Tokenizer.from_pretrained(model)
    elif model.startswith("bert"):
        assert model in ['bert-base-uncased', 'bert-medium-uncased', 'bert-large-uncased']
        cond_model = BertModel.from_pretrained(model)
        cond_tokenizer = BertTokenizer.from_pretrained(model)
    else:
        raise NotImplementedError

    return cond_model, cond_tokenizer

def generate_hidden(model, dataloader, noiser, device, cond_model = None, max_size = None, ground = False, h_noiser='none'):
    if max_size==-1:
        max_size = None
    hiddens = []
    cond = []
    cond_model = load_cond_model(cond_model)
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc=f"generate hidden"):
            if ground:
                input_ids_bert = batch[3]
            else:
                input_ids_bert = batch[0]
            input_ids_enc = noiser.noise(input_ids_bert)
            input_ids_enc = input_ids_enc.to(device)
            # if gpu:
            if h_noiser == 'vae':  
                mean = model.encoder_mean(input_ids_enc)
                log_var = model.encoder_log_var(input_ids_enc)
                sampled_h = model.reparameterize(mean, log_var)
                hiddens.extend([h for h in sampled_h])
            else:
                hiddens.extend([h for h in model.encoder_mean(input_ids_enc)])
            # else:
            #     hiddens.extend([h.cpu().numpy().astype(np.float16) for h in model.encoder_mean(input_ids_enc)])
            if len(batch)>3:
                cond_feature = (not isinstance(batch[3], list)) and (not any(np.array(batch[3].view(-1)) == None)) and batch[3].ndim > 1
                if cond_feature:
                    input_ids_bert_cond = noiser.noise(batch[3]).to(device)
                    if cond_model is None:
                        cond.extend([c.view(c.shape[0], -1) for c in model.encoder_mean(input_ids_bert_cond)])
                    else:
                        cond.extend([c.view(c.shape[0], -1) for c in cond_model(input_ids_bert_cond).last_hidden_state])
                else:
                    cond.extend(batch[3])
            if max_size and len(hiddens) >= max_size:
                break        
    return hiddens, cond


def calc_rouge(original_sentences, predict_sentences, default= "None"):
    rouge_1 = 0.0
    rouge_2 = 0.0
    rouge_l = 0.0
    num_sample = len(original_sentences)
    for original, predict in zip(original_sentences, predict_sentences):
        # Remove padding
        original, predict = original.replace("<PAD>", "").strip(), predict.replace("<PAD>", "").strip()
        if original == "": original = default
        if predict == "": predict = default
        rouge = RougeCalculator(stopwords=True, lang="en")
        r1 = rouge.rouge_1(summary=predict, references=original)
        r2 = rouge.rouge_2(summary=predict, references=original)
        rl = rouge.rouge_l(summary=predict, references=original)
        rouge_1 += r1
        rouge_2 += r2
        rouge_l += rl
    return rouge_1/num_sample, rouge_2/num_sample, rouge_l/num_sample

def calc_bleu(original_sentences, predict_sentences, default= "None"):
    bleu = 0.0
    num_sample = len(original_sentences)
    for original, predict in zip(original_sentences, predict_sentences):
        # Remove padding
        
        original, predict = original.replace("<PAD>", "").strip(), predict.replace("<PAD>", "").strip()
        if original == "": original = default
        if predict == "": predict = default
        b = BLEUCalculator(lang="en").bleu(summary=predict, references=original)
        bleu += b
    return bleu/num_sample

def calc_self_bleu(predict_sentences, default= "None"):
    bleu = 0.0
    cnt = 0
    for i, p1 in enumerate(predict_sentences):
        for p2 in predict_sentences[i+1:]:
            # Remove padding
            if p2 == "": p2 = default
            if p1 == "": p1 = default

            b = BLEUCalculator(lang="en").bleu(summary=p1.replace("<PAD>", "").strip(), references=p2.replace("<PAD>", "").strip())
            bleu += b
            cnt += 1
    return bleu/cnt

def compute_last_kernel_shape(args):
    t = args.sentence_len + 2 * (args.filter_shape - 1)
    for _ in range(args.num_layer-1):
        t = int(math.floor((t - args.filter_shape) / 2) + 1)
    return t

    # last_kernel_shape = compute_last_kernel_shape(args)

def eval_gpt2(model, tokenizer, text, max_len = 512):
    model = model.cuda()
    if len(text) == 0:
        text = "No text"
    encoded_input = tokenizer(text, return_tensors='pt')
    # encoded_input = tokenizer.batch_encode_plus(text, padding=True, truncation=True)
    input_ids = encoded_input.input_ids
    # input_ids = torch.LongTensor(input_ids)
    input_ids = input_ids[:,:max_len].cuda()
    n_token = input_ids.size(1)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return outputs[0], n_token