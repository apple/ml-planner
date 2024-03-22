#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import math
import torch
from torch.nn.utils.rnn import pad_sequence
import random

def get_whole_word_mask(tokenizer):
    def is_beginning_of_word(i):
        x = tokenizer.convert_ids_to_tokens(i)
        if x in tokenizer.all_special_tokens:
            return True
        return not x.startswith('##')
    mask_whole_words = torch.ByteTensor(list(
    map(is_beginning_of_word, range(len(tokenizer)))))
    return mask_whole_words

class noise:
    def __init__(self):
        pass

    def _noise(self):
        pass

    def noise(self, tensor, mlm_probability = 0.3):
        # tensor: LongTensor bsz x seq_len
        noised_tensor = []
        to_keep = torch.ne(tensor, 0)
        for sent, keep in zip(tensor.split(1, dim=0), to_keep.split(1, dim=0)):
            noised_tensor.append(self._noise(sent[keep]))
        noised_tensor = pad_sequence(noised_tensor, batch_first=True, padding_value=0)
        return noised_tensor


class noise_bart(noise):
    def __init__(self,
                 tokenizer,
                 mlm_probability=None,
                 poisson_lambda=3.0,#randomly shuffle sentences for this proportion of inputs
                 permute_ratio=0.1, #take this proportion of words and permute them
                 mask_ratio=0.3, #fraction of words/subwords that will be masked
                 random_ratio=0.2, #instead of using [MASK], use random token this often
                 replace_length=1 #when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)
                  ):
        _lambda = poisson_lambda
        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= (k + 1)
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        self.mask_span_distribution = torch.distributions.Categorical(ps)
        self.mask_whole_word = get_whole_word_mask(tokenizer)
        self.permute_ratio = permute_ratio
        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.replace_length = replace_length
        self.mask_idx = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)
        self.tokenizer = tokenizer
    
    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        is_word_end = is_word_start.clone()
        
        is_word_end[:-1] = is_word_start[1:]
        is_word_end[-1] = 0
        is_word_end[-2] = 1
        return is_word_start, is_word_end
    
    def _noise(self, source):
        if not torch.is_tensor(source):
            source = torch.tensor(source, dtype=torch.long)
        is_word_start, is_word_end = self.word_starts(source)
        if self.permute_ratio > 0.0:
            source = self.permute_words(source, is_word_end, self.permute_ratio)

        if self.mask_ratio > 0:
            source = self.add_whole_word_mask(source, is_word_start, self.mask_ratio)
        return source

    def permute_words(self, source, is_word_end, p=1.0):
        result = source.clone()
        is_word_end = is_word_end.bool()
        word_ends = (is_word_end[1:] * ~is_word_end[:-1]).nonzero() + 2
        num_words = word_ends.size(0)
        num_to_permute = math.ceil((num_words * 2 * p) / 2.0)
        substitutions = torch.randperm(num_words)[:num_to_permute]
        ordering = torch.arange(0, num_words)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        # Ignore <bos> at start
        index = 1
        for i in ordering:
            word = source[(word_ends[i - 1] if i > 0 else 1):word_ends[i]]
            result[index:index + word.size(0)] = word
            index += word.size(0)
        return result

    def add_whole_word_mask(self, source, is_word_start, p):
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source


        lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

        # Make sure we have enough to mask
        cum_length = torch.cumsum(lengths, 0)
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat([lengths, self.mask_span_distribution.sample(sample_shape=(num_to_mask,))], dim=0)
            cum_length = torch.cumsum(lengths, 0)

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i + 1
        lengths = lengths[:num_to_mask]

        # Handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.size(0)
        num_to_mask -= num_inserts
        if num_to_mask == 0:
            return self.add_insertion_noise(source, num_inserts / source.size(0))

        assert (lengths > 0).all()

        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero()
        indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[-1] = 255 # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(1, self.vocab_size, size=(mask_random.sum(),))

        assert len(lengths.size()) == 1
        assert lengths.size() == indices.size()
        lengths -= 1
        while indices.size(0) > 0:
            assert lengths.size() == indices.size()
            lengths -= is_word_start[indices + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]
            if self.replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = self.mask_idx
                source[indices[mask_random]] = torch.randint(1, self.vocab_size, size=(mask_random.sum(),))

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(low=1, high=self.vocab_size, size=(num_random,))

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

class noise_bert(noise):
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def _noise(self, inputs):
        """ Prepare masked tokens for masked language modeling: 80% MASK, 20% random. """

        # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.long)
        probability = torch.full(inputs.shape, self.mlm_probability)

        special_tokens_mask = [ 1 if x in [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, 0] else 0 for x in inputs.tolist()]
        probability.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability).bool()
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 20% of the time, we replace masked input tokens with random word
        indices_random = masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs


class noise_sub(noise):
    def __init__(self, tokenizer, mlm_probability=0.3):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
    def _noise(self, inputs ):
        """ Prepare masked tokens for masked language modeling: 30% random. """

        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.long)
        probability = torch.full(inputs.shape, self.mlm_probability)

        special_tokens_mask = [ 1 if x in [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, 0] else 0 for x in inputs.tolist()]
        probability.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability).bool()
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[masked_indices] = random_words[masked_indices]

        return inputs
    

class noise_sub_uniform(noise):
    def __init__(self, tokenizer, mlm_probability=0.3):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
    def _noise(self, inputs ):
        """ Prepare masked tokens for masked language modeling: uniformly sample from [0, mlm_prob] """
        cur_mlm_prob = random.uniform(0, self.mlm_probability)
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.long)
        probability = torch.full(inputs.shape, cur_mlm_prob)

        special_tokens_mask = [ 1 if x in [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, 0] else 0 for x in inputs.tolist()]
        probability.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability).bool()
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[masked_indices] = random_words[masked_indices]

        return inputs


class noise_none(noise):
    def __init__(self, tokenizer, mlm_probability=None):
        self.tokenizer = tokenizer

    def _noise(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.long)
        return inputs
