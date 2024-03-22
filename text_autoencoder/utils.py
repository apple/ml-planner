#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput


INIT_RANGE = 0.02

class MLPLayer(nn.Module):
    def __init__(self, config):
        super(MLPLayer, self).__init__()
        config.layer_norm_eps = 1e-12
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output, hidden_states)
        return layer_output

class MLP(nn.Module):
    def __init__(self, input_size, config):
        super(MLP, self).__init__()
        self.pre_layer = nn.Linear(input_size, config.hidden_size)
        self.layers = nn.ModuleList([MLPLayer(config) for _ in range(config.layers)])
        self.apply(init_bert_weights)

    def forward(self, hidden_states):
        hidden_states = self.pre_layer(hidden_states)
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        normalized = (hidden_states - self.layers[-1].output.LayerNorm.bias) / self.layers[-1].output.LayerNorm.weight
        
        return normalized

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, config):
        super(BinaryClassifier, self).__init__()
        self.pre_layer = nn.Linear(input_size, config.hidden_size)
        self.layers = nn.ModuleList([MLPLayer(config) for _ in range(config.layers)])
        self.post_layer = nn.Linear(config.hidden_size, 2)
        self.apply(init_bert_weights)

    def forward(self, hidden_states):
        hidden_states = self.pre_layer(hidden_states)
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        logits = self.post_layer(hidden_states)
        return logits

class DataLoader:
    def __init__(self, data, batch_size, shuffle=True, rank=0, num_replica=1):
        data = torch.load(data, map_location='cpu')
        print ('load', len(data))
        self.rank = rank
        self.num_replica = num_replica
        self.data = data[self.rank::self.num_replica]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        cur = 0
        if self.shuffle:
            random.shuffle(self.data)
        while cur < len(self.data):
            text = [ x['text'] for x in self.data[cur:cur+self.batch_size]]
            tensor = [ x['tensor'] for x in self.data[cur:cur+self.batch_size]]
            yield {'text':text, 'tensor':tensor}
            cur += self.batch_size

def init_bert_weights(module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=INIT_RANGE)
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def get_indices(tot_len):
    triples = []
    for i in range(tot_len):
        for j in range(i+1, tot_len):
            if j == i + 1:
                triples.append([i, j, -1])
            else:
                triples.append([i, j, (i+j)//2])
    triples = torch.tensor(triples, dtype=torch.long)
    start, end, middle = triples.split(1, dim=-1)
    start, end, middle = start.squeeze().view(-1), end.squeeze().view(-1), middle.squeeze().view(-1)
    not_missing = torch.eq(middle, -1)
    middle.clamp_(min=0)
    return start, end, middle, not_missing

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

class Statistics:
    def __init__(self, key_value_dict=None, **kwargs):
        self.statistics = {'steps':0}
        if key_value_dict is not None:
            for x in key_value_dict:
                self.statistics[x] = key_value_dict[x]
        for x in kwargs:
            self.statistics[x] = kwargs[x]

    def update(self, key_or_dict, value=None):
        if value is None:
            assert isinstance(key_or_dict, dict)
            for key in key_or_dict:
                if key not in self.statistics:
                    self.statistics[key] = 0.
                self.statistics[key] += key_or_dict[key]
        else:
            assert isinstance(key_or_dict, str)
            if key_or_dict not in self.statistics:
                self.statistics[key_or_dict] = 0.
            self.statistics[key_or_dict] += value
    
    def __getitem__(self, attr):
        return self.statistics[attr]

    def step(self):
        self.statistics['steps'] += 1

def slot_statistics(pred, gold, mask):
    """For debert performance measurement,
    pred 0/1 tensor, gold 0/1 tensor, mask bool (true if it is a data point)"""
    pred = pred[mask].float()
    gold = gold[mask].float()
    assert pred.size() == gold.size()
    tot = pred.numel()
    acc = (pred == gold).float().sum().item()
    true_positive = (pred * gold).sum().item()
    p_div_num = pred.sum().item()
    r_div_num = gold.sum().item()
    return {'tot':tot, 'acc':acc, 'true_positive':true_positive, 'p_div_num':p_div_num, 'r_div_num':r_div_num}


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'