#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import numpy as np
import scipy.io as sio
from math import floor
import torch


def add_noise(sents, args):
    if not args.noise_type:
        return sents
    else:
        sents = sents.cpu().numpy()
        num_corrupted = floor(args.noise_ratio * args.sentence_len)
        if args.noise_type == 's':
            sents_permutated= substitute_sent(sents, num_corrupted, args)
        elif args.noise_type == 'pw':
            sents_permutated= permutate_sent(sents, num_corrupted)
        elif args.noise_type == 'ps':
            sents_permutated= permutate_sent_whole(sents)
        elif args.noise_type == 'a':
            sents_permutated= add_sent(sents, num_corrupted, args)   
        elif args.noise_type == 'd':
            sents_permutated= delete_sent(sents, num_corrupted) 
        elif args.noise_type == 'm':
            sents_permutated= mixed_noise_sent(sents, num_corrupted, args)
        else:
            raise NotImplementedError
        
        return torch.LongTensor(sents_permutated).cuda()


def permutate_sent(sents, num_corrupted):
    # permutate the words in sentence
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        idx_s= np.random.choice(len(sent_temp)-1, size=num_corrupted, replace=True)
        temp = sent_temp[idx_s[0]]
        for ii in range(num_corrupted-1):
            sent_temp[idx_s[ii]] = sent_temp[idx_s[ii+1]]
        sent_temp[idx_s[num_corrupted-1]] = temp
        sents_p.append(sent_temp)
    return sents_p  

def permutate_sent_whole(sents):
    # permutate the whole sentence
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        sents_p.append(np.random.permutation(sent_temp))
    return sents_p  
    
    
def substitute_sent(sents, num_corrupted, args):
    # substitute single word 
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        idx_s= np.random.choice(len(sent_temp)-1, size=num_corrupted, replace=True)   
        for ii in range(num_corrupted):
            sent_temp[idx_s[ii]] = np.random.choice(args.v)
        sents_p.append(sent_temp)
    return sents_p       

def delete_sent(sents, num_corrupted):
    # substitute single word 
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        idx_s= np.random.choice(len(sent_temp)-1, size=num_corrupted, replace=True)   
        for ii in range(num_corrupted):
            sent_temp[idx_s[ii]] = -1
        sents_p.append([s for s in sent_temp if s!=-1])
    return sents_p 
    
def add_sent(sents, num_corrupted, args):
    # substitute single word 
    sents_p = []
    for ss in range(len(sents)):
        sent_temp = sents[ss][:]
        idx_s= np.random.choice(len(sent_temp)-1, size=num_corrupted, replace=True)   
        for ii in range(num_corrupted):
            sent_temp.insert(idx_s[ii], np.random.choice(args.v))
        sents_p.append(sent_temp[:args.sentence_len])
    return sents_p  


def mixed_noise_sent(sents, num_corrupted, args):
    sents = delete_sent(sents, num_corrupted)
    sents = add_sent(sents, num_corrupted, args)
    sents = substitute_sent(sents, num_corrupted, args)
    return sents