#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

'''
 * @Author: Yizhe Zhang 
 * @Date: 2019-04-05 16:50:50 
 * @Last Modified by:   Yizhe Zhang 
 * @Last Modified time: 2019-04-05 16:50:50 
'''
import torch
from tqdm import trange
import torch.nn.functional as F
import numpy as np
import logging
import pdb
# GPT
EOS_ID = 50256
# Bert
SEP_ID = 102
PAD_ID= 0
# T5
PAD_ID_T5 = 0
SEP_ID_T5 = 1
def prepare_for_bleu(sentence, tokenizer, skip_special_tokens = False):
    sent=[]
    tokenizer_name = tokenizer.__class__.__name__
    if skip_special_tokens:
        end_of_sentence = {'BertTokenizer': [], 'GPT2Tokenizer': [], 'T5Tokenizer': []}
    else:
        end_of_sentence = {'BertTokenizer': [SEP_ID, PAD_ID], 'GPT2Tokenizer': [EOS_ID], 'T5Tokenizer': [SEP_ID_T5, PAD_ID_T5],}
    for s in sentence[1:]:
        if s not in end_of_sentence[tokenizer_name]:
            sent.append(s)
        else:
            break
    return sent


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)



def generate_next_token(model_gpt, prev, temperature=1, top_k = 0, top_p=1.0, sample=False, past=None):

    with torch.no_grad():
        #pdb.set_trace()
        gpt_output = model_gpt.transformer(prev, position_ids=None, token_type_ids=None, past_key_values=past)
        hidden_states, past = gpt_output['last_hidden_state'], gpt_output['past_key_values']
        logits = model_gpt.lm_head(hidden_states)
        logits = logits[:, -1, :] / temperature
        if top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        else:
            logits = top_k_logits(logits, k=top_k)
        probs = F.softmax(logits, dim=-1)


        if sample:
            prev = torch.multinomial(probs, num_samples=1)
            return prev, probs[0][prev], past
        else:
            probs_sel, prev = torch.topk(probs, k=top_k, dim=-1)
            return prev, probs_sel, past

###########################################################################
# Beam search based on ottokart/beam_search
###########################################################################
class Node(object):
    def __init__(self, parent, state, value, cost):
        super(Node, self).__init__()
        self.value = value
        self.parent = parent  # parent Node, None for root
        self.state = state
        self.length = 1 if parent is None else parent.length + 1
        self.cum_cost = parent.cum_cost*(self.length-1)/self.length + cost/self.length if parent else cost
        # self.cum_cost = parent.cum_cost + cost if parent else cost
        self._sequence = None

    # def __repr__(self):
    #    return f'value = {self.value}, parent = {self.parent.value}, cost = {self.cum_cost}'

def beam_search_naive(model_gpt, bs, length=48, beam_width=3, beam_examples=1, past=None):
    """
    currently it does NOT support batch parallel
    """

    all_decode, all_decode_losses = [], []
    for b in range(bs):
        next_fringe = [Node(parent=None, state=past, value=EOS_ID, cost=0.0)]
        results = []
        for i in range(length):
            fringe, all_prev, all_probs, all_past = [], torch.Tensor(0).long().cuda(), [], []
            for nn in next_fringe:
                if (nn.value == EOS_ID) and (i>0):
                    results.append(nn)
                    continue
                else:
                    fringe.extend([nn]*beam_width)

                if not fringe:
                    break

                prev, probs, past = generate_next_token(model_gpt, torch.Tensor([[nn.value]]).long().cuda(), temperature=1, top_k=beam_width, sample=False, past=nn.state)
                # pdb.set_trace()

                log_probs = torch.log(probs)[0]
                all_prev = torch.cat((all_prev, prev[0]))
                all_probs.extend(log_probs.tolist())
                all_past.extend([past]*len(log_probs))


            next_fringe = []
            for prev, log_probs, past, nn in zip(all_prev, all_probs, all_past, fringe):
                new_node = Node(parent=nn, state=past, value=prev.item(), cost=log_probs)
                next_fringe.append(new_node)

            next_fringe = sorted(next_fringe, key=lambda nn: nn.cum_cost, reverse=True)[:beam_width]

        results.extend(next_fringe)

        results.sort(key=lambda nn : nn.cum_cost, reverse=True)

        if beam_examples == 1:
            # Single response version
            best_result = results[0].parent
            decode, decode_loss = [], []
            while best_result.value != EOS_ID:
                decode.append(best_result.value)
                decode_loss.append(best_result.cum_cost)
                best_result = best_result.parent
            decode.append(best_result.value)
            decode_loss.append(best_result.cum_cost)
            decode, decode_loss = decode[::-1], decode_loss[::-1]
            all_decode.append(decode)
            all_decode_losses.append(decode_loss)
        else:
            # Top beam_n_examples 
            best_results = results[:beam_examples]
            sent_all_decode, sent_all_decode_losses = [],[]
            for best_result in best_results:
                decode, decode_loss = [], []
                while best_result.value != -1:
                    decode.append(best_result.value)
                    decode_loss.append(best_result.cum_cost)
                    best_result = best_result.parent
                decode, decode_loss = decode[::-1], decode_loss[::-1]
                sent_all_decode.append(decode)
                sent_all_decode_losses.append(decode_loss)
            all_decode.append(sent_all_decode)
            all_decode_losses.append(sent_all_decode_losses)

    if beam_examples == 1:
        output = torch.nn.utils.rnn.pad_sequence([torch.tensor(f, dtype=torch.long) for f in all_decode], batch_first=True, padding_value=EOS_ID)
    else:
        output = torch.nn.utils.rnn.pad_sequence([torch.tensor(s, dtype=torch.long) for s in all_decode[0]], batch_first=True, padding_value=EOS_ID)

    return output




def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits



def generate_sequence(model, temperature=1, top_k=1, top_p = 1.0, length=20, sample=False, past=None, device='cuda'):
    output = past[0][0].new_zeros([past[0][0].size(0),0])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    bsz = past[0][0].size(0)
    prev = torch.Tensor([EOS_ID]*bsz).long().cuda().unsqueeze(1)
    output = torch.cat((output, prev), dim=1)
    for i in range(length):
        prev, probs, past = generate_next_token(model, prev, temperature=temperature, top_k=top_k, top_p=top_p, sample=sample, past=past)
        output = torch.cat((output, prev), dim=1)
    return output
