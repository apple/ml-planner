#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse
import torch
import logging
import json
import os
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import BertModel, BertTokenizer, BertConfig, BertLayer, T5ForConditionalGeneration, T5Config, T5Tokenizer, T5EncoderModel, T5Model
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from generation_utils import beam_search_naive, prepare_for_bleu, generate_sequence
import math
from os.path import join
import copy
import itertools
import tqdm

# current_path = os.path.dirname(os.path.abspath(__file__))

def print_chkpt_info(loading_info, chkpt_state_dict = None, model = None):
    missing_keys, unexpected_keys, mismatched_keys = loading_info["missing_keys"], loading_info["unexpected_keys"], loading_info["mismatched_keys"]

    if chkpt_state_dict is not None and model is not None:
        # Get all keys from the state dictionary
        all_state_dict_keys = set(chkpt_state_dict.keys())

        # Get all keys from the model's state dictionary
        all_model_keys = set(model.state_dict().keys())

        # Properly loaded keys are the intersection of model keys and state dictionary keys
        properly_loaded_keys = all_state_dict_keys.intersection(all_model_keys)

        print("Properly loaded:", properly_loaded_keys)

    # Any missing or unexpected keys indicate a problem
    if missing_keys:
        print("Warning: Missing keys in state_dict:", missing_keys)

    if unexpected_keys:
        print("Warning: Unexpected keys in state_dict:", unexpected_keys)

    if mismatched_keys:
        print("Warning: Mismatched keys in state_dict:", mismatched_keys)

    # If there are no missing and unexpected keys, the weights are properly loaded.
    if not missing_keys and not unexpected_keys and not mismatched_keys:
        print("All weights are properly loaded.")

class ConvModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        self.sentence_len = args.sentence_len
        self.padding = lambda x, filter_s: nn.ConstantPad1d((filter_s-1, self.sentence_len-x.shape[1]+filter_s-1), self.tokenizer.pad_token_id)(x)

    def _BN(self, shape):
        return nn.BatchNorm2d(shape)

    def _LN(self, shape, device = 'cuda'):
        # nasty way to save encoder checkpoint
        # global _permute
        f = nn.LayerNorm(shape, device = device) 
        def _permute(x):
            return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return _permute

    def _InitWeight(self, module):        
        # weight initialize for conv_transpose layer
        for m in self.modules():
            if isinstance(m, module):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def load(self, state_dict):
        self.load_state_dict(state_dict)

    def save(self, output_dir, prefix):
        torch.save(self.state_dict(), os.path.join(output_dir, f"{prefix}-{self.file_name}"))

    @classmethod
    def from_pretrained(cls, encoder, input_dir, prefix, *unused):
        ckpt = torch.load(os.path.join(input_dir, f"{prefix}-{cls.file_name}"), map_location='cpu')
        encoder['model_args']['state_dict'] = ckpt
        return encoder

    @classmethod
    def compute_last_filter_shape(cls, args):
        t = args.sentence_len + 2 * (args.filter_shape - 1)
        for _ in range(args.num_layer-1):
            t = int(math.floor((t - args.filter_shape) / 2) + 1)
        return t - args.num_feature * 2 + 2



class ConvolutionEncoder(ConvModel):
    file_name = 'CNN.pkl'
    def __init__(self, args, state_dict = None):
        super().__init__(args)
        last_filter_shape = ConvModel.compute_last_filter_shape(args)
        # assert(args.num_feature == final_len)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embed = nn.Embedding(self.tokenizer.vocab_size, args.embed_dim)
        self.filter_shape = args.filter_shape
        self.file_name = 'CNN.pkl'
        
        embed_size = self.embed.weight.size()[1]
        if args.reg_layer == 'none':
            self.reg_layer = nn.ModuleList([nn.Identity() for l in range(args.num_layer)])
        elif args.reg_layer == 'bn':
            self.reg_layer = nn.ModuleList([self._BN(embed_size)] + [self._BN(args.filter_size * (2 ** l)) for l in range(0,args.num_layer-1)])
        elif args.reg_layer == 'ln':
            self.reg_layer = [self._LN(embed_size)] + [self._LN(args.filter_size * (2 ** l)) for l in range(0,args.num_layer-1)]
        else:
            raise NotImplementedError

        conv_shapes = [(embed_size, args.filter_size, args.filter_shape, 1)]
        conv_shapes += [(args.filter_size * (2 ** l), args.filter_size * (2 ** (l+1)), args.filter_shape, 1) for l in range(0,args.num_layer-2)]
        conv_shapes += [(args.filter_size * (2 ** (args.num_layer-2)), args.latent_size, last_filter_shape, 1)]
        self.conv_layer = nn.ModuleList([self._CONV(*conv_shapes[l]) for l in range(0,args.num_layer)])

        self.model_size = sum(t.numel() for t in self.parameters())

        if state_dict:
            self.load(state_dict)
        else:
            self._InitWeight(nn.Conv2d)

    def _CONV(self, *shape):
        return nn.Conv2d(shape[0], shape[1], (shape[2], shape[3]), stride=2)
    
    def forward(self, x):
        x = self.padding(x[:, :self.sentence_len], self.filter_shape)
        x = self.embed(x)
        # x.size() is (L, emb_dim) if batch_size is 1.
        # So interpolate x's dimension if batch_size is 1.
        if len(x.size()) < 3:
            x = x.view(1, *x.size())
        # reshape for convolution layer
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
        x = self.dropout(x)
        # N 1 L emb => N emb L 1
        h = x.transpose_(1, 3)
        for l in range(len(self.reg_layer)):
            h = self.conv_layer[l](self.dropout(F.relu(self.reg_layer[l](h))))

        return h.squeeze(-1) #[bsz, latent_size, final_len]
        
    
class DeconvolutionDecoder(ConvModel):
    file_name = 'DCNN.pkl'
    def __init__(self, args, state_dict = None):
        super().__init__(args)
        last_filter_shape = ConvModel.compute_last_filter_shape(args)
        # assert(args.num_feature == final_len)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embed = nn.Embedding(self.tokenizer.vocab_size, args.embed_dim)
        self.tau = args.tau
        self.filter_shape = args.filter_shape
        self.out_layer = args.out_layer
        self.file_name = 'DCNN.pkl'
        if args.reg_layer == 'none':
            self.reg_layer = nn.ModuleList([nn.Identity() for l in range(args.num_layer)])
        elif args.reg_layer == 'bn':
            self.reg_layer = nn.ModuleList([self._BN(args.latent_size)] + [self._BN(args.filter_size * (2 ** (l-1))) for l in range(args.num_layer-1, 0, -1)])
        elif args.reg_layer == 'ln':
            self.reg_layer = [self._LN(args.latent_size)] + [self._LN(args.filter_size * (2 ** (l-1))) for l in range(args.num_layer-1, 0, -1)]
        else:
            raise NotImplementedError        
        deconv_shapes = [(args.latent_size, args.filter_size * (2 ** (args.num_layer-2)), last_filter_shape, 1)]
        deconv_shapes += [(args.filter_size * (2 ** l), args.filter_size * (2 ** (l-1)), args.filter_shape, 1) for l in range(args.num_layer-2, 0, -1)]
        
        ## last layer
        if args.out_layer == 'pred_emb':
            deconv_shapes += [(args.filter_size , self.embed.weight.size()[1], args.filter_shape, 1)]
        elif args.out_layer == 'pred_token':
            deconv_shapes += [(args.filter_size , self.embed.weight.size()[0], args.filter_shape, 1)]
        elif args.out_layer == 'lm_head':
            lm_head_dim = self.embed.weight.size()[1]
            config = BertConfig.from_pretrained('bert-base-uncased')
            config.hidden_size = lm_head_dim
            self.lm_head = BertLMPredictionHead(config)
            self.final_ln = nn.LayerNorm(lm_head_dim)
            deconv_shapes += [(args.filter_size, lm_head_dim, args.filter_shape, 1)]
        else:
            raise NotImplementedError


        self.deconv_layer = nn.ModuleList([self._DECONV(*deconv_shapes[l]) for l in range(0,args.num_layer)])

        self.model_size = sum(t.numel() for t in self.parameters())

        if state_dict:
            self.load(state_dict)
        else:
            self._InitWeight(nn.ConvTranspose2d)

    def _DECONV(self, *shape):
        return nn.ConvTranspose2d(shape[0], shape[1], (shape[2], shape[3]), stride=2)

    def forward(self, hidden_state, input_ids_dec=None):
        h = self.dropout(hidden_state)
        log_prob = self.compute_prob(h)
        input_ids_dec = self.padding(input_ids_dec[:, :self.sentence_len], 1)
        assert(log_prob.shape[1] == input_ids_dec.shape[1])
        loss = [F.nll_loss(sentence_emb_matrix, word_ids, size_average=False) for sentence_emb_matrix, word_ids in zip(log_prob, input_ids_dec)]
        average_loss = sum([torch.sum(l) for l in loss]) / log_prob.size()[0]/ input_ids_dec.shape[1]

        # total = torch.ne(input_ids_dec, -1).float().sum()
        # Not comparable with GPT2 decoder, as the pad_tokens are calculated in total loss
        total = torch.ne(input_ids_dec, -1).float().sum()
        correct = (log_prob.max(dim=-1)[1] == input_ids_dec).sum()

        return average_loss, correct, total, hidden_state

    def compute_prob(self, hidden_state):
        h = hidden_state.unsqueeze(-1) #[bsz, latent_size, final_len, 1]
        for l in range(len(self.reg_layer)):
            h = self.deconv_layer[l](self.dropout(F.relu(self.reg_layer[l](h))))
        x_hat = h.transpose_(1, 3).squeeze()
        return self.compute_logp(x_hat)


    def compute_logp(self, x_hat):
        # x.size() is (L, emb_dim) if batch_size is 1.
        # So interpolate x's dimension if batch_size is 1.
        if len(x_hat.size()) < 3:
            x_hat = x_hat.view(1, *x_hat.size())
        #[bsz, L, emb_dim]

        ######   orginal implementation  ######    
        if self.out_layer == 'pred_emb':
            # normalize
            norm_x_hat = torch.norm(x_hat, 2, dim=2, keepdim=True)
            rec_x_hat = x_hat / norm_x_hat
            # compute probability
            w = Variable(self.embed.weight.data).t()
            norm_w = w/torch.norm(w, 2, dim=0, keepdim=True)
            cos_sim = torch.einsum("ble,ev->blv", rec_x_hat, norm_w) / self.tau
            log_prob = F.log_softmax(cos_sim, dim=2)
        elif self.out_layer == 'pred_token':
            log_prob = F.log_softmax(x_hat, dim=2)
        elif self.out_layer == 'lm_head':
            x_hat = self.lm_head(self.final_ln(x_hat))  #[bsz, L, emb_dim] => [bsz, L, vocab_dim]
            log_prob = F.log_softmax(x_hat, dim=2)
        else:
            raise NotImplementedError

        log_prob = log_prob[:,(self.filter_shape-1):-(self.filter_shape -1),:]

        return log_prob

    def generate_from(self, hidden_state):
        log_prob = self.compute_prob(hidden_state)
        out = log_prob.max(dim=-1)[1]
        out = out.tolist()
        gen = [self.tokenizer.decode(prepare_for_bleu(s, self.tokenizer)) for s in out]
        resps = [g.encode('ascii','ignore').decode('ascii') for g in gen]
        return resps
      
    



class DeconformerDecoder(DeconvolutionDecoder):
    file_name = 'DCF.pkl'
    def __init__(self, args, state_dict = None):
        super().__init__(args)
        self.reg_layer2 = copy.deepcopy(self.reg_layer)
        self.file_name = 'DCF.pkl'
        config = BertConfig.from_pretrained('bert-base-uncased')
        configs = {}
        bert_modules = []
        for l in range(args.num_layer):
            configs[l] = copy.deepcopy(config)
            configs[l].hidden_size = args.latent_size if l==0 else args.filter_size * (2 ** (args.num_layer-l-1))    
            bert_modules.append(BertLayer(configs[l])) 
        self.bert_layer = nn.ModuleList(bert_modules)

        if state_dict:
            self.load(state_dict)
        self.model_size = sum(t.numel() for t in self.parameters())


    def compute_prob(self, hidden_state):
        h = hidden_state.unsqueeze(-1) #[bsz, latent_size, final_len, 1]
        for l in range(len(self.reg_layer)):
            # BERT block
            h = self.reg_layer2[l](h)
            h = h.squeeze(-1).permute(0, 2, 1)
            h = self.bert_layer[l](h)[0]
            h = self.dropout(h)
            h = h.permute(0, 2, 1).unsqueeze(-1)
            # Deconv block
            # h = self.deconv_layer[l](self.dropout(F.relu(self.reg_layer[l](h))))   #[bsz, latent_size, cur_len, 1]
            h = self.deconv_layer[l](self.dropout(F.relu(h)))   #[bsz, latent_size, cur_len, 1] This sometimes leads to unstability issue
        x_hat = h.transpose_(1, 3).squeeze()
        return self.compute_logp(x_hat)
    

class BertEncoder(nn.Module):
    def __init__(self, args, model_enc=None, hidden_size = None):
        super().__init__()
        self.name = args.enc_model
        if not hidden_size: hidden_size = args.latent_size
        self.model_enc = model_enc
        self.tokenizer = BertTokenizer.from_pretrained(self.name)
        self.num_feature = args.num_feature
        if model_enc is None:
            if hasattr(args, 'load_enc') and args.load_enc:
                # load pretrained bert 
                self.model_enc = BertModel.from_pretrained(self.name)
                args.latent_size = self.model_enc.config.hidden_size
            else: 
                # from scratch
                config = BertConfig.from_pretrained(self.name)
                config.hidden_size = hidden_size          
                self.model_enc = BertModel(config)  

        self.model_size = sum(t.numel() for t in self.model_enc.parameters())
    
    def forward(self, input_ids_bert=None, attention_mask = None):
        if attention_mask == None:
            attention_mask =torch.ne(input_ids_bert, 0)
        encoded_output = self.model_enc(input_ids_bert, attention_mask)
        hidden_state = encoded_output['last_hidden_state'][:, :self.num_feature, :]
        return hidden_state.permute(0, 2, 1) # bsz x latent x feature num

    def named_parameters(self):
        return self.model_enc.named_parameters()

    def save(self, output_dir, prefix):
        torch.save(self.model_enc.state_dict(), os.path.join(output_dir, prefix+'-BERT.pkl'))

    def from_pretrained(encoder, input_dir, prefix, name = 'bert-base-uncased'):
        model_enc = BertModel.from_pretrained(name, state_dict=torch.load(os.path.join(input_dir, prefix+'-BERT.pkl'), map_location='cpu'))
        encoder['model_args']['model_enc'] = model_enc 
        return encoder





class T5Encoder(nn.Module):
    def __init__(self, args, model_enc=None, hidden_size = None):
        super().__init__()
        self.name = args.enc_model
        if not hidden_size: hidden_size = args.latent_size
        self.model_enc = model_enc
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.t5_tokenizer = T5Tokenizer.from_pretrained(self.name)
        self.num_feature = args.num_feature
        if model_enc is None:
            if hasattr(args, 'load_enc') and args.load_enc:
                # load pretrained t5
                self.model_enc, loading_info = T5EncoderModel.from_pretrained(self.name, output_loading_info = True)
                print_chkpt_info(loading_info)
                args.latent_size = self.model_enc.config.hidden_size
            else: 
                # from scratch
                config = T5Config.from_pretrained(self.name)
                config.hidden_size = hidden_size          
                self.model_enc = T5EncoderModel(config)

        self.model_size = sum(t.numel() for t in self.model_enc.parameters())
    
    def forward(self, input_ids=None):
        text_batch = [self.tokenizer.decode(t, skip_special_tokens=True) for t in input_ids] # BERT tokenizer
        tokenized = self.t5_tokenizer.batch_encode_plus(
            text_batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        input_ids_t5, att_msk_t5 = tokenized.input_ids.to(input_ids.device), tokenized.attention_mask.to(input_ids.device)
        encoded_output = self.model_enc(input_ids_t5, attention_mask= att_msk_t5)
        hidden_state = encoded_output['last_hidden_state'][:, :self.num_feature, :]
        # hidden_state = hidden_state * self.mw + self.mb # TODO: check
        # torch.mean(hidden_state, -1), torch.std(hidden_state, -1)
        return hidden_state.permute(0, 2, 1) # bsz x latent x feature num

    def named_parameters(self):
        return self.model_enc.named_parameters()

    def save(self, output_dir, prefix):
        torch.save(self.model_enc.state_dict(), os.path.join(output_dir, prefix+'-T5_enc.pkl'))

    def from_pretrained(encoder, input_dir, prefix, name = 't5-large'):
        chkpt_state_dict = torch.load(os.path.join(input_dir, prefix+'-T5_enc.pkl'), map_location='cpu')
        model_enc, loading_info = T5EncoderModel.from_pretrained(name, state_dict = chkpt_state_dict, output_loading_info = True)

        print_chkpt_info(loading_info, chkpt_state_dict, model_enc)

        encoder['model_args']['model_enc'] = model_enc 
        return encoder

## Does not work for now. 
class T5Decoder(nn.Module):
    def __init__(self, args, model_dec=None, hidden_size=None):
        super().__init__()
        self.name = args.dec_model
        if not hidden_size: hidden_size = args.latent_size
        self.model_dec = model_dec
        self.tokenizer = T5Tokenizer.from_pretrained(self.name)
        self.num_feature = args.num_feature
        if model_dec is None:
            if hasattr(args, 'load_dec') and args.load_dec:
                # load pretrained t5
                self.model_dec = T5Model.from_pretrained(self.name)
                args.latent_size = self.model_dec.config.hidden_size
            else: 
                # from scratch
                config = T5Config.from_pretrained(self.name)
                config.hidden_size = hidden_size          
                self.model_dec = T5Model(config)
                self.model_size = sum(t.numel() for t in self.model_dec.parameters())
    
    def forward(self, input_ids_t5, att_msk_t5):
        output = self.model_dec(input_ids=input_ids_t5, attention_mask=att_msk_t5)
        return output
    
    def named_parameters(self):
        return self.model_dec.named_parameters()

    def save(self, output_dir, prefix):
        torch.save(self.model_dec.state_dict(), os.path.join(output_dir, prefix+'-T5_dec.pkl'))
    
    @staticmethod
    def from_pretrained(decoder, input_dir, prefix, name='t5-large'):
        model_dec = T5Model.from_pretrained(name, state_dict=torch.load(os.path.join(input_dir, prefix+'-T5_dec.pkl'), map_location='cpu'))
        decoder['model_args']['model_dec'] = model_dec 
        return decoder

class BertConvEncoder(BertEncoder):
    def __init__(self, args, state_dict = None, model_enc=None, hidden_size=768):
        super().__init__(args, model_enc=model_enc, hidden_size=hidden_size)
        last_filter_shape = args.sentence_len - 2*args.num_feature + 2
        self.final_layer = nn.Conv2d(hidden_size, args.latent_size, (last_filter_shape, 1), stride=2)
        self.model_size = sum(t.numel() for t in self.model_enc.parameters())
        self.padding = lambda x: nn.ConstantPad1d((0, args.sentence_len-x.shape[2]), self.tokenizer.pad_token_id)(x)
        if state_dict:
            self.load_state_dict(state_dict)
    
    def forward(self, input_ids_bert=None):
        encoded_output = self.model_enc(input_ids_bert, attention_mask=torch.ne(input_ids_bert, 0))
        bert_output = encoded_output['last_hidden_state']
        bert_output = F.relu(bert_output)

        bert_output = bert_output.permute(0, 2, 1)
        bert_output = self.padding(bert_output)
        bert_output = bert_output.unsqueeze(-1)
        hidden_state = self.final_layer(bert_output)
        hidden_state = hidden_state.squeeze(-1)
        return hidden_state # bsz x latent x feature num


    def save(self, output_dir, prefix):
        torch.save(self.state_dict(), os.path.join(output_dir, prefix+'-BERTCNN.pkl'))

    def from_pretrained(encoder, input_dir, prefix, *unused):
        enc_ckpt = torch.load(os.path.join(input_dir, prefix+'-BERTCNN.pkl'), map_location='cpu')
        encoder['model_args']['state_dict'] = enc_ckpt
        return encoder

class GPT2Decoder(nn.Module):
    def __init__(self, args, model_gpt=None, model_pre=None):
        super().__init__()
        self.name = args.dec_model
        self.model_gpt = model_gpt
        self.model_pre = model_pre
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.name) 
        self.num_feature = args.num_feature
        self.max_len = args.sentence_len
        if model_gpt is None:
            if hasattr(args, 'load_dec') and args.load_dec:
                # load pretrained bert 
                self.model_gpt = GPT2LMHeadModel.from_pretrained(self.name)
                config = GPT2Config.from_pretrained(self.name)
                if config.n_embd == args.latent_size:
                    self.embd_adapter = nn.Identity()
                else:
                    self.embd_adapter = nn.Linear(args.latent_size, config.n_embd) 
            else: 
                # from scratch
                config = GPT2Config.from_pretrained(self.name)
                config.n_embd = args.latent_size
                config.n_head = args.n_head               
                self.model_gpt = GPT2LMHeadModel(config) 
        if model_pre is None:
            if args.share_gpts:
                self.model_pre = self.model_gpt
                self.model_size = sum(t.numel() for t in self.model_gpt.parameters()) 
                return
            else:
                config = GPT2Config.from_pretrained(self.name)
                config.n_embd = args.latent_size           
                self.model_pre = GPT2LMHeadModel(config)
        self.model_size = sum(t.numel() for t in self.model_gpt.parameters()) + sum(t.numel() for t in self.model_pre.parameters())

    def forward(self, hidden_state, input_ids_dec=None, lm_labels=None):
        # TODO: if input length is shorter than num_feature, there will be issue with BERT-DECONV 
        # assert(hidden_state.shape[2] == self.num_feature)  
        # bsz x latent x feature num

        hidden_state = hidden_state.permute(0, 2, 1).contiguous()
        hidden_state = self.embd_adapter(hidden_state)
        hidden_state = hidden_state.permute(0, 2, 1).contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        context = self.model_pre(inputs_embeds=hidden_state.permute(0, 2, 1))[-1] #list
        lm_logits = self.model_gpt(input_ids=input_ids_dec, past_key_values=context)[0]
        bsz, seq_len, vocab_size = lm_logits.size()
        loss = loss_fct(lm_logits.view(-1, vocab_size), lm_labels.view(-1))
        loss = loss.view(bsz, seq_len)
        total = torch.ne(lm_labels, -1).float().sum()
        loss = torch.sum(loss) / total
        correct = (lm_logits.max(dim=-1)[1] == lm_labels).sum()

        # if hidden_state is None:
        #     correct, total = correct.item(), total.item()
        return loss, correct, total, hidden_state

    def save(self, output_dir, prefix):
        torch.save(self.model_gpt.state_dict(), os.path.join(output_dir, prefix+'-GPT2.pkl'))
        torch.save(self.model_pre.state_dict(), os.path.join(output_dir, prefix+'-PRE.pkl'))

    
    def from_pretrained(decoder, input_dir, prefix, name = 'gpt'):
        model_gpt = GPT2LMHeadModel.from_pretrained(name, state_dict=torch.load(os.path.join(input_dir, prefix+'-GPT2.pkl'), map_location='cpu'))
        model_pre = GPT2LMHeadModel.from_pretrained(name, state_dict=torch.load(os.path.join(input_dir, prefix+'-PRE.pkl'), map_location='cpu'))
        decoder['model_args']['model_gpt'] = model_gpt
        decoder['model_args']['model_pre'] = model_pre
        return decoder

    def named_parameters(self):
        return list(self.model_pre.named_parameters()) + list(self.model_gpt.named_parameters())

    def generate_from(self, hidden_states, sample=False, beam_width = -1, top_k = 1, skip_special_tokens = False):
        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        hidden_states = self.embd_adapter(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1).contiguous()
        with torch.no_grad():
            if beam_width == -1:
                #greedy/sample
                hidden_states = hidden_states.permute(0, 2, 1)

                batch_size = 64
                num_batches = (hidden_states.shape[0] - 1)  // batch_size + 1
                batches = [hidden_states[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]
                resps = []
                for b in tqdm.tqdm(batches):
                    context = self.model_pre(inputs_embeds=b)[-1]
                    out = generate_sequence(self.model_gpt, temperature=1, top_k=top_k, length=self.max_len, sample=sample, past=context, device='cuda')
                    out = out.tolist()
                    gen = [self.tokenizer.decode(prepare_for_bleu(s, self.tokenizer, skip_special_tokens = skip_special_tokens), skip_special_tokens = skip_special_tokens) for s in out]
                    resps.extend([g.encode('ascii','ignore').decode('ascii') for g in gen])
            else:
                # beam
                resps = []
                for hidden_state in hidden_states.permute(0, 2, 1):
                    # hidden_state: 1 x dim
                    context = self.model_pre(inputs_embeds=hidden_state)[-1]
                    out = beam_search_naive(self.model_gpt, 1, length=self.max_len+10, beam_width=beam_width, beam_examples=1, past=context)
                    out = out.tolist()
                    gen = [self.tokenizer.decode(prepare_for_bleu(s, self.tokenizer, skip_special_tokens = skip_special_tokens), skip_special_tokens = skip_special_tokens) for s in out]
                    resps.append(gen[-1].encode('ascii','ignore').decode('ascii'))
        return resps



        
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, args = None):
        # E.g. encoder = {'model_cls':'BertDecoder','model_args':{'encoder':encoder_object} }
        super().__init__()
        self.encoder = encoder['model_cls'](**encoder['model_args'])
        self.decoder = decoder['model_cls'](**decoder['model_args'])
        self.h_noiser = args.h_noiser
        self.h_noiser_ratio = args.h_noiser_ratio
        self.h_tanh = args.h_tanh 


    def forward(self, input_ids_enc, input_ids_dec=None, lm_labels=None):
        hidden_state = self.encoder_mean(input_ids_enc)
        if self.h_noiser == 'normal':
            hidden_state = hidden_state + self.h_noiser_ratio*torch.randn_like(hidden_state)
        elif self.h_noiser == 'none':
            hidden_state = hidden_state
        else:
            NotImplementedError
        if isinstance(self.decoder, GPT2Decoder):
            return self.decoder(hidden_state, input_ids_dec=input_ids_dec, lm_labels=lm_labels)
        elif isinstance(self.decoder, DeconvolutionDecoder):
            return self.decoder(hidden_state, input_ids_dec=input_ids_dec)
        else:
            NotImplementedError

    def encoder_mean(self, input_ids_enc, *kargs):
        hidden_state = self.encoder(input_ids_enc, *kargs)
        if self.h_tanh:
            hidden_state = torch.tanh(hidden_state)
        return hidden_state

    def save(self, output_dir, prefix):
        self.encoder.save(output_dir, prefix)
        self.decoder.save(output_dir, prefix)
              
    @classmethod
    def from_pretrained(cls, encoder, decoder, input_dir, args):
        prefix = args.resume_ckpt
        encoder_new = encoder['model_cls'].from_pretrained(encoder, input_dir, prefix, name=args.enc_model)
        decoder_new = decoder['model_cls'].from_pretrained(decoder, input_dir, prefix, name=args.dec_model)
        model = cls(encoder_new, decoder_new, args)
        return model

    def named_enc_parameters(self):
        return self.encoder.named_parameters()

    def named_dec_parameters(self):
        return self.decoder.named_parameters()

    # def named_pretrained_parameters(self):
    #     return list(self.model_enc.named_parameters()) + list(self.model_gpt.named_parameters())

    def generate_from(self, *kargs, **kwargs):
        return self.decoder.generate_from(*kargs, **kwargs)

    # def generate_from_beam(self, *kargs):
    #     return self.decoder.generate_from_beam(*kargs)

    def decode(self, outs, tokenizer = 'enc'):
        resps = []
        self.tokenizers = {'enc': self.encoder.tokenizer, 'dec': self.decoder.tokenizer}
        for out in outs:
            out = out.tolist()
            gen = self.tokenizers[tokenizer].decode(prepare_for_bleu(out, tokenizer=self.tokenizers[tokenizer]))
            resps.append(gen.encode('ascii','ignore').decode('ascii'))
        return resps

    def encode(self, text):
        input_ids = self.encoder.tokenizer.encode(text)
        input_ids = torch.tensor([input_ids]).cuda()
        return self.encoder_mean(input_ids)

    # def encode_batch(self, *kargs):    
    #     return self.encoder.encode(*kargs)

class VAE(AutoEncoder):
    def __init__(self, encoder, decoder, args = None):
        super().__init__(encoder, decoder, args = args)
        args.latent_size = self.encoder.model_enc.config.hidden_size
        self.fc_mean = torch.nn.Linear(args.latent_size, args.latent_size)
        self.fc_log_var = torch.nn.Linear(args.latent_size, args.latent_size)
        self.beta = args.h_noiser_ratio

    def forward(self, input_ids_enc, input_ids_dec=None, lm_labels=None):
        # bsz x latent x feature num
        mean = self.encoder_mean(input_ids_enc)
        log_var = self.encoder_log_var(input_ids_enc)
        sampled_h = self.reparameterize(mean, log_var)
        if isinstance(self.decoder, GPT2Decoder):
            BCE, correct, total, _ = self.decoder(sampled_h, input_ids_dec=input_ids_dec, lm_labels=lm_labels)
        elif isinstance(self.decoder, DeconvolutionDecoder):
            BCE, correct, total, _ = self.decoder(sampled_h, input_ids_dec=input_ids_dec)
        else:
            NotImplementedError
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) 
        loss = BCE + self.beta * KLD  
        return loss, correct, total, mean

    def encoder_mean(self, input_ids_enc, *kargs):
        hidden_state = self.encoder(input_ids_enc, *kargs)
        shapes = hidden_state.shape
        mean = self.fc_mean(hidden_state.permute(0, 2, 1)).view(-1, shapes[2], shapes[1]).permute(0, 2, 1)
        if self.h_tanh:
            mean = torch.tanh(mean)
        return mean

    def encoder_log_var(self, input_ids_enc, *kargs):
        hidden_state = self.encoder(input_ids_enc, *kargs)
        shapes = hidden_state.shape
        log_var = self.fc_log_var(hidden_state.permute(0, 2, 1)).view(-1, shapes[2], shapes[1]).permute(0, 2, 1) # offset with -5
        return log_var

    def save(self, output_dir, prefix):
        torch.save(self.state_dict(), os.path.join(output_dir, prefix+'-VAE.pkl'))
              
    @classmethod
    def from_pretrained(cls, encoder, decoder, input_dir, args):
        prefix = args.resume_ckpt
        model = cls(encoder, decoder, args =args)
        model.load_state_dict(torch.load(os.path.join(input_dir, prefix+'-VAE.pkl'), map_location='cpu'))
        return model

    def named_enc_parameters(self):
        return itertools.chain(self.encoder.named_parameters(), self.fc_mean.named_parameters(), self.fc_log_var.named_parameters())

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)
    



def encoderModels(args):
    encoder = {}
    encoder['model_args'] = {'args':args}
    if args.enc_model.startswith('bert'):
        encoder['model_cls'] = BertEncoder 
    elif 't5' in args.enc_model:
        encoder['model_cls'] = T5Encoder
    elif args.enc_model == 'conv':
        encoder['model_cls'] = ConvolutionEncoder
    elif args.enc_model == 'bertconv':
        encoder['model_cls'] = BertConvEncoder
    else:
        raise NotImplementedError
    return encoder

def decoderModels(args):
    decoder = {}
    decoder['model_args'] = {'args':args}
    if args.dec_model.startswith('gpt2'):
        decoder['model_cls'] = GPT2Decoder
    elif 't5' in args.enc_model:
        decoder['model_cls'] = T5Decoder
    elif args.dec_model == 'deconv':
        decoder['model_cls'] = DeconvolutionDecoder
    elif args.dec_model == 'deconformer':
        decoder['model_cls'] = DeconformerDecoder
    else:
        raise NotImplementedError
    return decoder

