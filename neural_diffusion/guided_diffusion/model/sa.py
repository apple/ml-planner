from abc import abstractmethod

# import math
import os
# import copy

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from typing import Callable, List, Optional
# from torch import Tensor
# from einops import rearrange, repeat
# from functools import partial

from transformers import BertModel, BertTokenizer, BertConfig, BertLayer
from transformers.models.bert.modeling_bert import BertLMPredictionHead, BertEmbeddings

from ..utils.fp16_util import convert_module_to_f16, convert_module_to_f32
# from ..utils.math_utils import gaussian_downsample, gaussian_filter
from ..nn import (
    SiLU,
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    base2fourierfeatures,
    conv2d_depthwise
)
from torchinfo import summary

# TODO: global flag based on environment variable.
FLAG = os.environ.get("DISABLE_ZERO_MODULE", "0")
zero_module = zero_module if (FLAG == "0") else lambda x: x


class SelfAttentionModel(nn.Module):
    def __init__(self, num_feature=16, hidden_size=768, image_size = None, **unused):
        super().__init__()
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.num_feature = num_feature
        # placeholder
        self.image_size = image_size
        self.hidden_size = hidden_size 
        # from scratch
        config = BertConfig.from_pretrained('bert-large-uncased') # 'bert-base-uncased'
        config.hidden_size = self.hidden_size          
        self.model_enc = BertModel(config)  

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # self.embedding = BertEmbeddings(config)

        self.model_size = sum(t.numel() for t in self.model_enc.parameters())
        time_embed_dim = hidden_size * 4   # TODO: check with @Jiatao
        self.time_embed = nn.Sequential(
            linear(hidden_size, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, hidden_size),  # TODO: check with @Jiatao
        )
    
    def get_summary(self):
        summary(self, depth=3, row_settings=["var_names"])

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.model_enc.apply(convert_module_to_f16) # TODO: check with @Jiatao

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.model_enc.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, stage=None, **unused):  # bsz x  feature num x H
        """
        Apply the model to an input batch.

        :param x: an [N x 1 x F x H] Tensor of inputs. 
        :param timesteps: a 1-D batch of timesteps. (N, )
        :param y: a label, if class-conditional. (optional)
        :return: an [N x 1 x F x H] Tensor of outputs.
        """
        input_shape = x.size()[:-1]


        x = x.squeeze(1)   
        # t_emb = timestep_embedding(timesteps, self.hidden_size) 
        t_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_size)) 
        t_emb = t_emb.view(t_emb.size(0), 1, -1)     # TODO: check with Jiatao
        x += t_emb  

        seq_length = x.size()[:-1][1]
        position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)/1000.0
        embedding = x + position_embeddings
        # embedding = self.embedding(inputs_embeds=x) 

        assert((not y) or (len(y) == 1))
        token_type_ids = torch.ones(input_shape, dtype=torch.long, device=x.device)*y if y else None
        encoded_output = self.model_enc(inputs_embeds=embedding, token_type_ids=token_type_ids)
        hidden_state = encoded_output['last_hidden_state'] #[:, :self.num_feature, :]
        # hidden_state = torch.tanh(hidden_state)
        # torch.mean(hidden_state, -1), torch.std(hidden_state, -1)
        return hidden_state.unsqueeze(1) # bsz 1 feature_num H


    

    # def named_parameters(self):
    #     return self.model_enc.named_parameters()

    # def save(self, output_dir, prefix):
    #     torch.save(self.model_enc.state_dict(), os.path.join(output_dir, prefix+'-BERT.pkl'))

    # def encode(self, text):
    #     input_ids = self.tokenizer.encode(text)
    #     input_ids = torch.tensor([input_ids]).cuda()
    #     return self.forward(input_ids)