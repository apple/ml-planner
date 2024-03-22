# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from guided_diffusion.model import DiffusionModelBase, register_model
from guided_diffusion.utils.fp16_util import convert_linear_to_f16, convert_linear_to_f32

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = th.exp(
            -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with gated adaptive layer norm (adaLN) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, y=None):
        if y is not None:
            y = y.reshape(x.shape[0], -1, x.shape[2]).to(x.dtype)
            x = th.cat((y, x), dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x.float()).to(x.dtype), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp( modulate(self.norm2(x.float()).to(x.dtype), shift_mlp, scale_mlp))
        if y is not None:
            x = x[:,y.shape[1]:,:]
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c, y=None):
        if y is not None:
            y = y.reshape(x.shape[0], -1, x.shape[2]).to(x.dtype)
            x = th.cat((y, x), dim=1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        if y is not None:
            x = x[:,y.shape[1]:,:]
        return x


@register_model('DiT')
class DiT(DiffusionModelBase):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        image_size=16,
        in_channels=1024,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=None,
        cond_feature=False,
        learn_sigma=False,
        use_fp16=False,
        **unused
    ):
        super().__init__()
        self.image_size = image_size
        self.dtype = th.float16 if use_fp16 else th.float32
        hidden_size = in_channels
        self.cond_feature = cond_feature

        self.learn_sigma = learn_sigma
        self.num_heads = num_heads

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if num_classes is not None and num_classes > 0:

            if self.cond_feature:
                num_classes = num_classes*hidden_size
                self.non_emb = nn.Parameter(th.randn(num_classes), requires_grad=True)
            if num_classes<100000:
                self.y_embedder = nn.Linear(num_classes, hidden_size)  # use linear layer
            else:
                self.y_embedder = nn.Linear(num_classes//hidden_size, 1)
  # use linear layer
            # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # else:
        #     self.y_embedder = None

        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:

        self.num_classes = num_classes
        self.pos_embed = nn.Parameter(th.zeros(1, image_size, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size)
        self.initialize_weights()

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_linear_to_f16)
        self.final_layer.apply(convert_linear_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_linear_to_f32)
        self.final_layer.apply(convert_linear_to_f32)


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[1])
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, y=None, use_cross_attention = False, **unused):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = x.squeeze(1)  #  (N, T, D)
        x = x + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)                   # (N, D)
        
        if (y is not None) and (self.y_embedder is not None): # and (not use_cross_attention):
            if self.num_classes<100000:
                y_emb = self.y_embedder(y)    # (N, D)
            else:
                y_emb = self.y_embedder(y.reshape(x.shape[0],-1,x.shape[2]).transpose(1,2)).transpose(1,2).reshape(x.shape[0],-1)
            c = t + y_emb                                # (N, D)
        else:
            c = t

        # x_type = x.dtype
        x, c = x.to(self.dtype), c.to(self.dtype)
        if use_cross_attention:
            for block in self.blocks:
                x = block(x, c, y)                      # (N, T, D)
            x = self.final_layer(x, c, y)               # (N, T, D)
        else:
            for block in self.blocks:
                x = block(x, c)                      # (N, T, D)
            x = self.final_layer(x, c)               # (N, T, D)
        # x = self.unpatchify(x.to(x_type))        # (N, out_channels, H, W)
        x = x.unsqueeze(1)   # (N, 1, T, D) 
        return x



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height
    return:
    pos_embed: [grid_size, embed_dim] 
    """
    grid = np.expand_dims(np.arange(grid_size, dtype=np.float32), axis = 0)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


