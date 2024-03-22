from abc import abstractmethod

import math
import os
import copy

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Optional
from torch import Tensor
from einops import rearrange, repeat
from functools import partial
from ..utils.fp16_util import convert_module_to_f16, convert_module_to_f32
from ..utils.math_utils import gaussian_downsample, gaussian_filter
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


def default_channel_mult_from_image(image_size):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 4)
    elif image_size == 16:
        channel_mult = (1, 4)
    else:
        raise NotImplementedError(f"We did not support ")
    return channel_mult


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
    def forward(self, x: Tensor) -> Tensor:
        return th.permute(x, self.dims)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, start=0, end=None, full=False):
        layers = self
        if end is None:
            end = len(layers)
        outs = []
        for layer in layers[start: end]:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
            outs.append(x)
        if full:
            return outs
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    :param use_convnext: if True, use a block similar to convnext
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        use_convnext=False,
        kernel_size=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_convnext = use_convnext

        # convnext is based on depth-wise convolution, which needs in_channel=out_channel
        need_transform = self.use_convnext and (self.out_channels != self.channels)

        # input module
        if not self.use_convnext:
            input_conv = conv_nd(
                dims, channels, self.out_channels, 
                3 if kernel_size is None else kernel_size, padding=1)
        else:
            input_conv = conv2d_depthwise(
                self.out_channels, 7 if kernel_size is None else kernel_size)
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            input_conv,
        )

        # upsampling or downsampling if used.
        self.updown = up or down or need_transform
        out_ch_updown = channels if not self.use_convnext else self.out_channels
        if up:
            self.h_upd = Upsample(channels, False, dims, out_ch_updown)
            self.x_upd = Upsample(channels, False, dims, out_ch_updown)
        elif down:
            self.h_upd = Downsample(channels, False, dims, out_ch_updown)
            self.x_upd = Downsample(channels, False, dims, out_ch_updown)
        elif need_transform:
            self.h_upd = conv_nd(dims, channels, out_ch_updown, 1)
            self.x_upd = conv_nd(dims, channels, out_ch_updown, 1)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # timestep or label conditioning
        if emb_channels > 0:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )

        # output module
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),  # group normalization
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if (self.out_channels == channels) or self.use_convnext:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)   # 1x1 conv / MLP

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb=None):
        if self.updown:          # needs upsample or downsample the feature map before the last input layer
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)    # this is necessary as it is for the input
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.out_layers(h)
        else:
            h = self.out_layers(h) 
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        use_attention_free=False
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.use_attention_free = use_attention_free
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_attention_free:
            # attention free transformer: useful for long sequences
            self.attention = AFTAttention(channels=channels, kernel_size=15)
        elif use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, *spatial)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, *unused):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, *unused):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).reshape(bs * self.n_heads, ch, length),
            (k * scale).reshape(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AFTAttention(nn.Module):
    """
    A module using local attention from Attention Free Transformer
    https://arxiv.org/abs/2105.14103v2
    """
    def __init__(self, channels, kernel_size=31):
        super().__init__()
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.channels = channels
        self.weights = nn.Parameter(th.randn(channels, 1, *kernel_size) * 1e-2)

    def apply_conv(self, x, weight, megvii=False):
        if megvii:
            from depthwise_conv2d_implicit_gemm import (
                _DepthWiseConv2dImplicitGEMMFP16,
                _DepthWiseConv2dImplicitGEMMFP32)
            conv = _DepthWiseConv2dImplicitGEMMFP16.apply if x.dtype == th.float16 else \
                _DepthWiseConv2dImplicitGEMMFP32.apply
        else:
            conv = partial(
                F.conv2d, padding=(self.kernel_size[0]//2, self.kernel_size[1]//2), groups=self.channels)

        B, C, H, W = x.shape
        # x = x / (64**2)  # average x before sum?
        x_sum = x.sum(dim=[-2, -1], keepdim=True)
        x = conv(x, weight) + x_sum
        return x

    def forward(self, qkv, *spatial):
        """
        Apply AFT attention
        """
        bs, channels, _ = qkv.shape
        qkv     = qkv.reshape(bs, channels, *spatial)
        assert channels % 3 == 0

        def softmax(x):
            return x.flatten(2).float().softmax(dim=-1).view(x.size()).to(x.dtype)
        dtype = qkv.dtype

        qkv   = qkv.float()
        q, k, v = qkv.chunk(3, dim=1)
        k       = softmax(k)   # more stable than exp()
        kv      = k * v
        # weights = self.norm(self.weights).expm1().to(qkv.dtype)
        weights = self.weights.float().expm1()
        # weights = (F.softplus(self.weights)).to(qkv.dtype)
        k       = self.apply_conv(k, weights)
        kv      = self.apply_conv(kv, weights)
        x       = q.sigmoid() * kv / (k + 1e-8)
        
        return x.reshape(bs, x.size(1), -1).type(dtype)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. **Deprecated.**
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_attention_free=False,
        use_convnext=False,
        maximum_atten_res=8,
        with_fourier_features=False,
        disable_skip_connection=False,
        average_skip_connection=False,
        pool_channels=0,
        final_upsampler=False,
        **unused,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.with_fourier_features = with_fourier_features
        if with_fourier_features:
            # https://github.com/google-research/vdm/blob/main/model_vdm.py 
            self.in_channels = self.in_channels * (1 + 2 * 2)
        self.disable_skip_connection = disable_skip_connection
        self.average_skip_connection = average_skip_connection
        self.pool_channels = pool_channels
        self.final_upsampler = final_upsampler

        time_embed_dim = model_channels * 4   # 128 -> 512
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            # self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            self.label_emb = nn.Linear(num_classes, time_embed_dim)  # use linear layer

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, self.in_channels, ch, 3, padding=1))]   # conv at input
        )
        self._feature_size = ch
        input_block_chans = [ch]

        # residual block function
        _resblock = partial(
            ResBlock, emb_channels=time_embed_dim, dropout=dropout, 
            dims=dims, use_checkpoint=use_checkpoint, 
            use_scale_shift_norm=use_scale_shift_norm, use_convnext=use_convnext)

        # define the encoder/downsampling layers
        ds = 1   # downsampling rate
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    _resblock(channels=ch, out_channels=int(mult * model_channels))
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:   # downsampling rates that needs self-attention
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            use_attention_free=use_attention_free,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:    # resolution changes, apply downsampling
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        _resblock(channels=ch, out_channels=out_ch, down=True)
                        if resblock_updown       # using residual block to perform downsampling
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )                        # downsampling directly without res-block
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2    # downsampling rates x2
                self._feature_size += ch

        # define the bottle-neck layers (using self-attention)
        self.middle_block = TimestepEmbedSequential(
            _resblock(channels=ch),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                use_attention_free=use_attention_free,
            ) if (self.image_size // ds) <= maximum_atten_res else nn.Identity(),
            _resblock(channels=ch),
        )
        self._feature_size += ch

        # define pooling layer (optional)
        if self.pool_channels > 0:
            self.pool_out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, self.pool_channels, 1)),
                nn.Flatten(),
            )

        # define the decoder/upsampling layers
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:   # reverse the channel multiplyer
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()  # the channel size of the skip connection
                layers = [
                    _resblock(
                        channels=ch + ich if not self.disable_skip_connection else ch,
                        out_channels=int(model_channels * mult))
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            use_attention_free=use_attention_free,
                        )
                    )
                if (level or self.final_upsampler) and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        _resblock(channels=ch, out_channels=out_ch, up=True)
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2   # reduce the downsampling rate 
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),  # output the final color
        )

    def get_summary(self):
        summary(self, depth=3, row_settings=["var_names"])

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.out.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.out.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, stage=None, **unused):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs. For exampla, an image N x C x H x W
        :param timesteps: a 1-D batch of timesteps. (N, )
        :param y: an [N] Tensor of labels, if class-conditional. (optional)
        :return: an [N x C x ...] Tensor of outputs.
        """
        # x.shape= torch.Size([512, 1, 16, 768])
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))  # Nx128 PE -> Nx512
        if self.num_classes is not None:    # (optional) label condition
            assert y.shape == (x.shape[0], self.num_classes)   # feature vector or one-hot
            emb = emb + self.label_emb(y)

        # input module
        h = x.type(self.dtype)
        if self.with_fourier_features:
            zf = base2fourierfeatures(h, start=6, stop=8, step=1)
            h  = th.cat([h, zf], 1)

        # encoder
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)

        # bottleneck
        h = self.middle_block(h, emb)
        if self.pool_channels > 0:
            h_pool = self.pool_out(h)
        
        # decoder
        for module in self.output_blocks:
            if not self.disable_skip_connection:
                hi = hs.pop()
                if self.average_skip_connection:
                    hi = F.adaptive_avg_pool2d(hi, (1, 1)).expand_as(hi)
                h = th.cat([h, hi], dim=1)
            h = module(h, emb)
        h = self.out(h)
        h = h.type(x.dtype)
        
        if self.pool_channels > 0:
            return h, h_pool.type(x.dtype)
        return h


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    Only the encoder + bottleneck module, and then output a feature vector.
    This is used to train classifier in classifier-based diffusion.
    
    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        **unused
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")
    
    def get_summary(self):
        summary(self, depth=3, row_settings=["var_names"])

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.out.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.out.apply(convert_module_to_f32)

    def forward(self, x, timesteps, return_maps=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = h_mid = self.middle_block(h, emb)

        if self.pool.startswith("spatial"):
            results.append(h.mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
        h = self.out(h)
        h = h.type(x.dtype)

        if return_maps:
            return h, h_mid, hs
        return h


class DecoderUNetModel(nn.Module):
    """
    The half UNet model with attention and input embedding.
    Only the decoder module without , and then output a feature vector.
    This is used to train classifier in classifier-based diffusion.
    
    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        input_dim=512,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        **unused,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.input_dim = input_dim
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        
        time_embed_dim  = model_channels * 4   # 128 -> 512
        ch, ds = model_channels * channel_mult[-1],  2 ** (len(channel_mult) - 1)
        self.input_proj = nn.Linear(input_dim, time_embed_dim)   # use linear layer for inputs
        
        # define the decoder/upsampling layers
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:   # reverse the channel multiplyer
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2   # reduce the downsampling rate 
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),  # output the final color
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_proj.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.out.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_proj.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        self.out.apply(convert_module_to_f32)

    def get_summary(self):
        summary(self, depth=3, row_settings=["var_names"])

    def forward(self, x, y, **unused):

        h = x.type(self.dtype)
        emb = self.input_proj(y)
        
        # only decoder generate from the inputs
        for module in self.output_blocks:
            h = module(h, emb)
        h = self.out(h)
        h = h.type(x.dtype)
        return h


class NoUNetModel(nn.Module):
    def __init__(self, image_size=256, in_channels=3, out_channels=3, **additional_encoder_config):
        super().__init__()
        ch = additional_encoder_config['model_channels'] * additional_encoder_config['channel_mult'][-1]
        ds = 2 ** (len(additional_encoder_config['channel_mult']) - 1)
        self.planes = nn.Parameter(th.randn(ch, image_size // ds, image_size // ds))    
        self.image_enc = EncoderUNetModel(
            image_size=image_size, out_channels=512, in_channels=in_channels, **additional_encoder_config)
        self.plane_gen = DecoderUNetModel(
            image_size=image_size, input_dim=512, out_channels=out_channels, **additional_encoder_config)

    def forward(self, x, timesteps, **unused):
        batch_size = x.size(0)
        planes = repeat(self.planes, 'c h w -> b c h w', b=batch_size)
        image_feature = self.image_enc(x, timesteps).contiguous()  # B x D
        return self.plane_gen(planes, image_feature)
    
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.image_enc.apply(convert_module_to_f16)
        self.plane_gen.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.image_enc.apply(convert_module_to_f32)
        self.plane_gen.apply(convert_module_to_f32)

    def get_summary(self):
        summary(self, depth=3, row_settings=["var_names"])

# -------------------- Progressive Coarse-to-Fine Diffusion Model --------------------------- #

class ProgressiveUNetModel(UNetModel):
    """
    A UNet but also accepts images at different resolutions
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # setup progressive growing stages
        self.setup_stages(*args, **kwargs)

        # additional input modules for low-resolution images
        self.channel_mult_res = {
            self.image_size // (2 ** i): self.channel_mult[i]
            for i in range(len(self.channel_mult))
        }
        input_channels = int(self.channel_mult[0] * self.model_channels)
        self.lowres_input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(
                self.dims, input_channels, 
                int(self.channel_mult_res[min(s[0] * 2, self.image_size)] * self.model_channels),  3, padding=1))
            for s in self.stages])
        self.lowres_output_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(
                self.dims, int(self.channel_mult_res[s[0]] * self.model_channels), 
                input_channels, 3, padding=1))
            for s in self.stages])

    def setup_stages(
        self, 
        initial_size=16, 
        image_size=256, 
        stage_schedule='cosine', 
        stage_type='keep', 
        resize_type='interp', 
        upsample_type='bicubic',
        noise_type='none',
        **unused):
        assert stage_type in ['both', 'sr', 'keep'], f"{stage_type} is not supported."
        assert resize_type in ['default', 'interp', 'gaussian'], f"{resize_type} is not supported."
        assert upsample_type in ['bicubic', 'bilinear', 'nearest', 'learnable'], f"{upsample_type} is not supported."
        assert noise_type in ['none', 'rescale_sigma', 'rescale_both'], f"{noise_type} is not supported"

        # prepare stages
        self.resize_type   = resize_type
        self.upsample_type = upsample_type
        self.start  = int(np.log2(initial_size))
        self.end    = int(np.log2(image_size))
        self.stages = []
        self.stage_type = stage_type
        self.noise_type = noise_type
        
        for s in range(self.start, self.end):
            if (stage_type == 'both') or (stage_type == 'keep'):
                self.stages += [(2**s, 2**s)]
            if (stage_type == 'both') or (stage_type == 'sr'):
                self.stages += [(2**s, 2**(s+1))]
        self.stages += [(image_size, image_size)]
        self.stages = self.stages[::-1]   # 0 --> high res, 1 --> low res

        self.stage_schedule = stage_schedule
        self.stage_timespaces = th.linspace(0, 1, len(self.stages)+1)
        if stage_schedule == 'cosine':
            self.stage_timespaces = th.cos((1 - self.stage_timespaces) * np.pi/2).clamp(min=0, max=1)

        if upsample_type == 'learnable':  # learnable upsampler (space transformation)
            learnable_upsampler = {}
            for s in range(self.start, self.end + 1):  # simple 2 layer convnet upsampler (?)
                learnable_upsampler[str(2**s)] = nn.Sequential(
                    conv_nd(2, 3, 128, 3, padding=1),
                    SiLU(), 
                    zero_module(conv_nd(2, 128, 3, 3, padding=1))
                )
            self.learnable_upsampler = nn.ModuleDict(learnable_upsampler)

    def get_transformed_alpha_sigma(self, alphas, sigmas, stages):
        if self.noise_type == 'none':
            return alphas, sigmas
        if isinstance(stages, th.Tensor):
            if stages.ndim == 1:  # scalar
                stages = repeat(stages, '() -> b', b=len(alphas))
        else:
            stages = th.Tensor([a for s in stages for a in [s[0] for _ in s[1]]]).to(alphas.device)
        
        d = 2 ** stages
        if self.noise_type == 'rescale_sigma':
            sigmas = sigmas / d
        elif self.noise_type == 'rescale_both':
            alphas = th.sqrt(d**2 * alphas**2 / (1 + (d**2-1) * alphas**2))
            sigmas = th.sqrt(sigmas ** 2 / (d**2 - (d**2-1) * sigmas**2))
        else:
            raise NotImplementedError
        return alphas, sigmas

    def get_initial_noise(self, noise):
        if self.noise_type == 'rescale_sigma':
            return noise * 2 ** (1 - len(self.stages))
        return noise

    def get_stage(self, s, t=None):
        """
        s: global random step
        t: local random step (optional)
        """
        timspaces = self.stage_timespaces.to(s.device)
        stage = (th.searchsorted(timspaces, s) - 1).clamp(min=0)
        if t is None:
            return stage
        new_t = timspaces[stage] + (timspaces[stage+1]-timspaces[stage]) * t
        return stage, new_t

    def get_upsampled_image(self, x, upsample_type=None):
        if upsample_type is None:
            upsample_type = self.upsample_type
        if upsample_type == 'bicubic':
            return F.interpolate(x, scale_factor=2, mode='bicubic')
        elif upsample_type == 'bilinear':
            return F.interpolate(x, scale_factor=2, mode='bilinear')
        elif upsample_type == 'nearest':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        elif upsample_type == 'learnable':
            x = F.interpolate(x, scale_factor=2, mode='bicubic')
            return x + self.learnable_upsampler[str(x.size(-1))](x)
        else:
            raise NotImplementedError

    def get_target(self, x_start, stage):
        if self.stages[stage][0] == x_start.size(-1):
            return x_start

        if self.resize_type == 'default':
            x_target = F.interpolate(x_start, size=self.stages[stage][1], mode='bilinear', antialias=True)
        
        elif self.resize_type == 'interp':
            x_target = F.interpolate(x_start, size=self.stages[stage][0], mode='bilinear', antialias=True)
            if self.stages[stage][1] > self.stages[stage][0]:                   
                x_target = self.get_upsampled_image(x_target).detach()
        
        elif self.resize_type == 'gaussian':
            x_target = gaussian_downsample(x_start, factor=x_start.size(-1) // self.stages[stage][0])
            if self.stages[stage][1] > self.stages[stage][0]:
                x_target = self.get_upsampled_image(x_target).detach()
        
        else:
            raise NotImplementedError
        return x_target

    def get_interp_ratio(self, stage, t):
        timspaces = self.stage_timespaces.to(t.device)
        t_min, t_max = timspaces[stage], timspaces[stage+1]
        ratio = ((t - t_min) / (t_max - t_min))
        if ratio.ndim == 1:
            ratio = ratio[:, None, None, None]
        return ratio

    def get_start(self, x_start, stage, t, return_states=False):
        states = {}
        if self.resize_type == 'default':
            x_start = F.interpolate(x_start, size=self.stages[stage][0], mode='bilinear', antialias=True)
        
        elif self.resize_type == 'interp':
            # prepare high resolution target
            if x_start.size(-1) == self.stages[stage][0]:
                x_high = x_start
            else:
                x_high = F.interpolate(x_start, size=self.stages[stage][0], mode='bilinear', antialias=True)
        
            # prepare low resolution target using additional interpolation
            x_low = F.interpolate(x_high, size=x_high.size(-1)//2, mode='bilinear', antialias=True)
            # x_low   = F.interpolate(x_low,   size=x_high.size(-1), mode='bilinear')
            x_up = self.get_upsampled_image(x_low).detach()  #TODO: also no gradient (?) for x_t

            # get interpolation
            ratio = self.get_interp_ratio(stage, t)
            x_start = x_high * (1 - ratio) + x_up * ratio
            states = {'x_high': x_high, 'x_low': x_low}

        elif self.resize_type == 'gaussian':
            if x_start.size(-1) == self.stages[stage][0]:
                x_high = x_start
            else:
                x_high = gaussian_downsample(x_start, factor=x_start.size(-1) // self.stages[stage][0])
            ratio = self.get_interp_ratio(stage, t)
            # TODO: is it a good idea to moving in linear?? noise kernel is exp(-freq * t)
            #       later try: moving in log-space??
            # ratio = 0.5 * th.exp((np.log(0.001 ** 2) * (1 - ratio) + np.log(2) * ratio)) * ratio.lt(0)
            x_start = gaussian_filter(x_high, ratio)
            states = {'x_high': x_high}

        else:
            raise NotImplementedError      

        if return_states:
            return x_start, states
        return x_start

    def forward(self, x, timesteps, y=None, stage=None, x_states=None, **unused):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        assert self.pool_channels == 0
        assert (not self.disable_skip_connection)
        if stage is None: stage = 0

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))  # Nx128 PE -> Nx512
        if self.num_classes is not None:    # (optional) label condition
            assert y.shape == (x.shape[0], self.num_classes)   # feature vector or one-hot
            emb = emb + self.label_emb(y)

        # input module
        h = x.type(self.dtype)
        if self.with_fourier_features:
            zf = base2fourierfeatures(h, start=6, stop=8, step=1)
            h  = th.cat([h, zf], 1)
        h = self.input_blocks[0](h, emb)
        
        # projector
        h = self.lowres_input_blocks[stage](h, emb)
        hs.append(h) # do not forget to put the input layer

        # encoder
        in_level = int(self.end - np.log2(self.stages[stage][0])) * 2
        for module in self.input_blocks[1+in_level:]:
            h = module(h, emb)
            hs.append(h)

        # bottleneck
        h = self.middle_block(h, emb)
        
        # decoder (before the last layer)
        num_hs = len(hs)
        for lyr, module in enumerate(self.output_blocks[:num_hs]):
            h = th.cat([h, hs.pop()], dim=1)
            if (stage > 0) and (self.stages[stage][0] == self.stages[stage][1]) and (lyr == num_hs-1):
                h = module(h, emb, end=-1)
            else:
                h = module(h, emb)
  
        # projector
        h = self.lowres_output_blocks[stage](h, emb)
        
        # output RGB
        h = self.out(h)
        h = h.type(x.dtype)

        # do additional loss (optional)
        if (x_states is not None) and (self.upsample_type == 'learnable'):
            x_loss = {'upsample': (self.get_upsampled_image(x_states['x_low']) - x_states['x_high']) ** 2}
            return h, x_loss
        return h, None

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.lowres_input_blocks.apply(convert_module_to_f16)
        self.lowres_output_blocks.apply(convert_module_to_f16)
        super().convert_to_fp16()

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.lowres_input_blocks.apply(convert_module_to_f32)
        self.lowres_output_blocks.apply(convert_module_to_f32)
        super().convert_to_fp32()


class SimpleResNetModel(nn.Module):
    """
    simple try of using 3 resnet blocks
    """
    def __init__(self, image_size, in_channels, model_channels, out_channels, 
                num_res_blocks=1, final_upsampler=False, num_classes=None, 
                use_fp16=False, kernel_size=3, dropout=0.0, channel_mult=4, 
                skip_connection=False, **unused):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks  # this is for both encoder/decoder
        self.final_upsampler = final_upsampler
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dtype = th.float16 if use_fp16 else th.float32
        self.skip_connection = skip_connection

        dims = 2
        time_embed_dim = model_channels * 4   # 128 -> 512
        middle_channels = model_channels * channel_mult
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = nn.Linear(num_classes, time_embed_dim)  # use linear layer

        ch = input_ch = model_channels
        self.blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, self.in_channels, input_ch, kernel_size, padding=1))]   # conv at input
        )
        input_block_chans = [ch]

        _resblock = partial(
            ResBlock, emb_channels=time_embed_dim, dropout=dropout, 
            dims=dims, use_checkpoint=False, 
            use_scale_shift_norm=True, use_convnext=True, kernel_size=15)

        for _ in range(num_res_blocks):
            layers = [_resblock(ch, out_channels=middle_channels)]
            ch = middle_channels
            self.blocks.append(TimestepEmbedSequential(*layers))
            input_block_chans += [ch]
        
        self.blocks.append(
            TimestepEmbedSequential(_resblock(ch))
        )
        
        in_ch = input_block_chans.pop()  # still keep skip-connection?
        for i in range(num_res_blocks):
            out_ch = input_block_chans.pop()
            layers = [_resblock(ch + in_ch, out_channels=out_ch)]
            in_ch = ch = out_ch
            # add attention layer here?? (TODO)
            if (i == (num_res_blocks - 1)) and self.final_upsampler:
                layers.append(Upsample(ch, use_conv=True, dims=dims))
            self.blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, kernel_size, padding=1)),  # output the final color
        )

    def convert_to_fp16(self):
        self.blocks.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        self.blocks.apply(convert_module_to_f32)
    
    def get_summary(self):
        summary(self, depth=3, row_settings=["var_names"])

    def forward(self, x, timesteps, y=None, **unused):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))  # Nx128 PE -> Nx512
        if self.num_classes is not None:    # (optional) label condition
            assert y.shape == (x.shape[0], self.num_classes)   # feature vector or one-hot
            emb = emb + self.label_emb(y)
        
        h  = x.type(self.dtype)
        l  = self.num_res_blocks
        hs = []
        h  = self.blocks[0](h, emb)  # input layer
        for module in self.blocks[1:l+1]:
            h = module(h, emb)
            hs.append(h)
        h = self.blocks[l+1](h, emb)
        for module in self.blocks[l+2:]:
            h = module(th.cat([h, hs.pop()], 1), emb) 
        h = self.out(h).type(x.dtype)
        return h


class ProgressiveMultiUNetModel(ProgressiveUNetModel, nn.Module):
    """
    Different from sharing the same architecture
    """
    def __init__(self, module_layer='unet', *args, **kwargs):
        nn.Module.__init__(self)

        # setup progressive growing stages
        self.setup_stages(*args, **kwargs)

        # setup multiple UNets
        all_unets = []
        for s, i in enumerate(self.stages):
            in_res, out_res = self.stages[s]
            unet_kwargs = copy.deepcopy(kwargs)
            if module_layer == 'unet':
                unet_kwargs.update({
                    'channel_mult': default_channel_mult_from_image(in_res),
                    'image_size': in_res, 'final_upsampler': out_res > in_res})
                all_unets.append(UNetModel(**unet_kwargs))
            
            elif module_layer.startswith('shallow_unet'):
                setting = module_layer.split('.')
                if len(setting) == 1:
                    max_res, max_atten_res = 64, 8
                elif len(setting) == 2:
                    max_res, max_atten_res = int(setting[1]), 8
                else:
                    max_res, max_atten_res = int(setting[1]), int(setting[2])
                
                unet_kwargs.update({
                    'channel_mult': default_channel_mult_from_image(min(in_res, max_res)),
                    'image_size': in_res, 'final_upsampler': out_res > in_res,
                    'maximum_atten_res': max_atten_res})
                all_unets.append(UNetModel(**unet_kwargs))
            
            elif module_layer == 'simple_resnet':
                unet_kwargs.update({
                    'num_res_blocks': 1, 'channel_mult': min(self.image_size // out_res, 4),
                    'image_size': in_res, 'final_upsampler': out_res > in_res})
                all_unets.append(SimpleResNetModel(**unet_kwargs))
            else:
                raise NotImplementedError('not implemented.')
        self.all_unets = nn.ModuleList(all_unets)

    def forward(self, x, timesteps, y=None, stage=None, x_states=None, **unused):
        if stage is None: stage = 0
        h = self.all_unets[stage](x, timesteps, y)
        
        # do additional loss (optional)
        if (x_states is not None) and (self.upsample_type == 'learnable'):
            x_loss = {'upsample': (self.get_upsampled_image(x_states['x_low']) - x_states['x_high']) ** 2}
            return h, x_loss
        return h, None

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.all_unets.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.all_unets.apply(convert_module_to_f32)

    def get_summary(self):
        summary(self, depth=4, row_settings=["var_names"])


class MultiStageUNetModel(ProgressiveUNetModel):
    """
    try to train all possible stages in one forward pass.
    """
    @property
    def is_multi_stage(self):
        return True

    def get_multi_stages(self, t):
        stages = {}
        for ti in t:
            si = self.get_stage(ti).item()
            if si not in stages:
                stages[si] = [ti]
            else:
                stages[si] += [ti]
        stages = {k: th.Tensor(v).type_as(t) for k, v in stages.items()}
        stages = sorted(stages.items(), key=lambda a:a[0])
        return stages

    def forward(self, x, timesteps, y=None, stage=None, x_states=None, **unused):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        assert self.pool_channels == 0
        assert (not self.disable_skip_connection), "do not support"
        assert x_states is None, "for now, does not support training upsampler"

        if stage is None:
            stage = 0

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))  # Nx128 PE -> Nx512
        if self.num_classes is not None:    # (optional) label condition
            emb = emb + self.label_emb(y)

        LIST_INPUT = True
        if not isinstance(x, list):
            x = [{'stage': stage, 'x': x, 'emb': emb, 'start_end': [0, len(emb)], 
                  'in_level': 2 * int(self.end - np.log2(self.stages[stage][0]))}]
            LIST_INPUT = False        
        else:
            for i in range(len(x)):
                x[i]['emb'] = emb[x[i]['start_end'][0]: x[i]['start_end'][1]]
                x[i]['in_level'] = 2 * int(self.end - np.log2(self.stages[x[i]['stage']][0]))

        def input_func(x, stage, emb):
            h = x.type(self.dtype)
            if self.with_fourier_features:
                zf = base2fourierfeatures(h, start=6, stop=8, step=1)
                h  = th.cat([h, zf], 1)
            h = self.input_blocks[0](h, emb)
            h = self.lowres_input_blocks[stage](h, emb)
            return h

        def output_func(h, stage, emb):
            h = self.lowres_output_blocks[stage](h, emb)
            x = self.out(h)
            return x

        # encoder
        index, h, emb = 0, None, None
        for lyr, module in enumerate(self.input_blocks[1:]):
            if (index < len(x)) and (lyr == x[index]['in_level']):
                if h is None:
                    h = input_func(x[index]['x'], x[index]['stage'], x[index]['emb'])
                    emb = x[index]['emb']
                    hs.append(h)  # HACK: do not forget to put the input layer
                else:
                    h = th.cat([h, input_func(x[index]['x'], x[index]['stage'], x[index]['emb'])], 0)
                    emb = th.cat([emb, x[index]['emb']], 0)
                    hs[-1] = h    # HACK: need to update to concatenate
                index += 1
            if h is not None:
                h = module(h, emb)
                hs.append(h)

        # bottleneck
        h = self.middle_block(h, emb)

        # decoder
        preds, index = [], len(x)-1
        for lyr, module in enumerate(self.output_blocks):
            bound_layer = (len(self.output_blocks) - x[index]['in_level']) - 1
            stage = x[index]['stage']
            
            h = th.cat([h, hs.pop()], dim=1)
            if (stage > 0) and (self.stages[stage][0] == self.stages[stage][1]) and (lyr == bound_layer):
                h_m1, h = module(h, emb, full=True)[-2:]
            else:
                h_m1, h = None, module(h, emb)
  
            if lyr == bound_layer:  # remove the boundary
                ls = len(x[index]['x'])
                preds += [
                    output_func(
                        h[-ls:] if h_m1 is None else h_m1[-ls:],
                        stage, x[index]['emb']).type(x[index]['x'].dtype)]
                h, emb = h[: -ls], emb[: -ls]
                index -= 1
            if index == -1:
                break
        
        if LIST_INPUT:
            return preds[::-1], None
        return preds[0], None