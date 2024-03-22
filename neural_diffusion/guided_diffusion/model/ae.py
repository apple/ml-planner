import torch.nn as nn
import torch.nn.functional as F
import torch as th
from functools import partial
from guided_diffusion.model.unet import ResBlock


class StackedAutoEncoder(nn.Module):
    """
    A simple stacked auto-encoder used for signal transformation
    KL regularization is needed.
    """
    def __init__(
        self,
        image_size,
        in_channels=3,
        model_channels=128,
        out_channels=4,  # this means the latent space dimension
        num_res_blocks=1,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=False,
        attention_resolutions=None,
        use_new_attention_order=False,
        **unused
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
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
    
        # residual block function
        _resblock = partial(
            ResBlock, emb_channels=0, dropout=dropout, 
            dims=dims, use_checkpoint=use_checkpoint)
        _atnblock = partial(
            AttentionBlock, use_checkpoint=use_checkpoint,
            num_heads=num_heads, num_head_channels=num_head_channels,
            use_new_attention_order=use_new_attention_order)

        # define all the levels
        ds = 1  # downsampling rate
        for level, mult in enumerate(channel_mult[:-1]):  # no last layer, useless here
            ch = int(channel_mult[level] * model_channels)

            # encoder module
            encoder_block = []

            # input 
            encoder_block.append(
                conv_nd(dims, self.in_channels if level == 0 else self.out_channels, ch, 3, padding=1)
            )
            # encode + downsample
            for _ in range(num_res_blocks):
                layers = [
                    _resblock(channels=ch, out_channels=int(mult * model_channels))
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:   # downsampling rates that needs self-attention
                    layers.append(_atnblock(ch))
                encoder_block.append(nn.Sequential(*layers))
            out_ch = ch
            encoder_block.append(
                nn.Sequential(
                    _resblock(channels=ch, out_channels=out_ch, down=True)
                    if resblock_updown       # using residual block to perform downsampling
                    else Downsample(
                        ch, conv_resample, dims=dims, out_channels=out_ch
                    )                        # downsampling directly without res-block
                )
            )
            # output
            encoder_block.append(
                conv_nd(dims, out_ch, 2 * self.out_channels, 3, padding=1)
            )

            # decoder module
            decoder_block = []

            


            ch = out_ch
            ds *= 2    




        from fairseq import pdb;pdb.set_trace()