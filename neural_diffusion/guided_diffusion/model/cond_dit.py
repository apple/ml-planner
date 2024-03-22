import os, copy
import torch as th
import torch.nn as nn
import torch.distributed as dist
from guided_diffusion.model import register_model
from guided_diffusion.model.dit import DiT
from autoencoder.autoencoder_utils import load_cond_model
from guided_diffusion.utils.fp16_util import convert_linear_to_f16, convert_linear_to_f32

@register_model('CondDiT')
class CondDiT(DiT):
    """
    DiT conditioning on external text input.
    """
    def __init__(
        self,
        image_size=16,
        in_channels=1024,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=None,
        learn_sigma=False,
        use_fp16=False,
        cond_dim=512,
        cond_model=None,
        cond_finetune=True,
        **unused
    ):
        super().__init__(image_size=image_size, in_channels=in_channels, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, num_classes=num_classes, cond_feature=True, learn_sigma=learn_sigma, use_fp16=use_fp16)
        self.cond_dim = cond_dim
        self.cond_model, self.cond_tokenizer = load_cond_model(cond_model)

        if not cond_finetune:
            self.cond_model = self.cond_model.eval()
            for param in self.cond_model.parameters():
                param.requires_grad = False

    def convert_to_fp16(self):
        # self.cond_model.apply(convert_linear_to_f16)
        super().convert_to_fp16()

    def convert_to_fp32(self):
        # self.cond_model.apply(convert_linear_to_f32)
        super().convert_to_fp32()

    def cond_tokenization(self, y):
        input_ids_cond = self.cond_tokenizer(y, padding='max_length', truncation=True, max_length=self.cond_dim, return_tensors='pt')
        return input_ids_cond
    
    def forward_cond_encoding(self, y):
        input_ids_cond = y
        outputs = self.cond_model(input_ids=input_ids_cond)
        h = outputs.last_hidden_state.reshape(input_ids_cond.shape[0], -1)
        # .encoder
        return h

    def forward(self, x, t, y=None, use_cross_attention = False, **unused):
        return super().forward(x, t, y=y, use_cross_attention = use_cross_attention)

    

