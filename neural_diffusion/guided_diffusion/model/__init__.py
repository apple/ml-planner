from .unet import UNetModel, SuperResModel, EncoderUNetModel, default_channel_mult_from_image
# from .nerf import PixelNeRFModel


import os, importlib
import torch.nn as nn
from torchinfo import summary

MODEL_REGISTRY = {}


def register_model(name):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.
    For example::
        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)
    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.
    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def get_model(name):
    name = name.split('.')[-1]  # in order to adapt OLD configs
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    raise ValueError('Cannot find registered model: {name} from ' + ','.join([key for key in MODEL_REGISTRY]))



class DiffusionModelBase(nn.Module):
    """
    Abstract module for all diffusion models.
    """
    # @property
    # def use_ldm(self):
    #     return getattr(self, '_use_ldm', False)

    # @use_ldm.setter
    # def use_ldm(self, value: bool):
    #     self._use_ldm = value

    def get_summary(self):
        summary(self, depth=3, row_settings=["var_names"])
    
    def convert_to_fp16(self):
        pass

    def convert_to_fp32(self):
        pass    

    # def clamp(self, x, bound=1, dynamic=False, clip_denoised=True):  # put clampping inside UNET
    #     if clip_denoised:               # clipping the gaussian noise?
    #         if dynamic:
    #             s = th.quantile(abs(x), 0.995)
    #             s = max(s, bound)
    #             return x.clamp(-s, s) / s
    #         return x.clamp(-bound, bound)
    #     return x


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('guided_diffusion.model.' + model_name)