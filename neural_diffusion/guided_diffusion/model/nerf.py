from matplotlib import image, use
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import cv2
import numpy as np
from copy import deepcopy
from functools import partial
from omegaconf import OmegaConf
from einops import rearrange, repeat
from guided_diffusion.nn import base2fourierfeatures
from guided_diffusion.model.unet import UNetModel
from guided_diffusion.utils.rendering_utils import (
    generate_planes,
    generate_rays,
    sample_from_planes,
    sample_importance,
    sample_stratified,
    unify_samples,
    depth_to_normal_image
)

from guided_diffusion.utils.fp16_util import convert_module_to_f16, convert_module_to_f32
from guided_diffusion import logger

class PixelNeRFModel(nn.Module):
    """ 
    For early experiments, we follow similar triplane implementation from EG3D
    https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py
    
    :: image encoder + triplane radiance fields
    """
    def __init__(
        self,
        image_size,
        in_channels,
        out_channels,
        encoder_config,
        decoder_config,
        rendering_options=OmegaConf.create(),
        use_fp16=False,
        **additional_encoder_args
        ):
        super().__init__()
        # resolve omegaconf to python dict to avoid further problems
        rendering_options, encoder_config, decoder_config = \
            [OmegaConf.to_container(c, resolve=True) for c in 
            (rendering_options, encoder_config, decoder_config)]

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = th.float16 if use_fp16 else th.float32
        self.options = rendering_options
        self.num_samples = self.options['depth_samples']
        self.num_fine_samples = self.options.get('depth_fine_samples', 0)

        # update configs
        encoder_config.update(additional_encoder_args)
        encoder_config.update({'use_fp16': use_fp16, 'in_channels': in_channels, 'num_classes': None})
        decoder_config.update({'use_fp16': use_fp16, 'output_dim': out_channels})

        # should the triplane learnable? or from input vector?
        self.encoder = PlaneEncoder(**encoder_config)
        self.decoder = MLPDecoder(**decoder_config)
        
        # cache
        self.cache = {}

    def forward(self, x, timesteps, y=None, subsample=None):
        """
        diffusion model with nerf decoder

        :param x: an [N x C x ...] Tensor of inputs. For exampla, an image N x C x H x W
        :param timesteps: a 1-D batch of timesteps. (N, )
        :param y: an [N] Tensor of labels, if class-conditional. (optional)
        :return: an [N x C x ...] Tensor of outputs.

        """
        assert y.size(-1) == 25, "input y must be camera for now."
        planes = self.encoder(x, timesteps)
        rgb_output = self.forward_rendering(planes, y, subsample)
        return rgb_output

    def forward_rendering(self, planes, camera=None, subsample=None, rendering_options={}):
        """
        Given diffused image(s) and get the rendered image using volume rendering
        """
        options = deepcopy(self.options)
        options.update(rendering_options)
        xdtype = planes[-1].dtype
        
        with th.no_grad():  # get rays given the camera parameters, image-size and subsample indices
            ray_origins, ray_dirs = generate_rays(camera, self.image_size, subsample)
            depths = sample_stratified(ray_origins, options['ray_start'], options['ray_end'], self.num_samples)
            points = rearrange(ray_origins.unsqueeze(-2) + depths * ray_dirs.unsqueeze(-2), 'b n s d -> b (n s) d')

        sigma, rgb = self.decoder(points.type(self.dtype), planes, depths.shape, subsample)

        if self.num_fine_samples > 0:  # (optional) use importantce sampling
            with th.no_grad():
                weights = self.volume_rendering(sigma.type(xdtype), rgb.type(xdtype), depths)[2]
                depths_fine = sample_importance(depths, weights, options['depth_fine_samples'])
                points_fine = rearrange(ray_origins.unsqueeze(-2) + depths_fine * ray_dirs.unsqueeze(-2), 'b n s d -> b (n s) d')
            
            sigma_fine, rgb_fine = self.decoder(points_fine.type(self.dtype), planes, depths_fine.shape, subsample)
            depths, sigma, rgb = unify_samples([depths, sigma, rgb], [depths_fine, sigma_fine, rgb_fine])

        depth_output, rgb_output = self.volume_rendering(sigma.type(xdtype), rgb.type(xdtype), depths)[:2]

        if subsample is None:
            rgb_output   = rearrange(rgb_output, 'b (h w) c -> b c h w', h=self.image_size)
            depth_output = rearrange(depth_output, 'b (h w) c -> b c h w', h=self.image_size)
        else:
            rgb_output   = rearrange(rgb_output, 'b n c -> b c n ()')
            depth_output = rearrange(depth_output, 'b n c -> b c n ()')
        
        self.cache['depth_output'] = \
            (depth_output.detach() - options['ray_start']) / (options['ray_end'] - options['ray_start'])
        
        output_format = options.get('output_format', 'color')
        if output_format == 'normal':
            return depth_to_normal_image(ray_origins, ray_dirs, depth_output)
        elif output_format == 'depth':
            return depth_output
        return rgb_output

    @property
    def depth_output(self):
        depth = self.cache.get('depth_output', None)
        if depth is not None:
            depth = rearrange(th.from_numpy(
                cv2.applyColorMap((rearrange(depth, 'b () h w -> b (h w)').cpu().numpy()*255).astype(np.uint8), 
                cv2.COLORMAP_TURBO)).to(depth.device), 'b (h w) c -> b c h w', h=depth.size(-2), w=depth.size(-1))
            depth = (depth.float() / 255) * 2 - 1
        return depth

    def get_density(self, sigmas):
        clamp_mode = self.options.get('clamp_mode', 'softplus') 
        if clamp_mode == 'softplus':
            sigmas = F.softplus(sigmas - 1) # activation bias of -1 makes things initialize better
        elif clamp_mode == 'exp_truncated':
            sigmas = th.exp(5 - F.relu(5 - (sigmas - 1)))  # up-bound = 5, also shifted by 1
        elif clamp_mode == 'relu':
            if self.training:
                sigmas = sigmas + th.rand_like(sigmas)
            sigmas = F.relu(sigmas)
        else:
            raise NotImplementedError
        return sigmas

    def volume_rendering(self, sigmas, rgbs, depths, threshold=1e-4):        
        deltas = depths[:,:,1:] - depths[:,:,:-1]
        if self.options.get('use_midpoint', False):  # manually compute the middple points for integral
            sigmas = (sigmas[:,:,:-1] + sigmas[:,:,1:]) / 2
            rgbs   = (rgbs[:,:,:-1] + rgbs[:,:,1:]) / 2
            depths = (depths[:,:,:-1] + depths[:,:,1:]) / 2
        else:   # use the first point for integral
            last_depth = self.options['ray_end'] if not self.options.get('inf_background', False) else 1e10
            deltas = th.cat([deltas, last_depth - depths[:,:,-1:]], -2)
        
        # transform density to positive number
        sigmas = self.get_density(sigmas)
        
        try: # by default, using cuda implementation of ray-marching. (save memory and density grids)
            from guided_diffusion.clib import volume_rendering_integration
            batch_size, num_rays, samples_per_ray, _ = sigmas.shape
            ray_idx = th.arange(batch_size * num_rays, device=sigmas.device)
            start_idx = ray_idx * samples_per_ray
            rays_a = th.stack([ray_idx, start_idx, th.ones_like(ray_idx) * samples_per_ray], -1)
            sigmas, rgbs, deltas, depths = sigmas.reshape(-1), rgbs.reshape(-1, 3), deltas.reshape(-1), depths.reshape(-1)
            opacity, depth_output, rgb_output, weights = volume_rendering_integration(
                sigmas, rgbs, deltas, depths, rays_a.int(), threshold)

            # reshape back to the original shapes (RGB need to rescale to -1 ~ 1)
            depth_output = depth_output.reshape(batch_size, num_rays, 1)
            rgb_output = rgb_output.reshape(batch_size, num_rays, 3) * 2 - 1  # rescale to (-1, 1)
            weights = weights.reshape(batch_size, num_rays, samples_per_ray)
            opacity = opacity.reshape(batch_size, num_rays)

        except ImportError:  # fall back to navie pytorch implementation
            density_delta   = sigmas * deltas
            alpha           = 1 - th.exp(-density_delta)
            alpha_shifted   = th.cat([th.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
            weights         = alpha * th.cumprod(alpha_shifted, -2)[:, :, :-1]
            rgb_output      = th.sum(weights * rgbs, -2) * 2 - 1
            depth_output    = th.sum(weights * depths, -2)
            opacity         = weights.sum(-2)

        # add bg depth
        depth_output = depth_output + (1 - opacity[..., None]) * last_depth
        return depth_output, rgb_output, weights, opacity

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.encoder.apply(convert_module_to_f16)
        self.decoder.apply(convert_module_to_f16)
      
    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.encoder.apply(convert_module_to_f32)
        self.decoder.apply(convert_module_to_f32)
        

class PlaneEncoder(nn.Module):
    def __init__(self, num_planes=3, f_dim=32, image_size=256, box_warp=1, arch='fixed_plane', **additional_encoder_config):
        """
        arch is chosen from 'fixed_plane', 'deconv_plane'
        """
        super().__init__()
        assert num_planes == 3, "only support triplane for now"
        self.arch = arch
        self.num_planes = num_planes
        self.f_dim = f_dim
        if self.arch == 'fixed_plane':
            self.planes = nn.Parameter(th.randn(num_planes, f_dim, image_size, image_size))
        
        else:
            from .unet import DecoderUNetModel, EncoderUNetModel, UNetModel
            ch = additional_encoder_config['model_channels'] * additional_encoder_config['channel_mult'][-1]
            ds = 2 ** (len(additional_encoder_config['channel_mult']) - 1)
            self.planes = nn.Parameter(th.randn(ch, image_size // ds, image_size // ds))
            
            if self.arch == 'deconv_plane':  # using a fixed z
                self.fixed_y = nn.Parameter(th.randn(256))
                self.plane_gen = DecoderUNetModel(
                    image_size=image_size, out_channels=f_dim * num_planes, input_dim=256,
                    **additional_encoder_config)
            
            elif self.arch == 'enc_dec_plane':
                self.image_enc = EncoderUNetModel(
                    image_size=image_size, out_channels=512, **additional_encoder_config)
                self.plane_gen = DecoderUNetModel(
                    image_size=image_size, input_dim=512, out_channels=f_dim*num_planes, **additional_encoder_config)

            elif self.arch == 'unet_plane':
                self.unet_enc = UNetModel(
                    image_size=image_size, pool_channels=512, out_channels=f_dim, **additional_encoder_config)
                self.plane_gen = DecoderUNetModel(
                    image_size=image_size, input_dim=512, out_channels=f_dim*num_planes, **additional_encoder_config)
            else:
                raise NotImplementedError("no such model")

        self.register_buffer('plane_axes', generate_planes())
        self.box_warp = box_warp
    
    def forward(self, x, timesteps, **unused):
        batch_size = x.size(0)
        if self.arch == 'fixed_plane':
            planes = repeat(self.planes, 's c h w -> b s c h w', b=batch_size)
        
        elif self.arch == 'deconv_plane':
            planes = repeat(self.planes, 'c h w -> b c h w', b=batch_size)
            fixed_feature = repeat(self.fixed_y, 'd -> b d', b=batch_size)
            planes = rearrange(self.plane_gen(planes, fixed_feature), 'b (s c) h w -> b s c h w', s=self.num_planes)
        
        elif self.arch == 'enc_dec_plane':
            planes = repeat(self.planes, 'c h w -> b c h w', b=batch_size)
            pool_feature = self.image_enc(x, timesteps).contiguous()  # B x D
            planes = rearrange(self.plane_gen(planes, pool_feature), 'b (s c) h w -> b s c h w', s=self.num_planes)
        
        elif self.arch == 'unet_plane':
            planes = repeat(self.planes, 'c h w -> b c h w', b=batch_size)
            unet_feature, pool_feature = self.unet_enc(x, timesteps)
            planes = rearrange(self.plane_gen(planes, pool_feature), 'b (s c) h w -> b s c h w', s=self.num_planes)
            
        else:
            raise NotImplementedError("no such model")
        
        if self.arch == 'unet_plane':
            return (self.plane_axes, self.box_warp, planes, unet_feature)
        return (self.plane_axes, self.box_warp, planes)
       
    def extra_repr(self):
        return f"Planes: {self.planes.shape}"


class MLPDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=64, use_fp16=False, **unused):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_fp16   = use_fp16
        try:
            import tinycudann as tcnn
            self.net = tcnn.Network(input_dim, 1+output_dim, {
                "otype": "FullyFusedMLP", "activation": "ReLU",
                "output_activation": "None", "n_neurons": self.hidden_dim, 
                "n_hidden_layers": 1
            })
            logger.log("use tiny-cuda-nn MLPs")

        except ImportError:
            self.net = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1 + self.output_dim)
            )
            logger.log("use PyTorch MLPs")
        
    def forward(self, points, planes, shape, subsample=None):
        batch_size, num_rays, samples_per_ray, _ = shape
        
        # get point features from pre-computed tri-plane representations
        plane_axes, box_warp, plane_features = planes[:3]
        x = sample_from_planes(plane_axes, plane_features, points, box_warp=box_warp)
        x = rearrange(x.mean(1), 'n m c -> (n m) c')  # Aggregate features

        # TODO: this is just a hack for now. In practice you need to project points
        # TODO: need to fix this and change to project points to plane...
        if len(planes) == 4:
            uf = rearrange(planes[-1], 'b c h w -> b (h w) c')
            if subsample is not None:
                uf = uf.gather(1, repeat(subsample[..., 0], 'b n -> b n c', c=uf.size(-1)))
            uf = repeat(uf, 'b n c -> (b n s) c', s=samples_per_ray) * 0.01
            x  = x + uf

        # get the MLP output        
        sigma_rgb = self.net(x)
        
        # reshape back to normal dimensions
        sigma_rgb = rearrange(sigma_rgb, '(b n s) c -> b n s c', b=batch_size, n=num_rays, s=samples_per_ray)
        rgb       = (th.sigmoid(sigma_rgb[..., 1:])*(1 + 2*0.001) - 0.001)
        sigma     = sigma_rgb[..., 0:1]
        return sigma, rgb


class LightFieldModel(UNetModel):
    """
    A 2D diffusion model but use ray information as inputs
    """
    def __init__(self, image_size, in_channels, lf_input='od_fourier', num_classes=25, *args, **kwargs):
        assert num_classes == 25, f"input camera {num_classes} have to be 25D (pose+intrinisic)"
        self.lightfield_input_format = lf_input
        if self.lightfield_input_format == 'od_fourier':  # PE(origin + view direciton)
            num_classes, L = None, 6
            in_channels = in_channels + L * 12
            self.posenc = partial(base2fourierfeatures, start=0, stop=L)
            logger.info('using ray inputs as condition')
        else:
            logger.info('using pose vector as the condition')    
        
        super().__init__(image_size, in_channels, num_classes=num_classes, *args, **kwargs)

    def forward(self, x, timesteps, y, **kwargs):
        assert (y is not None) and (y.size(-1) == 25)
        if self.lightfield_input_format == 'od_fourier':
            ray_origins, ray_dirs = generate_rays(y, self.image_size, None)
            rays = rearrange(th.cat([ray_origins, ray_dirs], -1), 'b (h w) c -> b c h w', h=self.image_size)
            x, y = th.cat([x, self.posenc(rays)], dim=1), None
        return super().forward(x, timesteps, y)
