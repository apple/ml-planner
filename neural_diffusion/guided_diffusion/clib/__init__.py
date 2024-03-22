import torch as th
from torch.cuda.amp import custom_fwd, custom_bwd

from guided_diffusion.utils.rendering_utils import volume_rendering_integration
try:
    import guided_diffusion.clib._ext as _ext
except ImportError:
    raise ImportError(
        "Could not import _ext module.\n"
        "Please see the setup instructions in the README")


class VolumeRenderer(th.autograd.Function):
    """
    Volume rendering with different number of samples per ray

    Inputs:
        sigmas: (N)
        rgbs: (N, 3)
        deltas: (N)
        ts: (N)
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
        T_threshold: float, stop the ray if the transmittance is below it

    Outputs:
        opacity: (N_rays)
        depth: (N_rays)
        rgb: (N_rays, 3)
    """
    @staticmethod
    @custom_fwd(cast_inputs=th.float32)
    def forward(ctx, sigmas, rgbs, deltas, ts, rays_a, T_threshold):
        opacity, depth, rgb, weights = \
            _ext.composite_train_fw(sigmas, rgbs, deltas, ts,
                                    rays_a, T_threshold)
        ctx.save_for_backward(sigmas, rgbs, deltas, ts, rays_a,
                              opacity, depth, rgb)
        ctx.T_threshold = T_threshold
        return opacity, depth, rgb, weights

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dopacity, dL_ddepth, dL_drgb, dL_weights):
        """ TODO: here dL_weights is ignored.. using dL_dopacity to compute gradients """
        sigmas, rgbs, deltas, ts, rays_a, \
        opacity, depth, rgb = ctx.saved_tensors
        dL_dsigmas, dL_drgbs = \
            _ext.composite_train_bw(dL_dopacity, dL_ddepth,
                                    dL_drgb, sigmas, rgbs, deltas, ts,
                                    rays_a,
                                    opacity, depth, rgb,
                                    ctx.T_threshold)
        return dL_dsigmas, dL_drgbs, None, None, None, None


volume_rendering_integration = VolumeRenderer.apply