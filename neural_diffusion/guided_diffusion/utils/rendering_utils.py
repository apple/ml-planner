import torch as th
import torch.nn.functional as F
from . import math_utils
from einops import rearrange, repeat


def generate_rays(camera, resolution=256, subsample=None):
    """
    https://github.com/NVlabs/eg3d/blob/main/eg3d/training/volumetric_rendering/ray_sampler.py
    camera parameters (pose + intrinsics): N x 25
    """
    cam2world_matrix = camera[:, :16].reshape(-1, 4, 4)
    intrinsics = camera[:, -9:].reshape(-1, 3, 3)

    N, M = cam2world_matrix.shape[0], resolution**2
    cam_locs_world = cam2world_matrix[:, :3, 3]
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    uv = th.stack(th.meshgrid(
            th.arange(resolution, dtype=th.float32, device=cam2world_matrix.device), 
            th.arange(resolution, dtype=th.float32, device=cam2world_matrix.device))) * (1./resolution) + (0.5/resolution)
    uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
    uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

    x_cam = uv[:, :, 0].reshape(N, -1)
    y_cam = uv[:, :, 1].reshape(N, -1)
    z_cam = th.ones((N, M), device=cam2world_matrix.device)

    x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y_cam/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
    y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

    cam_rel_points   = th.stack((x_lift, y_lift, z_cam, th.ones_like(z_cam)), dim=-1)
    world_rel_points = th.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

    ray_dirs = world_rel_points - cam_locs_world[:, None, :]
    ray_dirs = th.nn.functional.normalize(ray_dirs, dim=2)
    ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

    if subsample is not None:
        ray_origins, ray_dirs = ray_origins.gather(1, subsample), ray_dirs.gather(1, subsample)
    return ray_origins, ray_dirs


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return th.tensor([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]]], dtype=th.float32)


def project_onto_planes(plane_axes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = plane_axes.shape
    coordinates = repeat(coordinates, 'n m h -> (n s) m h', s=n_planes)
    inv_planes  = repeat(th.linalg.inv(plane_axes), 's h w -> (n s) h w', n=N)
    projections = th.bmm(coordinates, inv_planes.type(coordinates.dtype))
    return projections[..., :2]


def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    batch_size = plane_features.shape[0]
    plane_features = rearrange(plane_features, 'n s c h w -> (n s) c h w')
    coordinates = (2 / box_warp) * coordinates # box is assumed to be centered
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = F.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False)
    output_features = rearrange(output_features, '(n s) c () m -> n s m c', n=batch_size)
    return output_features


def sample_stratified(ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
    """
    Return depths of approximately uniformly spaced samples along rays.
    """
    batch_size, num_rays, _ = ray_origins.shape
    if disparity_space_sampling:  # sample in disparity (1/depth)
        depths_coarse = th.linspace(0, 1, depth_resolution+1, device=ray_origins.device)[...,:-1]
        depths_coarse = repeat(depths_coarse, 'd -> n m d ()', n=batch_size, m=num_rays)
        depth_delta   = 1 / depth_resolution
        depths_coarse = depths_coarse + th.rand_like(depths_coarse) * depth_delta
        depths_coarse = 1. / (1. / ray_start * (1. - depths_coarse) + 1. / ray_end * depths_coarse)
    else:
        if type(ray_start) == th.Tensor:
            depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution + 1).permute(1,2,0,3)[...,:-1]
            depth_delta   = (ray_end - ray_start) / depth_resolution
            depths_coarse = depths_coarse + th.rand_like(depths_coarse) * depth_delta[..., None]
        else:
            depths_coarse = repeat(
                th.linspace(ray_start, ray_end, depth_resolution + 1, device=ray_origins.device)[..., :-1], 
                'd -> n m d ()', n=batch_size, m=num_rays)
            depth_delta   = (ray_end - ray_start) / depth_resolution
            depths_coarse = depths_coarse + th.rand_like(depths_coarse) * depth_delta
    return depths_coarse


def sample_importance(z_vals, weights, N_importance):
    """
    Return depths of importance sampled points along rays. See NeRF importance sampling for more.
    """
    batch_size, num_rays, samples_per_ray, _ = z_vals.shape
    z_vals  = z_vals.reshape(batch_size * num_rays, samples_per_ray)
    weights = weights.reshape(batch_size * num_rays, -1)  # -1 to account for loss of 1 sample in MipRayMarcher

    # smooth weights (do we need this?)
    weights = F.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
    weights = F.avg_pool1d(weights, 2, 1).squeeze()
    weights = weights + 0.01

    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
    importance_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
    return importance_z_vals


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / th.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = th.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = th.cat([th.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) padded to 0~1 inclusive

    if det:
        u = th.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = th.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds  = th.searchsorted(cdf, u, right=True)
    below = th.clamp_min(inds-1, 0)
    above = th.clamp_max(inds, N_samples_)

    inds_sampled = th.stack([below, above], -1).reshape(N_rays, 2 * N_importance)
    cdf_g        = th.gather(cdf, 1, inds_sampled).reshape(N_rays, N_importance, 2)
    bins_g       = th.gather(bins, 1, inds_sampled).reshape(N_rays, N_importance, 2)

    denom = cdf_g[...,1] - cdf_g[...,0]
    denom[denom < eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                            # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0]) / denom * (bins_g[...,1] - bins_g[...,0])
    return samples


def unify_samples(samples, fine_samples):
    all_samples = []
    for s, fs in zip(samples, fine_samples):
        all_samples += [th.cat([s, fs], dim=-2)]

    # assuming the first item is depth
    _, indices = th.sort(all_samples[0], dim=-2)
    for i in range(len(all_samples)):
        all_samples[i] = th.gather(
            all_samples[i], -2, 
            repeat(indices, 'b n s () -> b n s d', d=all_samples[i].size(-1))
        )
    return all_samples


def volume_rendering_integration(colors, densities, depths, rendering_options):
    # https://github.com/NVlabs/eg3d/blob/main/eg3d/training/volumetric_rendering/ray_marcher.py
    # it seems to use middle point for estimating colors (?)
    deltas         = depths[:, :, 1:] - depths[:, :, :-1]
    colors_mid     = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
    densities_mid  = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
    depths_mid     = (depths[:, :, :-1] + depths[:, :, 1:]) / 2

    clamp_mode = rendering_options.get('clamp_mode', 'softplus') 
    if clamp_mode == 'softplus':
        densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
    elif clamp_mode == 'exp_truncated':
        densities_mid = th.exp(5 - F.relu(5 - (densities_mid - 1)))  # up-bound = 5, also shifted by 1
    else:
        densities_mid = F.relu(densities_mid)
    
    # volume integration
    density_delta   = densities_mid * deltas
    alpha           = 1 - th.exp(-density_delta)
    alpha_shifted   = th.cat([th.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
    weights         = alpha * th.cumprod(alpha_shifted, -2)[:, :, :-1]
    composite_rgb   = th.sum(weights * colors_mid, -2)
    composite_depth = th.sum(weights * depths_mid, -2) / weights.sum(2)

    # clip the composite to min/max range of depths
    composite_depth = th.nan_to_num(composite_depth, float('inf'))
    composite_depth = th.clamp(composite_depth, th.min(depths), th.max(depths))

    if rendering_options.get('white_back', False):
        composite_rgb = composite_rgb + 1 - weights.sum(2)
    composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)   # do we need this?
    return composite_rgb, composite_depth, weights


def depth_to_normal_image(ray_origins, ray_dirs, depths):
    ray_origins = rearrange(ray_origins, 'b (h w) c -> b c h w', h=depths.size(-2), w=depths.size(-1))
    ray_dirs = rearrange(ray_dirs, 'b (h w) c -> b c h w', h=depths.size(-2), w=depths.size(-1))
    img = ray_origins + ray_dirs * depths
    shift_l, shift_r = img[:,:,2:,:], img[:,:,:-2,:]
    shift_u, shift_d = img[:,:,:,2:], img[:,:,:,:-2]
    diff_hor = F.normalize(shift_r - shift_l, dim=1)[:, :, :, 1:-1]
    diff_ver = F.normalize(shift_u - shift_d, dim=1)[:, :, 1:-1, :]
    normal = -th.cross(diff_hor, diff_ver, dim=1)
    img = F.normalize(normal, dim=1)
    return img