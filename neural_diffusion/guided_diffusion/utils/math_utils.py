'''
 Copyright (C) 2022 Apple Inc. All rights reserved. 
 * @Last Modified by:   Jiatao Gu 
 * @Last Modified time: 2022-08-08 18:47:06 
'''
import torch as th
import numpy as np


def transform_vectors(matrix: th.Tensor, vectors4: th.Tensor) -> th.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = th.matmul(vectors4, matrix.T)
    return res


def normalize_vecs(vectors: th.Tensor) -> th.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (th.norm(vectors, dim=-1, keepdim=True))

def torch_dot(x: th.Tensor, y: th.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)


def get_ray_limits_box(rays_o: th.Tensor, rays_d: th.Tensor, box_side_length):
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)


    bb_min = [-1*(box_side_length/2), -1*(box_side_length/2), -1*(box_side_length/2)]
    bb_max = [1*(box_side_length/2), 1*(box_side_length/2), 1*(box_side_length/2)]
    bounds = th.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
    is_valid = th.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)

    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()

    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]

    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]
    tymax = (bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]

    # Resolve parallel rays.
    is_valid[th.logical_or(tmin > tymax, tymin > tmax)] = False

    # Use the shortest intersection.
    tmin = th.max(tmin, tymin)
    tmax = th.min(tmax, tymax)

    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]
    tzmax = (bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]

    # Resolve parallel rays.
    is_valid[th.logical_or(tmin > tzmax, tzmin > tmax)] = False

    # Use the shortest intersection.
    tmin = th.max(tmin, tzmin)
    tmax = th.min(tmax, tzmax)

    # Mark invalid.
    tmin[th.logical_not(is_valid)] = -1
    tmax[th.logical_not(is_valid)] = -2

    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)


def linspace(start: th.Tensor, stop: th.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in Pyth.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = th.arange(num, dtype=th.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but thscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N   = x_shape[-1]
    x   = x.contiguous().view(-1, N)
    v   = th.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc  = th.view_as_real(th.fft.fft(v, dim=1))  # add this line
    k   = - th.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = th.cos(k)
    W_i = th.sin(k)
    V   = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N   = x_shape[-1]
    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k     = th.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r   = th.cos(k)
    W_i   = th.sin(k)
    V_t_r = X_v
    V_t_i = th.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = th.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    v = th.fft.irfft(th.view_as_complex(V), n=V.shape[1], dim=1)   # add this line
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    return x.view(*x_shape)


def gaussian_filter(u, t):
    K       = u.size(-1)
    freq    = np.pi * th.linspace(0, K-1, K).to(device=u.device) / K
    freq_ss = freq[:, None] ** 2 + freq[None, :] ** 2
    freq_ss = freq_ss[None, None]
    u_proj  = dct(dct(u, norm='ortho').transpose(-1, -2), norm='ortho').transpose(-1, -2)
    if isinstance(t, th.Tensor) and t.ndim == 1:
        t   = t[:, None, None, None]
    u_proj  = th.exp(-freq_ss * t) * u_proj
    u_recon = idct(idct(u_proj, norm='ortho').transpose(-1, -2), norm='ortho').transpose(-1, -2)
    return u_recon


def gaussian_downsample(x, factor=2, t=1):
    """ 
    Progressively applying Gasussian filter and subsampling
    x: N x C x H x W
    """
    while factor != 1:
        assert factor % 2 == 0
        x = gaussian_filter(x, t=t)[..., ::2, ::2]
        factor = factor // 2
    return x