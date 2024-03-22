import torch as th
import torch.nn.functional as F
import math
import numpy as np
from ..nn import mean_flat
from einops import repeat, rearrange
from tqdm.auto import trange
from ..utils.math_utils import gaussian_filter
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from guided_diffusion.diffusion.gaussian_diffusion import (
    GaussianDiffusion, LossType, ModelMeanType, ModelVarType
)
# --------------------------------------------------------- #
# helper functions
# --------------------------------------------------------- #

def log_snr_to_alpha_sigma(log_snr):
    """Returns the scaling factors for the clean image and for the noise, given
    the log SNR for a timestep."""
    return log_snr.sigmoid().sqrt(), log_snr.neg().sigmoid().sqrt()


def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return th.cos(t * math.pi / 2).clamp(min=0, max=1), th.sin(t * math.pi / 2).clamp(min=0, max=1)

def t_to_alpha_sigma_beta(t):
    """continuous approximation of the beta-linear schedule"""
    alpha_2 = th.exp(-1e-4-10*t**2).clamp(min=0, max=1)
    alpha = alpha_2.sqrt()
    sigma = (1 - alpha_2).sqrt()
    return alpha, sigma


def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return th.atan2(sigma, alpha) / math.pi * 2


def alpha_sigma_to_log_snr(alpha, sigma):
    """Returns a log snr, given the scaling factors for the clean image and for
    the noise."""
    return th.log(alpha**2 / sigma**2)

def get_noise_schedule(t, mode='alpha', logsnr_range=None, shift=1):
    """
    t: diffusion timestep
    mode: alpha --> cosine schedule, beta --> linear schedule
    """
    if logsnr_range is not None:
        logsnr_min, logsnr_max = logsnr_range
        t_min = np.arctan(np.exp(-0.5 * logsnr_max)) * 2 / np.pi 
        t_max = np.arctan(np.exp(-0.5 * logsnr_min)) * 2 / np.pi 
        t = t_min + t * (t_max - t_min)

    alpha, sigma = t_to_alpha_sigma(t) if mode == 'alpha' else t_to_alpha_sigma_beta(t)
    
    if shift > 1:
        logsnr = alpha_sigma_to_log_snr(alpha, sigma)
        logsnr = logsnr + 2 * np.log(1 / shift)
        alpha, sigma = log_snr_to_alpha_sigma(logsnr)
    
    return alpha, sigma

def get_ddpm_schedule(ddpm_t):
    """Returns timesteps for the noise schedule from the DDPM paper."""
    log_snr = -th.special.expm1(1e-4 + 10 * ddpm_t**2).log()
    alpha, sigma = log_snr_to_alpha_sigma(log_snr)
    return alpha_sigma_to_t(alpha, sigma)


def get_subsampled_img(image, subsample):
    # BCHW -> BCN1
    if subsample.dim() == 2:
        subsample = repeat(subsample, 'b n -> b n d', d=3)
    if subsample.size(-1) == 3:
        subsample = subsample.transpose(-2, -1)   
    return rearrange(image, 'b c h w -> b c (h w)').gather(2, subsample).unsqueeze(-1)



class ContinuousDiffusion(GaussianDiffusion):
    """
    re-write the sampler for more flexiable sampling support
    https://github.com/crowsonkb/v-diffusion-pytorch    
    """
    def __init__(self, step_cond='time', linear_beta=False, noise_shift=1, **kwargs):
        super().__init__(**kwargs)

        self.sampling_type = 'continuous'
        self.step_cond = step_cond
        self.mode = 'beta' if linear_beta else 'alpha'
        self.noise_shift = noise_shift
        self.t_min, self.t_max  = 0.0005, 0.9995
        self.logsnr_max = -np.log(np.tan(self.t_min * np.pi / 2)) * 2
        self.logsnr_min = -np.log(np.tan(self.t_max * np.pi / 2)) * 2
        # self.sigma_factor = sigma_factor
        if self.betas is None:
            self.diffusion_type = 'continuous'
            self.num_timesteps = 1
            # for simplicity, we use a fixed alpha scheduler
            # alpha_t = cos(t * pi / 2) for t ~ [0, 1] so we don't need beta_t


    def t_to_alpha_sigma(self, t):
        alpha, sigma = get_noise_schedule(
            t, self.mode, 
            (self.logsnr_min, self.logsnr_max), 
            self.noise_shift)
        return alpha, sigma


    def get_model_output(self, model, x, t, alphas, sigmas, subsample=None, cfg_weight=None, cfg_dropout=None, **model_kwargs):
        if hasattr(model, 'module'):
            cond_feature = model.module.cond_feature
        else:
            cond_feature = model.cond_feature

        if not model_kwargs['y'][0][0].dtype.is_floating_point:
            if hasattr(model, 'module'):
                # encode with cond model
                model_kwargs['y'] = model.module.forward_cond_encoding(model_kwargs['y'])
            else:
                model_kwargs['y'] = model.forward_cond_encoding(model_kwargs['y'])
        
        if model.training:
            # ignore variance if model predicts
            if (cfg_dropout is None) or (cfg_dropout == 0.0) or ('y' not in model_kwargs): # @JG
                model_output = model(x, self._scale_timesteps(t), subsample=subsample, **model_kwargs)
            else:
                assert 'y' in model_kwargs
                labels = model_kwargs['y']
                drop_ids = th.rand(labels.shape[0]) < cfg_dropout
                if cond_feature:  # y is a cont feature
                    non_emb = model.module.non_emb if hasattr(model, 'module') else model.non_emb
                else:
                    non_emb = np.zeros(labels.shape[1], dtype=np.float32)
                    non_emb[-1] = 1
                labels = th.where(drop_ids.unsqueeze(-1).cuda(), th.Tensor(non_emb).cuda(), labels)
                model_kwargs['y'] = labels
                model_output = model(x, self._scale_timesteps(t), subsample=subsample, **model_kwargs)

        else:
            # ignore variance if model predicts
            if (cfg_weight is None) or (cfg_weight == 1.0):
                model_output = model(x, self._scale_timesteps(t), subsample=subsample, **model_kwargs)
            else:
                assert 'y' in model_kwargs
                if cond_feature:  # y is a cont feature
                    y_un = model.non_emb.expand(*model_kwargs['y'].shape)
                    z_in, t_in, y_in = th.cat([x] * 2), th.cat([t] * 2), th.cat([model_kwargs['y'], y_un])
                else:
                    y_un = th.zeros_like(model_kwargs['y']); y_un[:, -1] = 1  # @JG
                    z_in, t_in, y_in = th.cat([x] * 2), th.cat([t] * 2), th.cat([model_kwargs['y'], y_un])
                pred_c, pred_un = model(z_in, self._scale_timesteps(t_in), y=y_in, use_cross_attention=model_kwargs['use_cross_attention']).chunk(2)
                pred = pred_un + cfg_weight * (pred_c - pred_un)  # reweighting eps
                model_output = pred
                

        if subsample is not None:
            x = get_subsampled_img(x, subsample)
        
        # get all values
        if self.model_mean_type == ModelMeanType.START_X:
            pred = model_output
            eps  = (x - alphas * model_output) / sigmas
        elif self.model_mean_type == ModelMeanType.EPSILON:
            eps  = model_output
            pred = (x - sigmas * model_output) / alphas
        # elif self.model_mean_type == ModelMeanType.VELOCITY:
        #     pred = alphas * x - sigmas * model_output
        #     eps  = sigmas * x + alphas * model_output
        else:
            raise NotImplementedError
        return pred, eps #, pred_un


    def get_loss(self, x_start, noise, pred, eps, log_snr, subsample=None):
        # P2 weighting (https://arxiv.org/abs/2204.00227) ==> additional weigting based on SNR
        weight = 1 / ((self.p2_k + log_snr.exp()) ** self.p2_gamma)
        if self.target_type == ModelMeanType.START_X:
            if subsample is not None:
                x_start = get_subsampled_img(x_start, subsample)
            loss = (x_start - pred) ** 2 * log_snr.exp() * weight   # loss is SNR weighted (TODO: can be modified later.)
        elif self.target_type == ModelMeanType.EPSILON:
            if subsample is not None:
                noise = get_subsampled_img(noise, subsample)
            loss = (noise - eps) ** 2 * weight                      # eps loss does not have such weight
        else:
            raise NotImplementedError
        return loss

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, subsample=None, cfg_dropout=None):
        # prepare parameters
        assert self.diffusion_type == 'continuous', "this training loss is only for training continuous diffusion model"
        assert self.model_var_type == ModelVarType.FIXED_SMALL, "continuous diffusion does not support learning variance."
        assert self.loss_type in [LossType.MSE, LossType.RESCALED_MSE], "do not suppor standard KL loss"
        assert self.target_type in [ModelMeanType.START_X, ModelMeanType.EPSILON], "do not support predicting the mean of x_{t-1}"
        assert self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON], "do not support predicting the mean of x_{t-1}"

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        terms = {}

        # get the model's output
        alphas, sigmas = self.t_to_alpha_sigma(t)
        alphas, sigmas = alphas[:, None, None, None], sigmas[:, None, None, None]
        log_snr = alpha_sigma_to_log_snr(alphas, sigmas)
    
        if self.step_cond == 'time':
            steps = t * 1000.  # for now still use 0-1000.  ## TODO: try log-SNR conditioning if this works
        elif self.step_cond == 'logSNR':
            # steps = log_snr[:, 0, 0, 0]
            steps = (log_snr[:, 0, 0, 0] - self.logsnr_min) / (self.logsnr_max - self.logsnr_min) * 1000
        else:
            raise NotImplementedError

        if subsample is not None:
            subsample = repeat(subsample, 'b n -> b n d', d=3)

        x = x_start * alphas + noise * sigmas
        pred, eps = self.get_model_output(model, x, steps, alphas, sigmas, subsample, cfg_dropout=cfg_dropout, **model_kwargs)
        loss = self.get_loss(x_start, noise, pred, eps, log_snr, subsample=subsample)
        
        terms["loss"] = terms['mse'] = mean_flat(loss)
        
        if subsample is not None:
            pred = rearrange(th.ones_like(x).neg(), 'b c h w -> b c (h w)').scatter_(
                2, subsample.transpose(-2, -1), pred.squeeze(-1))
            pred = pred.reshape(x.size())
        output_tensor = th.stack([x_start, x, pred], 0).detach()
        
        # ********* optional ********
        _model = model.module if isinstance(model, DDP) else model
        if getattr(_model, 'depth_output', None) is not None:
            depth = _model.depth_output
            if subsample is not None:
                depth = rearrange(th.ones_like(x).neg(), 'b c h w -> b c (h w)').scatter_(
                    2, subsample.transpose(-2, -1), depth.squeeze(-1))
                depth = depth.reshape(x.size())
            output_tensor = th.cat([output_tensor, depth[None, ...]], 0)        
        return terms, output_tensor

    @th.no_grad()
    def ddpm_sample(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        full_traj=False,
        eta=1,
        sample_steps=None,
        cosine_sample=False,
        extra_noise=None,
        cfg_weight=None,
        time_power=1
        ):

        # get the noise schedule
        if device is None:
            device = next(model.parameters()).device

        if sample_steps is None:
            sample_steps = self.num_timesteps

        if (self.diffusion_type == 'discrete') and (not cosine_sample):  
            # using the standard discrete sampling used in training
            steps = list(np.linspace(0, 1, sample_steps) * (self.num_timesteps - 1))
            steps = [int(s) for s in steps[::-1]]
            log_snr = th.from_numpy(self.snr).log().to(device)[steps]
            alphas, sigmas = log_snr_to_alpha_sigma(log_snr)
        
        else:
            continuous_time = th.linspace(0.9995, 0, sample_steps + 1, device=device)[:-1]
            continuous_time = continuous_time**time_power
            alphas, sigmas  = self.t_to_alpha_sigma(continuous_time)
            log_snr = alpha_sigma_to_log_snr(alphas, sigmas)
            if self.diffusion_type == 'discrete':
                model_log_snr = th.from_numpy(self.snr).log().to(device)
                steps = th.searchsorted(-model_log_snr, -log_snr)  # searchsorted need to be increasing..
            else:
                if self.step_cond == 'time':
                    steps = continuous_time * 1000.
                elif self.step_cond == 'logSNR':
                    # steps = log_snr
                    steps = (log_snr - self.logsnr_min) / (self.logsnr_max - self.logsnr_min) * 1000
                else:
                    raise NotImplementedError

        # prepare for the sampling loop
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            x = noise
        else:
            x = th.randn(*shape, device=device)
        ts = x.new_ones([x.shape[0]], device=device)
        model_kwargs = {k: v.to(device) if not isinstance(v, bool) else v for k, v in model_kwargs.items()}

        # sampling loop
        outputs = []
        # for i in range(len(steps)):
        for i in trange(len(steps), desc='Continuous Diffusion Sampler', disable=(not progress)):
            output = []

            # Get the model output (v, the predicted velocity)
            t = ts * steps[i]
            alpha, sigma = alphas[i], sigmas[i]
            with th.no_grad():
                # pred, eps = self.get_model_output(model, x, t, alpha, sigma, cfg_weight=cfg_weight, **model_kwargs)
                pred, eps = self.get_model_output(model, x, t, alpha, sigma, cfg_weight=cfg_weight, **model_kwargs)
                # output += [pred]

            def process_xstart(x, i, dynamic = True):
                if denoised_fn is not None:     # additional denoising function?
                    x = denoised_fn(x)
                if clip_denoised:               # clipping the gaussian noise?
                    if dynamic and i != len(steps)-1:
                        if x.dtype == th.float32 or x.dtype == th.float64:
                            s = th.quantile(abs(x), 0.995)
                            s = max(s, 1)
                            # print(f"{s}")
                            return x.clamp(-s, s) #/s
                        # print(f"Warning: X at step {i} is not float: {x} \n")
                    return x.clamp(-1, 1)
                return x

            # 
            pred  = process_xstart(pred, i)  # clip the value to (-1, 1) ??
            # output += [pred]
            output += [pred]

            
            if i < len(steps) - 1:
                # If eta > 0, adjust the scaling factor for the predicted noise
                # downward according to the amount of additional noise to add
                ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                    (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
                adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

                # Recombine the predicted noise and predicted denoised image in the
                # correct proportions for the next step
                x = pred * alphas[i + 1] + eps * adjusted_sigma

                # Add the correct amount of fresh noise
                if eta:
                    if extra_noise is None:
                        x += th.randn_like(x) * ddim_sigma
                    else:
                        x += extra_noise[i] * ddim_sigma
                output += [x] 
            else:
                output += [pred]

            outputs += [th.cat(output, 2).detach()]

        if full_traj:
            return outputs

        # If we are on the last timestep, output the denoised image
        return pred


class ProgressiveContinuousDiffusion(ContinuousDiffusion):
    """
    Experimental code that progressively increasing the resolution during the diffusion process
    """
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, subsample=None, cfg_dropout=None):
        _model = model.module if isinstance(model, DDP) else model
        assert hasattr(_model, 'stages'), "stages are defined in model"
        
        # during training, only sample one batch for each stage
        stage, t = _model.get_stage(th.rand(size=(1,), device=t.device), t)
        x_target = _model.get_target(x_start, stage)
        x_start, x_states = _model.get_start(x_start, stage, t, return_states=True)
        
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs['stage'] = stage
        model_kwargs['x_states'] = x_states
        if noise is None:
            noise = th.randn_like(x_start)

        # get the model's output
        alphas, sigmas = self.t_to_alpha_sigma(t)
        alphas, sigmas = _model.get_transformed_alpha_sigma(alphas, sigmas, stage)  # (optional) adjust alpha/sigma according to timestep
        alphas, sigmas = alphas[:, None, None, None], sigmas[:, None, None, None]
        log_snr = alpha_sigma_to_log_snr(alphas, sigmas)
    
        if self.step_cond == 'time':
            steps = t * 1000.  # for now still use 0-1000.  ## TODO: try log-SNR conditioning if this works
        elif self.step_cond == 'logSNR':
            steps = (log_snr[:, 0, 0, 0] - self.logsnr_min) / (self.logsnr_max - self.logsnr_min) * 1000
        else:
            raise NotImplementedError

        if subsample is not None:
            subsample = repeat(subsample, 'b n -> b n d', d=3)

        # model forward
        x = x_start * alphas + noise * sigmas
        pred, x_loss = model(x, self._scale_timesteps(steps), subsample=subsample, **model_kwargs)
        terms, pred = self.get_loss(x, x_start, x_target, x_loss, noise, pred, log_snr, alphas, sigmas)
        
        # output visualization
        output_tensor = th.stack([F.interpolate(xi, size=_model.stages[0][1], mode='nearest') 
                            for xi in [x_start, x, x_target, pred]], 0).detach()
        return terms, output_tensor

    def get_loss(self, x, x_start, x_target, x_loss, noise, pred, log_snr, alphas, sigmas, **unused):
        weight = 1 / ((self.p2_k + log_snr.exp()) ** self.p2_gamma)
        losses = {}

        if self.target_type == ModelMeanType.START_X:
            loss = (x_target - pred) ** 2 * log_snr.exp() * weight   # loss is SNR weighted (TODO: can be modified later.)
            losses['mse_x0'] = mean_flat(loss)
        elif self.target_type == ModelMeanType.EPSILON:
            assert x.size(-1) == pred.size(-1), "only works in the same resolution"
            if pred.size(1) == 3:
                eps  = pred    
                loss = (noise - eps) ** 2 * weight    # eps loss does not have such weight
                pred = (x - eps * sigmas) / alphas    # get the estimated pred
            else:
                eps, laplacian = pred[:, :3], pred[:, 3:]
                pred = (x - eps * sigmas) / alphas + laplacian
                loss = (x_target - pred) ** 2 * log_snr.exp() * weight
            losses['mse_eps'] = mean_flat(loss)
        elif self.target_type == ModelMeanType.EPSILON_DOUBLE:
            assert x.size(-1) == pred.size(-1), "only works in the same resolution"
            assert pred.size(1) == 6, "model needs to predict eps and laplacian"
            eps, laplacian = pred[:, :3], pred[:, 3:]
            loss_eps = (noise - eps) ** 2 * weight 
            loss_lap = (x_target - (x_start + laplacian)) ** 2 * log_snr.exp() * weight
            pred = (x - eps * sigmas) / alphas + laplacian
            losses['mse_eps'] = mean_flat(loss_eps)
            losses['mse_lap'] = mean_flat(loss_lap)
            
        else:
            raise NotImplementedError
        
        # upsampler loss (?)
        if x_loss is not None:
            x_loss = {k: mean_flat(v) for k, v in x_loss.items()}
            losses.update(x_loss)
        
        # all the loss term
        losses['loss'] = sum({losses[k] for k in losses})
        return losses, pred

    @th.no_grad()
    def ddpm_sample(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        full_traj=False,
        eta=1,
        sample_steps=None,
        extra_noise=None,
        ):

        # get the noise schedule
        if device is None:
            device = next(model.parameters()).device
        if sample_steps is None:
            sample_steps = self.num_timesteps

        # get continuous time add fix the boundary
        continuous_time = th.linspace(0.9995, 0, sample_steps + 2 - len(model.stages), device=device)
        continuous_time = th.cat([continuous_time, model.stage_timespaces[1:-1].to(device) - 1e-6], 0)
        continuous_time = th.sort(continuous_time, descending=True)[0]
        alphas, sigmas  = self.t_to_alpha_sigma(continuous_time[:-1])
        log_snr = alpha_sigma_to_log_snr(alphas, sigmas)
        stages  = model.stages

        if self.step_cond == 'time':
            steps = continuous_time[:-1] * 1000.
        elif self.step_cond == 'logSNR':
            steps = (log_snr - self.logsnr_min) / (self.logsnr_max - self.logsnr_min) * 1000
        else:
            raise NotImplementedError

        # prepare for the sampling loop
        assert isinstance(shape, (tuple, list))
        if noise is None:
            noise = th.randn(*shape, device=device)
        l = stages[0][0] // stages[-1][0]
        x = model.get_initial_noise(noise[:,:,::l,::l])  # get initial noise
        ts = x.new_ones([x.shape[0]], device=device)
        model_kwargs = {k: v.to(device) for k, v in model_kwargs.items()}

        # sampling loop
        outputs, G = [], stages[0][0]
        for i in trange(len(steps), desc='Continuous Diffusion Sampler', disable=(not progress)):
            output = []
            
            # Get the model output (v, the predicted velocity)
            t = ts * steps[i]
            curr_stage = model.get_stage(continuous_time[i])
            next_stage = model.get_stage(continuous_time[i + 1])
            alpha, sigma = alphas[i], sigmas[i]
            if i == len(steps) - 1:
                alpha_m1, sigma_m1 = alpha, sigma
            else:
                alpha_m1, sigma_m1 = alphas[i+1], sigmas[i+1]

            # (optional) adjust alpha/sigma according to timestep
            alpha, sigma       = model.get_transformed_alpha_sigma(alpha, sigma, curr_stage)
            alpha_m1, sigma_m1 = model.get_transformed_alpha_sigma(alpha_m1, sigma_m1, next_stage)

            with th.no_grad():
                pred, _ = model(x, self._scale_timesteps(t), stage=curr_stage, **model_kwargs)
                if self.model_mean_type == ModelMeanType.EPSILON:
                    # predicting eps
                    assert pred.size(1) == 6, "for now, we assume double noise prediction"
                    eps, laplacian = pred[:, :3], pred[:, 3:]
                    pred_x_start = (x - eps * sigma) / alpha
                    pred = pred_x_start + laplacian

                if self.target_type == ModelMeanType.EPSILON_DOUBLE and full_traj:
                    output += [F.interpolate(laplacian, G)]      
                else:
                    pred_x_start = model.get_start(pred, curr_stage, continuous_time[i])
                    eps  = (x - alpha * pred_x_start) / sigma
                if full_traj:
                    output += [F.interpolate(pred, G)]
                
                # decision at the boundary
                if stages[next_stage][0] > stages[curr_stage][0]:  # apply upsample, include additional noise
                    if pred.size(-1) == pred_x_start.size(-1):     # model does not perform upsample
                        pred = model.get_upsampled_image(pred)
                    l = l // 2
                    eps_us = noise[:,:,::l,::l]
                    if model.noise_type == 'none':                
                        eps_us[:,:,::2,::2] = eps
                    else:
                        # e1 + e2 + e3 + e4 = 2 * eps
                        eps_reshape = rearrange(eps_us, 'b c (h x) (w y) -> b c h w x y', x=2, y=2)
                        # aa = 2 * eps
                        # e1 = aa / 4 + 3 / 4 * eps_reshape[..., 0, 0]; aa = aa - e1; eps_reshape[..., 0, 0] = e1
                        # e2 = aa / 3 + 2 / 3 * eps_reshape[..., 0, 1]; aa = aa - e2; eps_reshape[..., 0, 1] = e2
                        # e3 = aa / 2 + 1 / 2 * eps_reshape[..., 1, 0]; aa = aa - e3; eps_reshape[..., 1, 0] = e3
                        # e4 = aa; eps_reshape[..., 1, 1] = e4
                        eps_delta   = eps / 2 - eps_reshape.sum((-1, -2)) / 4   # eps/2 is the target mean vector (?)
                        eps_reshape = eps_reshape + eps_delta[..., None, None]
                        eps_us = rearrange(eps_reshape, 'b c h w x y -> b c (h x) (w y)')
                    eps = eps_us
                else:
                    if (model.resize_type == 'interp') and (model.stage_type == 'keep'):  # slightly different. we can avoid doing another downsample (?)
                        ratio_t   = model.get_interp_ratio(curr_stage, continuous_time[i])
                        ratio_tm1 = model.get_interp_ratio(next_stage, continuous_time[i+1])
                        pred = (ratio_t - ratio_tm1) / ratio_t * pred + ratio_tm1 / ratio_t * pred_x_start
                    else:
                        pred = model.get_start(pred, next_stage, continuous_time[i+1]) 

                if pred_x_start.size(-1) < pred.size(-1): # model has upsampled
                    pred_x_start = pred

            def process_xstart(x):
                if denoised_fn is not None:     # additional denoising function?
                    x = denoised_fn(x)
                
                if clip_denoised:               # clipping the gaussian noise?
                    s = th.quantile(abs(x), 0.995)
                    s = max(s, 1)
                    return x.clamp(-s, s) / s
                return x
            
            pred  = process_xstart(pred)  # clip the value to (-1, 1) ??
            
            if i < len(steps) - 1:
                # If eta > 0, adjust the scaling factor for the predicted noise
                # downward according to the amount of additional noise to add
                
                # apply upsample, use DDIM step for safety
                if stages[next_stage][0] > stages[curr_stage][0]:
                    x = pred * alpha_m1 + eps * sigma_m1
                else:
                    # TODO: CHECK correct version of noise?????
                    alpha_t_s   = alpha / alpha_m1
                    sigma_t_s_2 = sigma**2 - alpha_t_s**2 * sigma_m1**2
                    ddim_sigma  = eta * sigma_m1 / sigma * sigma_t_s_2.sqrt()
                    # ddim_sigma = eta * (sigma_m1**2 / sigma**2).sqrt() * (1 - alpha**2 / alpha_m1**2).sqrt()
                    adjusted_sigma = (sigma_m1**2 - ddim_sigma**2).sqrt()

                    # Recombine the predicted noise and predicted denoised image in the
                    # correct proportions for the next step
                    x = pred * alpha_m1 + eps * adjusted_sigma

                    # Add the correct amount of fresh noise
                    if eta:
                        if extra_noise is None:
                            x += th.randn_like(x) * ddim_sigma
                        else:
                            x += extra_noise[i] * ddim_sigma
                if x.isnan().any():
                    from fairseq import pdb;pdb.set_trace()
                # visualization
                # out = x.new_ones(*x.size()[:2], G, G).neg()
                # out[:,:,:x.size(-2),:x.size(-1)] = x
                out = F.interpolate(x, G)
            if full_traj:
                if i < len(steps) - 1:
                    output += [out] 
                else:
                    output += [pred]
                outputs += [th.cat(output, 2).detach()]

        if full_traj:
            return outputs

        # If we are on the last timestep, output the denoised image
        return pred


class MultiStageContinuousDiffusion(ProgressiveContinuousDiffusion):
    """
    Similar to the ProgressiveContinuousDiffusion, we tried to train all the stages directly
    """
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, subsample=None, cfg_dropout=None):
        _model = model.module if isinstance(model, DDP) else model
        assert hasattr(_model, 'stages'), "stages are defined in model"
        
        if noise is None:
            noise = th.randn_like(x_start)
        if model_kwargs is None:
            model_kwargs = {}
        
        stages, i = _model.get_multi_stages(t), 0
        multi_stage_data = []

        # get noise schedule
        timesteps = th.cat([s[1] for s in stages])
        alphas, sigmas = self.t_to_alpha_sigma(timesteps)
        alphas, sigmas = _model.get_transformed_alpha_sigma(alphas, sigmas, stages)  # (optional) adjust alpha/sigma according to timestep
        alphas, sigmas = alphas[:, None, None, None], sigmas[:, None, None, None]
        log_snr = alpha_sigma_to_log_snr(alphas, sigmas)
        if self.step_cond == 'time':
            steps = timesteps * 1000.  # for now still use 0-1000.  ## TODO: try log-SNR conditioning if this works
        elif self.step_cond == 'logSNR':
            steps = (log_snr[:, 0, 0, 0] - self.logsnr_min) / (self.logsnr_max - self.logsnr_min) * 1000
        else:
            raise NotImplementedError

        # prepare inputs and targets
        for stage, timestep in stages:
            l = len(timestep)
            x_real   = x_start[i:i+l]
            x_target = _model.get_target(x_real, stage)
            x_input  = _model.get_start(x_real, stage, timestep)
            x_noise  = F.interpolate(noise[i:i+l], x_input.size(-1), mode='nearest')
            x_alpha, x_sigma, x_log_snr  = alphas[i:i+l], sigmas[i:i+l], log_snr[i:i+l]

            # prepare the model inputs
            x = x_input * x_alpha + x_noise * x_sigma
            multi_stage_data += [{
                'stage': stage, 'x': x, 'x_start': x_input, 'x_target': x_target, 'x_loss': None, 'noise': x_noise,
                'log_snr': x_log_snr, 'alphas': x_alpha, 'sigmas': x_sigma,
                'start_end': [i, i+l]
            }] 
            i += l
            
        # model forward
        preds, _ = model(multi_stage_data, self._scale_timesteps(steps), **model_kwargs)
        
        # get loss functions, output visualization
        total_terms, output_tensors = {'loss': []}, []
        for i, (data, pred) in enumerate(zip(multi_stage_data, preds)):
            terms, pred = self.get_loss(pred=pred, **data)
            output_tensor = th.stack([F.interpolate(xi, size=_model.stages[0][1], mode='nearest') 
                            for xi in [data['x_start'], data['x'], data['x_target'], pred]], 0).detach()
            output_tensors.append(output_tensor)
            for k in terms:
                total_terms[f'{k}.stage{data["stage"]}'] = terms[k]
            total_terms['loss'] += [terms['loss']]

        # TODO: avoid errors from unused parameters
        dummy_loss = 0.0 * sum([p.flatten()[0] for p in model.parameters()])
        total_terms['loss'] = th.cat(total_terms['loss']) + dummy_loss
        output_tensors = th.cat(output_tensors, dim=1)
        return total_terms, output_tensors