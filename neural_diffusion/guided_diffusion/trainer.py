import copy
import os
import numpy as np
import blobfile as bf
import torch as th
import torch.distributed as dist

from omegaconf import OmegaConf
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torchvision import utils
from einops import rearrange

from functools import partial
from .utils import dist_util
from . import logger
from .utils.fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .diffusion.resample import LossAwareSampler, UniformSampler


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,     # batch_size per GPU
        microbatch,     # used for gradient accumulation
        lr,             # learning rate for AdamW
        ema_rate,       # using EMA
        subsample_pixels=None,
        log_interval=10,
        image_interval=500,
        sample_interval=10000,
        save_interval=10000,
        resume_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        max_steps=3000000,
        cfg_dropout=None,
        use_cross_attention=False,
        cond_model=None,
    ):
        # setup modules and hyperparameters
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.image_size = self.model.image_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.use_cross_attention = use_cross_attention
        self.cond_model = cond_model
        self.subsample_pixels = subsample_pixels
        if self.subsample_pixels is not None:
            self.subsampler = PixelSampler(self.model.image_size ** 2, self.subsample_pixels)

        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.image_interval = image_interval
        self.sample_interval = sample_interval

        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.cfg_dropout = cfg_dropout

        self.step = 0
        self.resume_step = 0
        self.max_steps = max_steps
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:   # if resume_checkpoint is provided, this value will be overrided.
            self.step = self.resume_step  # starting from resume step
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True   # by default, distributed training even with single GPU
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False if 
                    getattr(self.model, 'is_multi_stage', False) else True,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        # self.reporter = MemReporter(self.model)

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint if self.resume_checkpoint else find_resume_checkpoint()
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint} | load step {self.resume_step}")
            self.model.load_state_dict(dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev()))
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = self.resume_checkpoint if self.resume_checkpoint else find_resume_checkpoint()
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint if self.resume_checkpoint else find_resume_checkpoint()
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step//1000:06}K.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step < self.lr_anneal_steps
        ) and (self.step < self.max_steps):
            batch, cond = next(self.data)   # fetch the next batch data
            if batch.shape[-2]!=self.image_size:
                logger.log(f"Warning: a batch's length is {batch.shape[-2]} which is different than {self.image_size}")
                continue
            if len(batch.shape) == 5:
                batch = batch.permute(1,0,2,3,4)
            batch = batch.reshape(-1,batch.shape[-3], batch.shape[-2],batch.shape[-1])
            if 'y' in cond:
                if isinstance(cond['y'][0][0], str):
                    cond['y'] = [x for l in cond['y'] for x in l]
                    cond['y'] = self.model.cond_tokenization(cond['y'])['input_ids']
                    cond['y'] = th.tensor(cond['y'])
                else:
                    cond['y'] = cond['y'].view(-1, cond['y'].shape[-1])
                cond['use_cross_attention'] = self.use_cross_attention
            outputs = self.run_step(batch, cond)      # main loop  
            channel_dim = outputs.shape[-1]//256
            # torch.Size([3, 512, 1, 16, 768]) => torch.Size([3, 512, 3, 64, 64])
            if outputs.shape[2] == 1:
                outputs = outputs.permute([0,1,2,4,3]).view([outputs.shape[0], outputs.shape[1], channel_dim, 256, -1]).permute([0,1,2,4,3]).reshape([outputs.shape[0], outputs.shape[1], channel_dim, 64, -1])
            # logging
            if self.step % self.log_interval == 0:
                kvs = logger.dumpkvs()
        
            
            # output denoised image to tensorboards
            if (self.step % self.image_interval == 0) and ('tensorboard' in logger.get_current().output_formats):
                outputs = rearrange(outputs, 'n b c h w -> b c h (n w)')
                outputs = utils.make_grid(
                    outputs, nrow=dividable(batch.size(0))[0], value_range=(-1, 1), normalize=True)
                logger.get_current().output_formats['tensorboard'].writeimage("denoising", outputs, self.step)
            
            # sample images to tensorboards;
            if (self.step % self.sample_interval == 0): # and (self.step > 0):
                max_bsz = {512: 6, 256: 16, 128: 16, 64: 16, 16: 16}[self.image_size]
                if self.subsample_pixels is not None:  # training use subsample.
                    max_bsz = 1
                batch, cond = batch[:max_bsz], {k: v[:max_bsz] for k, v in cond.items() if not isinstance(v, bool)}
                cur_state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
                for key, param in cur_state_dict.items():
                    cur_state_dict[key]= param.clone()   # make sure not to override the parameters
                for rate, params in zip(self.ema_rate, self.ema_params):
                    ema_state_dict = self.mp_trainer.master_params_to_state_dict(params)
                    self.model.load_state_dict(ema_state_dict)
                    self.sample_step(batch, cond, f'sampling_ema_{rate}')
                self.model.load_state_dict(cur_state_dict)
                del cur_state_dict  # potentially save memory (?)
                # self.sample_step(batch, cond)  # sample with current model

            # save checkpoints
            if (self.step % self.save_interval == 0) and (self.step > self.resume_step):
                self.save()
                # Run for a finite amount of time in integration tests
                # This is from the original code, which uses a weird way of terminating training.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        outputs = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        # self.reporter.report()
        return outputs

    @th.no_grad()
    def sample_step(self, batch, cond, name='sampling'):
        self.model.eval()
        sample_fn = self.diffusion.p_sample_loop if self.diffusion.sampling_type == 'discrete' else \
            partial(self.diffusion.ddpm_sample, sample_steps=250, eta=1)    # DDPM sampler
        samples = sample_fn(self.model, batch.size(), clip_denoised=True, model_kwargs=cond, progress=True)
        channel_dim = samples.shape[-1]//256
        if samples.shape[1] == 1:

            samples_print = samples.permute([0,1,3,2]).view([samples.shape[0], channel_dim, 256, -1]).permute([0,1,3,2]).reshape([samples.shape[0], channel_dim, 64, -1])
            batch_print = batch.permute([0,1,3,2]).view([samples.shape[0], channel_dim, 256, -1]).permute([0,1,3,2]).reshape([samples.shape[0], channel_dim, 64, -1])
        else:
            samples_print = samples
            batch_print = batch


        if len(cond) > 0:  # also output real image
            samples_print = th.cat([samples_print, batch_print.to(samples.device)], -1)

        if self.diffusion.sampling_type == 'continuous':
            noise = th.from_numpy(np.random.RandomState(1373).randn(*batch.shape)).to(samples.device).float()
            samples_ddim = self.diffusion.ddpm_sample(
                self.model, batch.size(), noise=noise, clip_denoised=True, 
                model_kwargs=cond, progress=True, sample_steps=250, eta=0)  # DDIM sampling
            channel_dim = samples.shape[-1]//256
            if samples.shape[1] == 1:
                samples_ddim = samples_ddim.permute([0,1,3,2]).view([samples.shape[0], channel_dim, 256, -1]).permute([0,1,3,2]).reshape([samples.shape[0], channel_dim, 64, -1])
            if len(cond) > 0:  # also output real image
                samples_ddim = th.cat([samples_ddim, batch_print.to(samples.device)], -1)
        if ('tensorboard' in logger.get_current().output_formats):
            logger.get_current().output_formats['tensorboard'].writeimage(
                name, utils.make_grid(samples_print, nrow=dividable(batch_print.size(0))[0], 
                value_range=(-1, 1), normalize=True), self.step, save_png=True)
            if self.diffusion.sampling_type == 'continuous':
                logger.get_current().output_formats['tensorboard'].writeimage(
                    name + '_ddim', utils.make_grid(samples_ddim, nrow=dividable(batch_print.size(0))[0], 
                    value_range=(-1, 1), normalize=True), self.step, save_png=True)


        self.model.train()
        return samples

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        outputs = []
        for i in range(0, batch.shape[0], self.microbatch):  # gradient accumulation to save memory
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: (v if isinstance(v, bool) else v[i : i + self.microbatch].to(dist_util.dev()))
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            compute_losses = partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                cfg_dropout=self.cfg_dropout,
                subsample=None if self.subsample_pixels is None else \
                    self.subsampler.nextids_batch(self.batch_size).to(dist_util.dev())
            )

            if last_batch or not self.use_ddp:
                losses, out = compute_losses()
            else:
                with self.ddp_model.no_sync():   # async training, speed-up?
                    losses, out = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):   # loss aware sampler (not used.)
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()   # this weights is from the sampler, typically always 1.
            log_loss_dict(
                self.diffusion, t, {k: v for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            outputs.append(out)
        return th.cat(outputs, 1)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.step / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.set_steps(self.step)  # set step for tensorboard
        logger.logkv("iter", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)
        logger.logkv("gpumem",  th.cuda.max_memory_allocated(dist_util.dev()) / 2**30)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{((self.step)//1000):06d}K.pt"
                else:
                    filename = f"ema_{rate}_{((self.step)//1000):06d}K.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{((self.step)//1000):06d}K.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        if split1[-1] == 'K':
            return int(split1[:-1]) * 1000
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def save_user_config(config, name):
    bf.makedirs(f'{get_blob_logdir()}/config/{name}')
    OmegaConf.save(config=config, f=f'{get_blob_logdir()}/config/{name}/user.yaml')


def find_resume_checkpoint(log_dir=None, temp='model'):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    if log_dir is None:
        log_dir = get_blob_logdir()
    
    # automatically load the latest checkpoints
    checkpoints = sorted(bf.glob(log_dir + f'/{temp}*.pt'))
    if len(checkpoints) > 0:
        return checkpoints[-1]
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step//1000):06d}K.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def dividable(n):
    for i in range(int(np.sqrt(n)), 0, -1):
        if n % i == 0:
            break
    return i, n // i


class PixelSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = th.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]        

    def nextids_batch(self, batch_size):
        ids = [self.nextids() for _ in range(batch_size)]
        return th.stack(ids)