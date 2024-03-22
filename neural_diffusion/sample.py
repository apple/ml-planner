"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

from locale import normalize
import os

import hydra
import imageio
import blobfile as bf

from omegaconf import DictConfig, OmegaConf

import tqdm
import numpy as np
import torch as th
import torch.distributed as dist

from einops import repeat
from functools import partial
from guided_diffusion import logger
from guided_diffusion import create_model, create_gaussian_diffusion
from torchvision import utils, io
from guided_diffusion.trainer import dividable, find_resume_checkpoint
from guided_diffusion.utils import dist_util
from guided_diffusion.data import get_dataset

SEED=137

@hydra.main(version_base=None, config_path="conf", config_name="sample_config")
def main(args: DictConfig):
    dist_util.setup_dist()
    rank, world_size = dist_util.get_rank(), dist_util.get_world_size()
    dist_util.setup_seed(SEED + rank)

    logger.configure(dir=args.sample_dir, format_strs=['simple','log'])
    logger.log("creating model and diffusion...")

    assert (not args.model_config.model.class_cond) or (args.data_dir is not None), \
        "class-conditioned generation requires data input"
    dataset = None
    image_size = args.model_config.model.image_size

    if args.data_dir is not None:
        logger.log("creating dataset...")
        dataset = get_dataset(
            data_dir=args.data_dir,
            image_size=image_size,
            class_cond=args.class_cond,
            random_flip=False)
    model = create_model(**args.model_config.model, 
        num_classes=dataset.label_shape if dataset is not None else None)
    diffusion = create_gaussian_diffusion(**args.model_config.diffusion)
    model.get_summary()

    if args.model_path is not None:
        if bf.isdir(args.model_path):  # load the latest EMA checkpoint
            args.model_path = find_resume_checkpoint(args.model_path, 'ema')
        logger.log(f"loading checkpoint from {args.model_path}")
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if args.output_full_traj:
        nrow, ncol = dividable(args.batch_size * 2)
        nrow_real  = nrow if args.batch_size % nrow == 0 else ncol
    else:
        nrow, ncol = dividable(args.batch_size)
        nrow_real  = nrow

    logger.log("start sampling...")
    with tqdm.tqdm(total=args.num_samples, desc="generate samples") as pbar:
        all_images, all_labels, count = [], [], 0
        while count * args.batch_size < args.num_samples:
            model_kwargs = {}
            fname = f'{count:05}_{args.batch_size}'
            if args.class_cond:
                indices = np.random.randint(
                    low=0, high=dataset.local_classes.shape[0], 
                    size=(args.batch_size // nrow_real,))
                labels = np.array([dataset.get_label(l) for l in dataset.local_classes[indices]])
                labels = th.from_numpy(labels).to(device=dist_util.dev())
                labels = repeat(labels, 'b d -> (b c) d', c=nrow_real).contiguous()
                model_kwargs["y"] = labels
                fname += '_cond'

            if diffusion.sampling_type == 'continuous':
                sample_fn = partial(
                    diffusion.ddpm_sample, 
                    sample_steps=args.model_config.diffusion.continuous_steps,
                    eta=args.model_config.diffusion.ddpm_ddim_eta)
                fname += f'_eta{args.model_config.diffusion.ddpm_ddim_eta}'
            else:
                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                fname += f'_eta{1 if not args.use_ddim else 0}'

            shape = (args.batch_size, 3, image_size, image_size)          

            with th.no_grad():
                th.manual_seed(SEED + args.seed)
                sample = sample_fn(
                    model,
                    shape,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    progress=args.diffusion_pbar,
                    full_traj=args.output_full_traj,
                )
            
            # saving png
            if args.output_full_traj:
                last_sample = (utils.make_grid(
                    sample[-1].cpu().clamp(-1, 1)[:, :, -image_size:], nrow=nrow_real, normalize=True, value_range=(-1, 1)) * 255
                    ).permute(1,2,0).to(th.uint8)
                full_sample = th.stack(
                    [utils.make_grid(s.cpu().clamp(-1, 1), nrow=nrow_real, 
                    normalize=True, value_range=(-1,1)) * 255 for s in sample]).permute(0,2,3,1).to(th.uint8)
                
                if args.output_quality is None:
                    io.write_video(os.path.join(logger.get_dir(), fname + '.mp4'), full_sample, fps=15)
                else:
                    # output high-quality video
                    with imageio.get_writer(os.path.join(logger.get_dir(), fname + '.mp4'), 
                        fps=25, quality=args.output_quality) as writer:
                        for i in range(len(full_sample)):
                            writer.append_data(full_sample[i].numpy())
                imageio.imsave(os.path.join(logger.get_dir(), fname + '.png'), last_sample.numpy())
                sample = sample[-1]
            else:
                utils.save_image(
                    sample.clamp(-1, 1), 
                    os.path.join(logger.get_dir(), fname + '.png'), 
                    value_range=(-1,1), normalize=True,
                    nrow=nrow_real)

            # saving npz
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(labels) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, labels)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

            pbar.update((len(all_images) - count) * args.batch_size)
            count = len(all_images)

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")

if __name__ == "__main__":
    main()