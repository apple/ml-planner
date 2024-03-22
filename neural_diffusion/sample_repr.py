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
import re
from functools import partial
from guided_diffusion import logger
from guided_diffusion import create_model, create_gaussian_diffusion
from torchvision import utils, io
from guided_diffusion.trainer import dividable, find_resume_checkpoint
from guided_diffusion.utils import dist_util
from guided_diffusion.data import get_dataset, load_data
import guided_diffusion.data.image_datasets

SEED=137

@hydra.main(version_base=None, config_path="conf", config_name="sample_config")
def main(args: DictConfig, plot_png = False):
    dist_util.setup_dist()
    rank, world_size = dist_util.get_rank(), dist_util.get_world_size()
    dist_util.setup_seed(SEED + rank)

    logger.configure(dir=args.sample_dir, format_strs=['simple','log'])
    logger.log("creating model and diffusion...")

    assert (not args.model_config.model.class_cond) or (args.data_dir is not None), \
        "class-conditioned generation requires data input"
    dataset = None
    image_size = args.model_config.model.image_size
    # image_size = 16

    batch_size = args.batch_size
    if args.data_dir is not None:
        logger.log("creating dataset...")
        dataset = get_dataset(
            args=args,
            data_dir=args.data_dir,
            image_size=image_size,
            class_cond=args.class_cond,
            random_flip=False,
            task_id=args.task_id,
            num_feature=args.num_feature, 
            sentence_len=args.sentence_len,
            load_ckpt=args.load_ckpt if hasattr(args, 'load_ckpt') else None,
            ae_step=args.ae_step,
            cfg_weight=args.cfg_weight,
            output_target=args.print_dataset
            )
        label_shape = dataset.label_shape if dataset is not None else None
        # if args.cfg_weight is not None and isinstance(label_shape, np.int64):  # use classifier-free guidance
        #     label_shape += 1
    model = create_model(**args.model_config.model, 
        num_classes=label_shape)
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
    th.manual_seed(SEED + args.seed)
    num_steps = args.model_config.diffusion.continuous_steps
    args.num_samples = args.batch_size * dataset.ae_args.batch_size_per_gpu    
    data = load_data(dataset, 1, deterministic=True)
    cond_y, cond_t = [], []
    if not args.forward:
        logger.log("start sampling...")
        with tqdm.tqdm(total=args.num_samples, desc="generate samples") as pbar:
            all_reprs, all_labels, all_text, count = [], [], [], 0
            while count < args.num_samples:
                model_kwargs = {}
                fname = f'{count:05}_{args.num_samples}_{num_steps}'
                if args.class_cond:
                    if model.cond_feature:
                        if hasattr(dataset, "local_classes"):  
                            classes = dataset.local_classes[:args.num_samples]  #[6,768,16]
                            labels = th.stack([th.Tensor(c.reshape(-1)) for c in classes]).to(device=dist_util.dev())
                        else: # dataset from TextDataset, cond DIT
                            batch, cond = next(data) 
                            batch = batch.transpose(0,1)
                            # args.batch_size = min(batch_size, args.batch_size * dataset.ae_args.batch_size_per_gpu)
                            batch, cond['y'] = batch[:args.num_samples], cond['y'][:args.num_samples]
                            cond['y'] = [x for l in cond['y'] for x in l]
                            labels = model.cond_tokenization(cond['y'])['input_ids'].to(device=dist_util.dev())
                            dataset.local_hiddens = batch.reshape(-1, batch.shape[-2],batch.shape[-1]).transpose(2,1)
                            
                            if args.print_dataset:
                                cond_y.extend(cond['y'])
                                cond_t.extend([x for l in cond['target'] for x in l])
                                pbar.update(batch.shape[0])
                                count += batch.shape[0]
                                continue                        
                            
                    else:
                        labels = np.array([dataset.get_label(l) for l in np.array([args.cond]*args.num_samples)])  #range(0,6)
                        labels = th.from_numpy(labels).to(device=dist_util.dev())
                        # labels = repeat(labels, 'b d -> (b c) d', c=nrow_real).contiguous()
                    labels = labels.contiguous()
                    # model_kwargs["y"] = labels
                    model_kwargs['use_cross_attention'] = args.use_cross_attention
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
                fname += f'_tp{args.time_power}' if args.time_power!=1 else ""
                num_feature = args.model_config.model.image_size
                hidden_size = args.model_config.model.in_channels


                with th.no_grad():
                    model_kwargs["y"] = labels
                    sample = sample_fn(
                        model,
                        (dataset.ae_args.batch_size_per_gpu, 1, num_feature, hidden_size),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                        progress=args.diffusion_pbar,
                        full_traj=args.output_full_traj,
                        cfg_weight=args.cfg_weight,
                        time_power=args.time_power
                    )
                    # sample.extend(sample_current)
                    # assert(len(sample) == args.batch_size)
                
                if plot_png:
                    if args.output_full_traj:
                        nrow, ncol = dividable(args.batch_size * 2)
                        nrow_real  = nrow if args.batch_size % nrow == 0 else ncol
                    else:
                        nrow, ncol = dividable(args.batch_size)
                        nrow_real  = nrow

                    # saving png
                    num_sample = len(sample) #args.num_samples
                    num_channel = hidden_size//256
                    image = dataset.local_hiddens[:num_sample]    #[6,768,16]
                    if isinstance(image[0], th.Tensor):
                        image = np.array([i.cpu().numpy() for i in image])
                    image = np.reshape(image, [num_sample, num_channel, 256, -1]).transpose(0,1,3,2).reshape([num_sample,num_channel,64,-1])
                    # image = np.expand_dims(image, 1).transpose(0,1,3,2)
                    
                    # image_norm = (image - image.mean())/(image.std())/4

                    # sample_norm = [(s - s.mean())/(s.std())/4 for s in sample]

                    if args.output_full_traj:
                        sample_image = [s.permute(0,1,3,2).view([num_sample,num_channel,256,-1]).permute(0,1,3,2) for s in sample]  #sample: 250 *[6, 1, 32, 768] => 250 * torch.Size([6, 3, 32, 256])
                        sample_image = [th.cat([s[:, :, :image_size].reshape([num_sample,num_channel,64,-1]), s[:, :, -image_size:].reshape([num_sample,num_channel,64,-1])], dim = 2) for s in sample_image]
                        image_size = sample_image[-1].shape[2]//2
                        last_sample = (utils.make_grid(
                            sample_image[-1].cpu().clamp(-1, 1)[:, :, -image_size:], nrow=nrow_real, normalize=True, value_range=(-1, 1)) * 255  #[B, C, I, I]
                            ).permute(1,2,0).to(th.uint8)  #
                        full_sample = th.stack(
                            [utils.make_grid(s.cpu().clamp(-1, 1), nrow=nrow_real, 
                            normalize=True, value_range=(-1,1)) * 255 for s in sample_image]).permute(0,2,3,1).to(th.uint8)
                        
                        if args.output_quality is None:
                            io.write_video(os.path.join(logger.get_dir(), fname + '.mp4'), full_sample[:,:,:,:3], fps=15)
                        else:
                            # output high-quality video
                            with imageio.get_writer(os.path.join(logger.get_dir(), fname + '.mp4'), 
                                fps=25, quality=args.output_quality) as writer:
                                for i in range(len(full_sample)):
                                    writer.append_data(full_sample[i,:,:,:3].numpy())
                        imageio.imsave(os.path.join(logger.get_dir(), fname + '.png'), last_sample.numpy()[:,:,:3])
                        # sample = sample[-1]
                    else:
                        utils.save_image(
                            sample.clamp(-1, 1), 
                            os.path.join(logger.get_dir(), fname + '.png'), 
                            value_range=(-1,1), normalize=True,
                            nrow=nrow_real)

                    
                    # sample from oracle
                    image_grid = (utils.make_grid(
                        th.from_numpy(image.astype(float)).clamp(-1, 1), nrow=nrow_real, normalize=True, value_range=(-1, 1)) * 255  #[B, C, I, I]
                        ).permute(1,2,0).to(th.uint8)
                    imageio.imsave(os.path.join(logger.get_dir(), 'oracle.png'), image_grid.numpy()[:,:,:3])


                # saving npz
                # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                # sample = sample.permute(0, 2, 3, 1)
                # sample = sample.contiguous()
                sample_all = th.stack(sample)
                gathered_samples = [th.zeros_like(sample_all) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample_all)  # gather not supported with NCCL
                reprs_batch = [sample.cpu().numpy() for sample in gathered_samples]
                all_reprs.extend(reprs_batch)
                # all_text.extend(gen_from_repr())
                if args.class_cond:
                    gathered_labels = [
                        th.zeros_like(labels) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(gathered_labels, labels)
                    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

                pbar.update(len(all_reprs) * dataset.ae_args.batch_size_per_gpu - count)
                count = len(all_reprs) * dataset.ae_args.batch_size_per_gpu


        if args.print_dataset:
            save_path = args.sample_dir
            with open(os.path.join(save_path, 'prefix.txt'), "w") as f_pre, open(os.path.join(save_path, 'human.txt'), "w") as f_human:
                for c, t in zip(cond_y[:args.num_samples], cond_t[:args.num_samples]):
                    c = re.sub('\n', ' ', c)
                    t = re.sub('\n', ' ', t)
                    f_pre.write(f"{c}\n")
                    f_human.write(f"{t}\n")
            exit() 


        arr = np.concatenate(all_reprs, axis=1)
        if args.store_only_last_sample:
            arr = arr[-1:]
        # arr = arr[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            if args.class_cond:
                shape_str = f"{shape_str}_{str(args.cond)}"
            if args.use_ddim:
                shape_str = f"{shape_str}_ddim"
            shape_str += f'_tp{args.time_power}' if args.time_power!=1 else ""
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

  

    else:
        logger.log("start forward process...")
        outputs = []
        count = 0
        with tqdm.tqdm(total=args.num_samples, desc="generate samples") as pbar:
            while count < args.num_samples:
                batch, cond = next(data)  
                cond['use_cross_attention'] = args.use_cross_attention
                # import pdb;pdb.set_trace()
                batch = batch.to(dist_util.dev())
                if hasattr(dataset, "local_classes"):  
                    cond = {
                        k: (v if isinstance(v, bool) else v.to(dist_util.dev()).reshape(v.shape[0], -1))
                        for k, v in cond.items()
                    }
                else: # dataset from TextDataset, cond DIT
                    if len(batch.shape) == 5 and isinstance(dataset, guided_diffusion.data.image_datasets.TextDataset):
                        batch = batch.permute(1,0,2,3,4)
                    elif isinstance(dataset, guided_diffusion.data.image_datasets.RawTextDataset):
                        batch = batch.permute(1,0,3,2)
                    # args.batch_size = min(batch_size, args.batch_size * dataset.ae_args.batch_size_per_gpu)
                    cond['y'] = [x for l in cond['y'] for x in l]
                    batch = batch.reshape(-1,batch.shape[-3], batch.shape[-2],batch.shape[-1])   #(dataset.ae_args.batch_size_per_gpu, 1, num_feature, hidden_size)
                    # batch = batch.reshape(batch.shape[-3],batch.shape[-4], batch.shape[-3],batch.shape[-2])   #(dataset.ae_args.batch_size_per_gpu, 1, num_feature, hidden_size)
                    cond['y'] = model.cond_tokenization(cond['y'])['input_ids'].to(device=dist_util.dev())
                    batch, cond['y'] = batch[:args.num_samples], cond['y'][:args.num_samples]

                # sampler = UniformSampler(diffusion)

                num_steps = args.model_config.diffusion.continuous_steps
                num_sample = dataset.ae_args.batch_size_per_gpu
            
                output = []
                alpha_sq = []
                for step in tqdm.tqdm(range(0, num_steps+1), desc="generate samples"):
                    t = th.Tensor([step]*num_sample).to(dist_util.dev())/num_steps
                    t = t**args.time_power
                    # get the model's output
                    alphas, sigmas = diffusion.t_to_alpha_sigma(t)
                    alphas, sigmas = alphas[:, None, None, None], sigmas[:, None, None, None]

                    steps = t * 1000.  
                    x_start = batch
                    noise = th.randn_like(x_start)
                    x = x_start * alphas + noise * sigmas
                    with th.no_grad():
                        # cond = {
                        #     k: (v if isinstance(v, bool) else v.reshape(v.shape[0], -1))
                        #     for k, v in cond.items()
                        # }
                        pred, _ = diffusion.get_model_output(model, x, steps, alphas, sigmas, cfg_weight=args.cfg_weight, **cond)    
                    output.append(pred)
                    alpha_sq.append(alphas.cpu().numpy().squeeze()[0]**2)
                outputs_b = th.stack(output, 0).detach().cpu()
                print(outputs_b.shape)
                outputs.append(outputs_b)
                pbar.update(batch.shape[0])
                count += batch.shape[0]


        outputs = th.cat(outputs, 1)

        
        
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in outputs.shape])
            out_path = os.path.join(logger.get_dir(), f"forward_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, outputs)
            np.savetxt(os.path.join(logger.get_dir(), f"forward_{shape_str}.alpha_sq.csv"), np.array(alpha_sq), delimiter=",")





    dist.barrier()
    logger.log("sampling complete")

if __name__ == "__main__":
    main()
