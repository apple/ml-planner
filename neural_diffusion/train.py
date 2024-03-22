"""
Train a basic diffusion model on images.
"""

import hydra
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from guided_diffusion import logger
from guided_diffusion.data import get_dataset, load_data
from guided_diffusion.diffusion.resample import create_named_schedule_sampler
from guided_diffusion import (  
    create_model, create_gaussian_diffusion,
)
from guided_diffusion.trainer import TrainLoop, save_user_config
from guided_diffusion.utils import dist_util
import numpy as np
import os
import torch.multiprocessing as mp


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(args: DictConfig):
    # mp.set_start_method("spawn", force=True)
    dist_util.setup_dist()
    rank, world_size = dist_util.get_rank(), dist_util.get_world_size()
    dist_util.setup_seed(args.seed + rank)  # setup seeds
    
    logger.configure(dir=args.log_dir, format_strs=args.log_format)
    logger.log("configs: {}".format(args))
    
    assert args.data_dir is not None
    logger.log("creating data loader...")
    batch_size = args.batch_size
    dataset = get_dataset(
        args=args,
        data_dir=args.data_dir,
        image_size=args.model_config.model.image_size,
        class_cond=args.class_cond,
        random_flip=args.random_flip,
        cfg_dropout=args.trainer.cfg_dropout,
        task_id=args.task_id,
        num_feature=args.num_feature, 
        sentence_len=args.sentence_len,
        load_ckpt=args.load_ckpt,
        ae_step=args.ae_step,
    )

    if os.path.isdir(args.data_dir):
        data = load_data(dataset, args.batch_size, num_workers=0, deterministic=True)
    else:
        data = load_data(dataset, args.batch_size)

    args.batch_size = batch_size 
    logger.log("creating model and diffusion...")
    model = create_model(**args.model_config.model, num_classes=dataset.label_shape)
    diffusion = create_gaussian_diffusion(**args.model_config.diffusion)
    model.get_summary()
    model.to(dist_util.dev())

    if rank == 0:
        save_user_config(args.model_config, 'model_config')
        logger.log("saved user configurations")

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f"training on {dist.get_world_size()} GPUs, total bsz={dist.get_world_size() * args.batch_size}")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        resume_checkpoint=args.resume_checkpoint,
        schedule_sampler=schedule_sampler,
        **args.trainer
    ).run_loop()

if __name__ == "__main__":
    main()
