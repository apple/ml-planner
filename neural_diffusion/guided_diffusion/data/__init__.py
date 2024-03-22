import blobfile as bf
import torch.distributed as dist
import torch
import os, zipfile, json
import numpy as np

from .. import logger
from mpi4py import MPI
from .image_datasets import ImageDataset, ReprDataset, TextDataset, RawTextDataset
from torch.utils.data import DataLoader
from interpolation import parse_args as parse_ae_args
from interpolation import load_model
import argparse
import torch.nn as nn



def get_dataset(
    *,
    data_dir,
    image_size,
    args = None,
    class_cond=False,
    random_crop=False,
    random_flip=True,
    cfg_dropout=None,
    cfg_weight=None,
    output_target=False, 
    **kwargs,
    ):
    """
    :param data_dir: a dataset directory or .zip file.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    classes, zipreader = None, None
    if os.path.isdir(data_dir) or os.path.splitext(data_dir)[1].lower() == '.pt':
        logger.info(f"read dataset: {data_dir}")
        if not os.path.isdir(data_dir):
            data_dir = os.path.dirname(data_dir)
        kwargs['train_pt_dir']= data_dir
        kwargs['load_dec']=True
        kwargs['load_enc']=True
        parser = parse_ae_args()
        ae_args, _ = parser.parse_known_args()
        new_args = argparse.Namespace(**kwargs)
        ae_args.__dict__.update(new_args.__dict__)


        # Download the model
        if torch.cuda.device_count() > 1:
            # Initialize the model on the first GPU
            if (not dist.is_initialized()) or (dist.get_rank() == 0):
                ae_args, model, _ = load_model(ae_args, suffix="load_ae.txt")
                # model = model.to(torch.device('cuda:0'))
            else:
                model = None
            dist.barrier()
            ae_args, model, _ = load_model(ae_args, suffix="load_ae.txt")
            # Replicate the model on all GPUs
            # model = nn.DataParallel(model)
            # model = model.module
            # print(dist.get_rank(), model)
        else:
            # Initialize the model on GPU 0
            ae_args, model, _ = load_model(ae_args, suffix="load_ae.txt")
            model = model.to(torch.device('cuda:0'))





        # modify the batch size of the data loader
        args.batch_size = max(args.batch_size//ae_args.batch_size_per_gpu, 1)
        if hasattr(args, 'input_text') and os.path.splitext(args.input_text)[1].lower() == '.txt':
            DataClass = RawTextDataset
        else:
            DataClass = TextDataset
    
        return DataClass(args=args,
                    ae_args=ae_args,
                    model=model,   
                    class_cond=class_cond,
                    cfg_dropout=cfg_dropout,
                    cfg_weight=cfg_weight,
                    shard=MPI.COMM_WORLD.Get_rank(),
                    num_shards=MPI.COMM_WORLD.Get_size(),
                    output_target=output_target
                )
        # all_files = _list_image_files_recursively(data_dir)
        # if class_cond:
        #     # Assume classes are the first part of the filename, before an underscore.
        #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
        #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        #     classes = np.array([sorted_classes[x] for x in class_names])

    elif os.path.splitext(data_dir)[1].lower() == '.zip':
        zipreader = zipfile.ZipFile(data_dir)
        all_files = zipreader.namelist()
        if class_cond:
            if 'dataset.json' in all_files:
                all_files, classes = zip(*json.load(zipreader.open('dataset.json', 'r'))["labels"])
                classes = np.array(classes)
            else:
                raise FileNotFoundError("dataset.json is not found")
        else:
            all_files = [f for f in all_files if os.path.splitext(f)[-1] in ['.jpg', '.png', '.jpeg']]
        logger.info(f"read dataset: {data_dir}, {len(all_files)} images")
        return ImageDataset(
            image_size,
            zipreader,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=random_crop,
            random_flip=random_flip,
        )

    elif os.path.splitext(data_dir)[1].lower() == '.npy' or os.path.splitext(data_dir)[1].lower() == '.npz':
        # all_files = torch.load(data_dir)
        # all_files = [f.cpu().numpy() for batch in all_files for f in batch]
        if os.path.splitext(data_dir)[1].lower() == '.npy':
            with open(data_dir, 'rb') as f:
                all_files = np.load(f)  ###  TODO: make sure all GPUs 
            logger.info(f"read dataset: {data_dir}, {len(all_files)} hidden matrices")
        else:
            with open(data_dir, 'rb') as f:
                all_files = np.load(f, allow_pickle=True)  ###  TODO: make sure all GPUs 
                all_classes, all_files = all_files['conds'], all_files['hiddens']
                if class_cond:
                    if isinstance(all_classes[0], torch.Tensor) or len(all_classes[0].shape)>1: # cont feature
                        classes = all_classes
                    else:
                        classes = np.array([c if c else 0 for c in all_classes], dtype='int') # None => 0 
            logger.info(f"read dataset: {data_dir}, {len(all_files)} hidden matrices")
        return ReprDataset(
            args=args,
            hiddens=all_files,
            classes=classes,
            cfg_dropout=cfg_dropout,
            cfg_weight=cfg_weight,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )
    else:
        raise ValueError("dataset format does not support.")
    


def load_data(dataset, batch_size, deterministic=False, num_workers=0):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
   
    """
    shuffle = not deterministic

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True
    )
    while True:
         yield from loader





def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results