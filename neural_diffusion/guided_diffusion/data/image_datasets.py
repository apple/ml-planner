import math
import random
import numpy as np
import blobfile as bf

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from gen_repr import gen_hiddens
import os
import argparse

from autoencoder.autoencoder_utils import InfiniteDistributedBucketingDataLoader, DistributedBucketingDataLoader, generate_hidden
from interpolation import load_model
from autoencoder.noiser import noise_none
import torch
import re



class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        zip_reader,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution    = resolution
        self.zip_reader    = zip_reader
        self.local_images  = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop   = random_crop
        self.random_flip   = random_flip
        self.label_shape   = None
        if classes is not None:
            if classes.ndim == 1:  # discrete label
                self.label_shape = 1 + classes.max()
            else:
                self.label_shape = classes.shape[1]
        
    def __len__(self):
        return len(self.local_images)

    def _open_file(self, fname):
        if self.zip_reader is not None:
            return self.zip_reader.open(fname, 'r')
        else:
            return bf.BlobFile(fname, "rb") 

    def get_label(self, label):
        if label.ndim == 0:   # get one-hot
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            y = onehot
        else:  # continuous label (features)
            y = label.astype(np.float32) 
        return y

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with self._open_file(path) as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1  #[256 256 3]

        out_dict = {}
        if self.local_classes is not None:
            out_dict['y'] = self.get_label(np.array(self.local_classes[idx]))  # [1, 0, 0]

        return np.transpose(arr, [2, 0, 1]), out_dict  # [3,256,256]


class ReprDataset(Dataset):
    def __init__(
        self,
        args,
        hiddens,
        cfg_dropout=None,
        cfg_weight=None,
        classes=None,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.local_hiddens  = hiddens[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.label_shape   = None
        self.cond_feature = (not classes is None) and (not any([c is None for c in classes])) and (not isinstance(classes[0], int)) and (not isinstance(classes[0], np.int64))
        if classes is not None:
            if not self.cond_feature :  # discrete label
                self.label_shape = 1 + classes.max()
                if cfg_dropout is not None or cfg_weight is not None:  # use classifier-free guidance
                    self.label_shape += 1
            else:
                self.label_shape = classes[0].shape[0]
                args.model_config.model.update({"cond_feature": True})

        
    def __len__(self):
        return len(self.local_hiddens)

    def get_label(self, label):
        if not self.cond_feature: # get one-hot
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            y = onehot
        else:  # continuous label (features)
            y = label
        return y

    def __getitem__(self, idx):
        arr = self.local_hiddens[idx]
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        arr = np.expand_dims(arr, axis = 0) 
        out_dict = {}
        if self.local_classes is not None:
            if isinstance(self.local_classes[idx], torch.Tensor):
                self.local_classes[idx] = np.array(self.local_classes[idx].cpu())
            out_dict['y'] = self.get_label(self.local_classes[idx].reshape(-1))  
            # out_dict['y'] = np.array(self.local_classes[idx])

        return np.transpose(arr, [0, 2, 1]), out_dict  # [1, 16 ,768]







class TextDataset(Dataset):
    def __init__(
        self,
        args,
        ae_args,
        model,
        cfg_dropout=None,
        cfg_weight=None,
        class_cond=False, 
        shard=0,
        num_shards=1,
        output_target=False,
        infinite_loader=False, 
    ): 
        super().__init__()
        # self.shard = shard
        # self.num_shards = num_shards
        self.device = torch.device("cuda")
        self.model = model.to(self.device)
        # self.cond_model = cond_model.to(self.device) if cond_model is not None else None
        self.shard = shard
        self.num_shards = num_shards
        self.output_target = output_target

        self.noiser = noise_none(self.model.encoder.tokenizer, mlm_probability=0.0)
        self.ae_args = ae_args

        if infinite_loader:
            self.train_dataloader = InfiniteDistributedBucketingDataLoader(ae_args.train_pt_dir, ae_args.batch_size_per_gpu, ae_args.sentence_len, rank=shard, num_replica=num_shards)
        else:
            self.train_dataloader = DistributedBucketingDataLoader(ae_args.train_pt_dir, ae_args.batch_size_per_gpu, ae_args.sentence_len, rank=shard, num_replica=num_shards)
        # just for determine the label shape
        _, conds = generate_hidden(self.model, self.train_dataloader, self.noiser, self.device, max_size=self.ae_args.batch_size_per_gpu)
        classes = conds
        self.cond_feature = (not classes is None) and (not any([c is None for c in classes])) and (not isinstance(classes[0], int)) and (not isinstance(classes[0], np.int64)) and (not isinstance(classes[0], str))
        self.label_shape = None
        self.class_cond = class_cond
        if class_cond:
            assert(not any([c == None for c in classes]))
            print("class_cond")
            if not self.cond_feature:  
                if isinstance(classes[0], str):
                    self.label_shape = args.model_config.model.cond_dim
                else:   # discrete label
                    classes = np.array(classes)
                    self.label_shape = 1 + classes.max()
                    if cfg_dropout is not None or cfg_weight is not None:  # use classifier-free guidance
                        self.label_shape += 1
            else:
                self.label_shape = classes[0].shape[0]
                args.model_config.model.update({"cond_feature": True})
            self.reset()

    def __len__(self):
        return int(1e8)

    def reset(self):
        # reset the distributed data loader
        self.train_dataloader = InfiniteDistributedBucketingDataLoader(self.ae_args.train_pt_dir, self.ae_args.batch_size_per_gpu, self.ae_args.sentence_len, rank=self.shard, num_replica=self.num_shards)

    def get_label(self, label):
        if not self.cond_feature and not isinstance(label[0], str): # get one-hot
            label = np.array(label)
            y = np.zeros((len(label), self.label_shape), dtype=np.float32)
            y[np.arange(len(label)), label] = 1
        else:  # continuous label (features)
            y = label  #.astype(np.float32) 
        return y

    def __getitem__(self, idx):
        # arr, conds = generate_hidden(self.model, self.train_dataloader, self.noiser, self.device, gpu=True)
        with torch.no_grad():
            batch = next(self.train_dataloader)
            input_ids_bert = batch[0]
            input_ids_enc = self.noiser.noise(input_ids_bert)
            input_ids_enc = input_ids_enc.to(self.device)
            arr = self.model.encoder_mean(input_ids_enc)
            # hiddens.extend([h.cpu().numpy().astype(np.float16) for h in model.encoder_mean(input_ids_enc)])
            classes = None
            if len(batch)>3:
                if self.cond_feature and self.cond_model is None:
                    input_ids_bert_cond = self.noiser.noise(batch[3]).to(self.device)
                    classes = self.model.encoder_mean(input_ids_bert_cond).reshape(input_ids_bert_cond.shape[0], -1)
                    # outputs = self.cond_model(input_ids_bert_cond)
                    # classes = outputs.last_hidden_state.reshape(input_ids_bert_cond.shape[0], -1)
                else:
                    classes = batch[3]
        
        
        arr = arr.permute([0,2,1])# # [B, 768, 16] => [B, 16 ,768]
        arr = arr.unsqueeze(1)

        out_dict = {}
        if self.class_cond and (not classes == None) and (not any([c == None for c in classes])):
            out_dict['y'] = self.get_label(classes)  

        if self.output_target:
            target = self.model.decode(batch[1], tokenizer='dec')
            target = [re.sub('\n', ' ', s) for s in target]
            target = [s.rstrip("!") for s in target]
            out_dict['target'] = target 

        return arr, out_dict  


class RawTextDataset(TextDataset):
    def __init__(
        self,
        **kwargs
    ): 
        super().__init__(**kwargs)
        with open(kwargs['args'].input_text, 'r') as f:
            self.cond_data = f.readlines()
        
        

    def get_label(self, label):
        return label
    
    def __getitem__(self, idx):
        with torch.no_grad():
            batch = next(self.train_dataloader)
            input_ids_bert = batch[0]
            input_ids_enc = self.noiser.noise(input_ids_bert)
            input_ids_enc = input_ids_enc.to(self.device)
            arr = self.model.encoder_mean(input_ids_enc) #bsz x latent x feature num
            batch_size = arr.shape[0]
            num_batch = len(self.cond_data)//batch_size
            classes = self.cond_data[idx*batch_size:(idx+1)*batch_size]  #[idx::num_batch]  # TODO check transpose issue
            out_dict = {}
            out_dict['y'] = self.get_label(classes)  

        return arr, out_dict  # [B, 768, 16]



def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
