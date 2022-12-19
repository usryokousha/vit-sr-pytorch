# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, random_split
from torch.utils.data._utils.collate import default_collate
from .transforms import JoinTransform, RandomCropRelative, RandomBlurRelative, RandomFlipRelative, Compose, Identity
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from ..dist import get_world_size, get_rank

from typing import List, Tuple, Callable

def bash_to_win32_path(path: str) -> str:
    """Changes path drive designation from /a/... to a:/..."""
    if sys.platform == "win32":
        path = path[1] + ":" + path[2:]
    return path

def load_file_list(file_path):
    with open(file_path, 'r') as f:
        file_list = f.read().splitlines()
        file_list = sorted(map(bash_to_win32_path, file_list))
    return file_list

def get_dataset_list(path_0: str, path_1: str) -> List[Tuple[str, str]]:
    """Check consistency of each file returned by os.walk()
        for low and high resolution images and returns updated lists.

    Args:
        path_0 (str): directories / list of files
        path_1 (str): directories / list of files

    Returns:
        List[tuple]: list of matching paths for each example
    """

    files_0 = load_file_list(path_0)
    files_1 = load_file_list(path_1)

    def get_filename(path):
        return os.path.splitext(os.path.basename(path))[0].lower()

    df_0 = pd.DataFrame(files_0, columns=['files_0'])
    df_1 = pd.DataFrame(files_1, columns=['files_1'])

    df_0['filename'] = df_0['files_0'].apply(get_filename)
    df_1['filename'] = df_1['files_1'].apply(get_filename)

    merged = pd.merge(df_0, df_1, on='filename')

    return list(zip(merged['files_0'].tolist(),
                    merged['files_1'].tolist()))

def split_dataset_list(dataset_list, val_split, seed=0):
    """Split dataset list into train and validation sets.

    Args:
        dataset_list (list): list of dataset
        split (float): split ratio

    Returns:
        tuple: train and validation dataset
    """
    val_size = int(val_split * len(dataset_list))
    train_size = len(dataset_list) - val_size
    np.random.RandomState(seed).shuffle(dataset_list)

    return dataset_list[:train_size], dataset_list[train_size:]

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths: list, loader: Callable = default_loader, transform=None):
        self.transform = transform
        self.loader = loader
        self.file_paths = file_paths

    def __getitem__(self, index):
        data_path, label_path = self.file_paths[index]
        data = self.loader(data_path)
        label = self.loader(label_path)
        if self.transform is not None:
            return self.transform(data, label)
        else:
            return data, label

    def __len__(self):
        return len(self.file_paths)

class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask
    def __str__(self):
        format_string = self.__class__.__name__ + '(\n'
        format_string += '    input_size={},\n'.format(self.input_size)
        format_string += '    mask_patch_size={},\n'.format(self.mask_patch_size)
        format_string += '    model_patch_size={},\n'.format(self.model_patch_size)
        format_string += '    mask_ratio={},\n'.format(self.mask_ratio)
        format_string += ')'
        return format_string


class SimMIMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])
 
        if config.MODEL.TYPE == 'swin':
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask

# TODO: Fix random blur size issue (occurs during collation)
class MIMTransform:
    def __init__(self, config, eval=False):
        input_size = config.DATA.IMG_SIZE // config.DATA.REL_SCALE
        target_scale = config.DATA.REL_SCALE
        self.eval = eval
        if eval:
            self.transform = Compose([
                JoinTransform([T.CenterCrop(input_size), T.CenterCrop(config.DATA.IMG_SIZE)]),
                JoinTransform([T.ToTensor(), T.ToTensor()])
            ])
        else:
            self.transform = Compose([
                RandomCropRelative(input_size, config.DATA.REL_SCALE),
                RandomFlipRelative(),
                JoinTransform([T.ToTensor(), T.ToTensor()]),
                RandomBlurRelative(scale=target_scale, kernel_size=target_scale, sigma=(0.1, target_scale)) if config.DATA.USE_BLUR else Identity(),
            ])

        self.mask_generator = MaskGenerator(
            input_size=input_size,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=config.DATA.MODEL_PATCH_SIZE,
            mask_ratio=config.DATA.MASK_RATIO,
        )

    def __call__(self, img_small, img_large):
        img_small, img_large = self.transform(img_small, img_large)

        if not self.eval:
            mask = self.mask_generator()
            return img_small, img_large, mask
        else:
            return img_small, img_large

    def __str__(self) -> str:
        format_string = self.__class__.__name__ + '(\n'
        format_string += '    ' + str(self.transform) + '\n'
        if not self.eval:
            format_string += '    ' + str(self.mask_generator) + '\n'
        format_string += ')'
        return format_string


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(config, logger):
    transform = SimMIMTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset):,}')
    
    sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    return dataloader



def build_loader_mim(config, logger):
    dataset_list = get_dataset_list(config.DATA.DATA_PATH, config.DATA.LABEL_PATH)
    train_dataset_list, val_dataset_list = split_dataset_list(dataset_list, config.DATA.VAL_RATIO, config.SEED)

    train_transform = MIMTransform(config, eval=False)
    val_transform = MIMTransform(config, eval=True)

    train_dataset = ListDataset(train_dataset_list, transform=train_transform)
    val_dataset = ListDataset(val_dataset_list, transform=val_transform)

    logger.info(f'Train data transform:\n{train_dataset.transform}')
    logger.info(f'Val data transform:\n{val_dataset.transform}')
    logger.info(f'Build dataset: train images = {len(train_dataset):,}, val images = {len(val_dataset):,}')

    if config.DISTRIBUTED:
        train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset, 
        config.DATA.BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY, 
        drop_last=True, 
        collate_fn=collate_fn)

    val_loader = DataLoader(
        val_dataset,
        config.DATA.BATCH_SIZE,
        sampler=val_sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY, 
        drop_last=False,
        collate_fn=collate_fn)

    return train_loader, val_loader