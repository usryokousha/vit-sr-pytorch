import os
import torch
from torch.utils.data import random_split

from ml_collections import ConfigDict

from dataset_tools_pytorch import datasets as ds
from .transforms import FineTuneTransform
from logging import Logger

from ..dist import get_world_size, get_rank

from typing import Union


def build_finetune_loader(config: ConfigDict, logger: Logger):
    """Build dataloader for fine-tuning"""
    dataset = FineTuneDataset(list_path=config.data.data_path, metadata_path=config.data.data_label_path)

    train_size = int(config.data.train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(config.seed))

    train_dataset.dataset.transform = FineTuneTransform(config.data.image_size, eval=False)
    val_dataset.dataset.transform = FineTuneTransform(config.data.image_size, eval=True)

    config.unlock()
    config.data.num_classes = len(dataset.class_names)
    config.lock()

    logger.info(f'{config.data.name} training set contains {len(train_dataset)} examples.')
    logger.info(f'{config.data.name} validation set contains {len(val_dataset)} examples.')

    print(f'Train dataset transformations: \n{train_dataset.dataset.transform}')

    if config.training.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False)

    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=val_sampler is None,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        sampler=val_sampler,
        drop_last=False)

    return train_loader, val_loader
    

class FineTuneDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""
    def __init__(self, image_list_path: str, metadata_path: Union[str, tuple], transform=None):
        image_paths = ds.read_file_list(image_list_path)
        images, labels = ds.check_image_label(image_paths, metadata_path)
        self.class_names = sorted(list(set(labels)))
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_names)}

        self.loader = ds.default_loader
        self.samples = images
        self.targets = [self.class_to_idx[label] for label in labels]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.loader(self.samples[index])
        target = self.targets[index]
        return self.transform(sample), target



