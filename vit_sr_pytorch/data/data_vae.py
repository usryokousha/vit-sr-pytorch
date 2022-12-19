import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split
from ml_collections import ConfigDict
from logging import Logger

from dataset_tools_pytorch import ImagePath

from ..dist import get_world_size, get_rank
from .transforms import VAETransform

def build_vae_dataloader(config: ConfigDict, logger: Logger):
    def noop(*args, **kwargs):
        return True
    train_transform = VAETransform(config.data.image_size, config.data.channels, is_train=True)
    dataset = ImagePath(config.data.image_path_list,
                        transform=train_transform, is_valid_file=noop)

    train_size = int(config.data.train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(config.seed))

    train_dataset.dataset.transform = VAETransform(config.data.image_size, scale=config.data.scale, eval=False)
    val_dataset.dataset.transform = VAETransform(config.data.image_size, scale=config.data.scale, eval=True)

    logger.info(f'{config.data.name} training set contains {len(train_dataset)} examples.')
    logger.info(f'{config.data.name} validation set contains {len(val_dataset)} examples.')

    config.unlock()
    config.data.num_train_samples = len(train_dataset)
    config.data.num_val_samples = len(val_dataset)
    config.lock()

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
   