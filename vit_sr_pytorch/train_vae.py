from absl import app

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_collections import config_dict, config_flags

from torchvision.datasets import ImageFolder
from torchvision.transforms import (RandomVerticalFlip, 
                                    RandomHorizontalFlip,
                                    RandomCrop,
                                    CenterCrop, 
                                    Compose, 
                                    ToTensor, 
                                    Normalize)
from torch.utils.data import DataLoader
from vit_sr_pytorch.vqgan_vae import VQGanVAE
from vit_sr_pytorch.optimizer import get_optimizer, separate_weight_decayable_params
from ema_pytorch import EMA
from accelerate import Accelerator

config = config_dict.ConfigDict()

config.training = config_dict.ConfigDict()
config.training.num_epochs = 20
config.training.learning_rate = 1e-3
config.training.weight_decay = 1e-5
config.training.num_images_save = 8
config.training.num_workers = 8
config.training.save_results_every = 100
config.training.save_model_every = 1000
config.training.output_dir = config_dict.placeholder(str)
config.training.log_dir = config_dict.placeholder(str)
config.training.ema_beta = 0.995
config.training.ema_update_every = 10
config.training.ema_update_after_step = 500
config.training.mixed_precision = 'fp16'
config.training.with_tracking = True

config.data = config_dict.ConfigDict()
config.data.train_dir = config_dict.placeholder(str)
config.data.val_dir = config_dict.placeholder(str)
config.data.image_size = 128
config.data.num_channels = 3
config.data.batch_size = 8
config.data.random_seed = 42
config.data.mean = [0.5, 0.5, 0.5]
config.data.std = [0.5, 0.5, 0.5]

config.model = config_dict.ConfigDict()
config.model.layers = 4
config.model.vq_codebook_size = 512
config.model.vq_codebook_dim = 64
config.model.vq_commitment_weight = 1.0
config.model.vq_decay = 0.8
config.model.vq_kmeans_init = True
config.model.vq_use_cosine_sim = True

_CFG = config_flags.DEFINE_config_dict('vae_config', config)

def train_vae(cfg: config_dict.ConfigDict):
    if cfg.training.with_tracking:
        accelerator = Accelerator(
            cpu=False,
            mixed_precision=cfg.training.mixed_precision,
            log_with="tensorboard",
            log_dir=cfg.training.logging_dir
        )
    else:
        accelerator = Accelerator(
            cpu=False,
            mixed_precision=cfg.training.mixed_precision,
        )

    if cfg.training.with_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config)

    torch.manual_seed(cfg.data.random_seed)
    torch.cuda.manual_seed(cfg.data.random_seed)

    train_transform = Compose([
        RandomCrop(cfg.data.image_size),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=config.data.mean, std=config.data.std)
    ])

    val_transform = Compose([
        CenterCrop(cfg.data.image_size),
        ToTensor(),
        Normalize(mean=config.data.mean, std=config.data.std)
        ])

    train_dataset = ImageFolder(root=cfg.data.data_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.training.num_workers)

    val_dataset = ImageFolder(root=cfg.data.data_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.training.num_workers)

    vae = VQGanVAE(
        layers=cfg.model.layers,
        vq_codebook_size=cfg.model.vq_codebook_size,
        vq_codebook_dim=cfg.model.vq_codebook_dim,
        vq_commitment_weight=cfg.model.vq_commitment_weight,
        vq_decay=cfg.model.vq_decay,
        vq_kmeans_init=cfg.model.vq_kmeans_init,
        vq_use_cosine_sim=cfg.model.vq_use_cosine_sim
    )
    vae = vae.to(accelerator.device)
    
    ema_vae = EMA(vae, beta=cfg.training.ema_beta)

    optimizer = get_optimizer(vae.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.num_epochs, eta_min=0.0)

    vae, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_loader, val_loader, lr_scheduler)

    

def main(_):
    cfg = _CFG.value
    train_vae(cfg)

if __name__ == "__main__":
    app.run()