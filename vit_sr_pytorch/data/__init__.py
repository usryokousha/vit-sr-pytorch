from .data_finetune import build_finetune_loader
from .data_pretrain import build_loader_mim
from .data_vae import build_vae_dataloader

from ml_collections import ConfigDict
from logging import Logger

def build_loader(config: ConfigDict, logger: Logger):
    if config.model.type == 'vae':
        return build_vae_dataloader(config, logger)
    elif config.model.type == 'pretrain':
        return build_loader_mim(config, logger)
    elif config.model.type == 'finetune':
        return build_finetune_loader(config, logger)
    else:
        raise NotImplementedError