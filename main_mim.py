# --------------------------------------------------------
# SR MIM training script
# Adapted from SimMIM
# --------------------------------------------------------

import os
import time
import math
import sys
import argparse
import datetime
import numpy as np
from yacs.config import CfgNode

import torch
import torch.backends.cudnn as cudnn
from torch.cuda import amp
from timm.utils import AverageMeter

from SimMIM.lr_scheduler import build_scheduler
from SimMIM.optimizer import build_optimizer
from SimMIM.logger import create_logger
from torchmetrics.functional import peak_signal_noise_ratio
from utils import (
                load_checkpoint, 
                save_checkpoint, 
                get_grad_norm, 
                auto_resume_helper, 
                reduce_tensor, 
                load_vae, 
                pixel_accuracy)


from vit_sr_pytorch.discrete_vae import DiscreteVAE
from vit_sr_pytorch.mim import build_mim, MIM
from vit_sr_pytorch.data.data_pretrain import build_loader_mim
from config import get_config
from utils import load_vae, pixel_accuracy
from vit_sr_pytorch.dist import get_world_size, get_rank

# add tensorboard logging
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from timm.scheduler.scheduler import Scheduler

def parse_option():
    parser = argparse.ArgumentParser('SimMIM pre-training script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--label_path', type=str, help='path to label file')
    parser.add_argument('--vae', type=str, help='path to vae checkpoint')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation_steps', type=int, default=0, help="gradient accumulation steps")
    parser.add_argument('--use_checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--use_amp', action='store_true', help="whether to use mixed precision training")
                  
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/pretrain/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    writer = SummaryWriter(config.OUTPUT)
    data_loader_train, data_loader_val = build_loader_mim(config, logger)

    logger.info(f"Loading trained VAE")
    vae = load_vae(config, DiscreteVAE, logger)
    vae.cuda()
    logger.info(str(vae))

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_mim(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    if config.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
        
    scaler = amp.GradScaler(enabled=config.TRAIN.USE_AMP)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters:,}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, scaler, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if config.DISTRIBUTED:
            data_loader_train.sampler.set_epoch(epoch)
        global_step = epoch * len(data_loader_train)

        train_one_epoch(config, model, vae, data_loader_train, optimizer, scaler, epoch, lr_scheduler, writer)
        if get_rank() == 0 and (epoch % config.VAL_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            validate(config, data_loader_val, model, vae, writer, global_step)
            
        if get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, scaler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(
    config: CfgNode, 
    model: MIM, 
    vae: DiscreteVAE, 
    data_loader: DataLoader, 
    optimizer: Optimizer, 
    scaler: amp.GradScaler, 
    epoch: int, 
    lr_scheduler: Scheduler, 
    writer: SummaryWriter):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (img_small, img_large, mask) in enumerate(data_loader):
        img_small = img_small.cuda(non_blocking=True)
        img_large = img_large.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        with amp.autocast(enabled=config.TRAIN.USE_AMP):
            output = model(img_small, mask)
            target = vae.get_codebook_indices(img_large)[mask.flatten(1).bool()]
            loss = criterion(output, target)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), img_small.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        
        # update tensorboard
        writer.add_scalar('train/loss', loss_meter.val, epoch * num_steps + idx)
        writer.add_scalar('train/grad_norm', norm_meter.val, epoch * num_steps + idx)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * num_steps + idx)
        writer.add_scalar('train/epoch', epoch, epoch * num_steps + idx)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    
@torch.no_grad()
def validate(
    config: CfgNode, 
    data_loader: DataLoader, 
    model: MIM, 
    vae: DiscreteVAE, 
    writer: SummaryWriter, 
    global_step: int):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    def image_grid(images):
        images = images.cpu()
        images = make_grid(images, nrow=int(math.sqrt(images.shape[0])), normalize=True, value_range=(0, 1))
        return images

    def calc_psnr(preds, targets):
        rescaled = list(map(lambda x: torch.clip(x * 255, 0, 255), (preds, targets)))
        return peak_signal_noise_ratio(*rescaled, data_range=255).item()

    end = time.time()
    psnr = 0
    for idx, (images_small, images_large) in enumerate(data_loader):
        images_small = images_small.cuda(non_blocking=True)
        images_large = images_large.cuda(non_blocking=True)
        target = vae.get_codebook_indices(images_large)

        # compute output
        with torch.no_grad():
            codes, logits = model.get_codebook_indices(images_small, return_logits=True)
            if idx == 0:
                recon = vae.decode(codes, normalized=False)
                recon_target = vae.decode(target, normalized=False)
                psnr = calc_psnr(recon, images_large)
                images, recon = images_small.cpu(), recon.cpu()

                images, recon, recon_target = map(lambda x: image_grid(x), (images, recon, recon_target))
                writer.add_image('val/images', images, global_step)
                writer.add_image('val/recon', recon, global_step)
                writer.add_image('val/recon_target', recon_target, global_step)

            # measure accuracy and record loss
            loss = criterion(logits, target)

        acc = pixel_accuracy(logits, target)

        acc = reduce_tensor(acc)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc_meter.update(acc.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc_meter.val:.3f} ({acc_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc_meter.avg:.3f} PSNR(Avg) {psnr:.3f}')
    writer.add_scalar('val/loss', loss_meter.avg, global_step)
    writer.add_scalar('val/acc', acc_meter.avg, global_step)
    writer.add_scalar('val/psnr', psnr, global_step)



if __name__ == '__main__':
    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    
    # deal with Windows based distributed training
    if config.DISTRIBUTED:
        if sys.platform() == 'win32':
            torch.distributed.init_process_group(backend='gloo', init_method='tcp://localhost:23456', world_size=world_size, rank=rank)
        else:
            torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()

    seed = config.SEED + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=get_rank(), name=f"{config.MODEL.NAME}")

    if get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
