import math
from math import sqrt
import argparse
from pathlib import Path
import PIL.Image as Image

# torch

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter

# dalle classes and utils

from vit_sr_pytorch.discrete_vae import DiscreteVAE

from dataset_tools_pytorch import ImagePath, Scale

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--image_folder', type = str, required = True,
                    help='path to your folder of images for learning the discrete VAE and its codebook')

parser.add_argument('--model_path', type = str, required = False, default = None,)

parser.add_argument('--image_size', type = int, required = False, default = 256,
                    help='image size')


train_group = parser.add_argument_group('Training settings')

train_group.add_argument('--epochs', type = int, default = 20, help = 'number of epochs')

train_group.add_argument('--batch_size', type = int, default = 8, help = 'batch size')

train_group.add_argument('--learning_rate', type = float, default = 1e-3, help = 'learning rate')

train_group.add_argument('--lr_decay_rate', type = float, default = 0.98, help = 'learning rate decay')

train_group.add_argument('--starting_temp', type = float, default = 1., help = 'starting temperature')

train_group.add_argument('--temp_min', type = float, default = 0.5, help = 'minimum temperature to anneal to')

train_group.add_argument('--anneal_rate', type = float, default = 1e-6, help = 'temperature annealing rate')

train_group.add_argument('--num_images_save', type = int, default = 4, help = 'number of images to save')

train_group.add_argument('--save_folder', type = str, required = True, help = 'the location to save training logs.')

model_group = parser.add_argument_group('Model settings')

model_group.add_argument('--num_tokens', type = int, default = 8192, help = 'number of image tokens')

model_group.add_argument('--num_layers', type = int, default = 3, help = 'number of layers (should be 3 or above)')

model_group.add_argument('--num_resnet_blocks', type = int, default = 2, help = 'number of residual net blocks')

model_group.add_argument('--smooth_l1_loss', dest = 'smooth_l1_loss', action = 'store_true')

model_group.add_argument('--emb_dim', type = int, default = 512, help = 'embedding dimension')

model_group.add_argument('--hidden_dim', type = int, default = 256, help = 'hidden dimension')

model_group.add_argument('--kl_loss_weight', type = float, default = 0., help = 'KL loss weight')

model_group.add_argument('--transparent', dest = 'transparent', action = 'store_true')

args = parser.parse_args()

# constants

IMAGE_SIZE = args.image_size
IMAGE_PATH = args.image_folder

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
LR_DECAY_RATE = args.lr_decay_rate

NUM_TOKENS = args.num_tokens
NUM_LAYERS = args.num_layers
NUM_RESNET_BLOCKS = args.num_resnet_blocks
SMOOTH_L1_LOSS = args.smooth_l1_loss
EMB_DIM = args.emb_dim
HIDDEN_DIM = args.hidden_dim
KL_LOSS_WEIGHT = args.kl_loss_weight

TRANSPARENT = args.transparent
CHANNELS = 4 if TRANSPARENT else 3
IMAGE_MODE = 'RGBA' if TRANSPARENT else 'RGB'

STARTING_TEMP = args.starting_temp
TEMP_MIN = args.temp_min
ANNEAL_RATE = args.anneal_rate

NUM_IMAGES_SAVE = args.num_images_save

def noop(*args, **kwargs):
    return True

# data
transforms = [
        T.Lambda(lambda img: img.convert(IMAGE_MODE) if img.mode != IMAGE_MODE else img),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomCrop(IMAGE_SIZE),
        T.ToTensor()
    ]

ds = ImagePath(
    IMAGE_PATH,
    transform=T.Compose(transforms),
    is_valid_file=noop
)

assert len(ds) > 0, 'folder does not contain any images'
print(f'Dataset contains {len(ds):,} images')

data_sampler = None

dl = DataLoader(ds, BATCH_SIZE, shuffle = not data_sampler, sampler=data_sampler)

vae_params = dict(
    image_size = IMAGE_SIZE,
    num_layers = NUM_LAYERS,
    num_tokens = NUM_TOKENS,
    channels = CHANNELS,
    codebook_dim = EMB_DIM,
    hidden_dim   = HIDDEN_DIM,
    num_resnet_blocks = NUM_RESNET_BLOCKS
)

if args.model_path is not None:
    checkpoint = torch.load(args.model_path)
    vae_params, weights = checkpoint['hparams'], checkpoint['weights']
else:
    weights = None

vae = DiscreteVAE(
    **vae_params,
    smooth_l1_loss = SMOOTH_L1_LOSS,
    kl_div_loss_weight = KL_LOSS_WEIGHT
)
if weights is not None:
    vae.load_state_dict(weights)
vae.cuda()


# optimizer

opt = Adam(vae.parameters(), lr = LEARNING_RATE)
sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)

save_path = Path(args.save_folder)
save_path.mkdir(exist_ok = True, parents = True)

writer = SummaryWriter(log_dir=save_path)

def save_model(path):
    save_obj = {
        'hparams': vae_params,
    }

    save_obj = {
        **save_obj,
        'weights': vae.state_dict()
    }

    torch.save(save_obj, path)

# starting temperature

global_step = 0
temp = STARTING_TEMP

# add these parameters as tb hparams
model_config = dict(
    num_tokens = NUM_TOKENS,
    smooth_l1_loss = SMOOTH_L1_LOSS,
    num_resnet_blocks = NUM_RESNET_BLOCKS,
    kl_loss_weight = KL_LOSS_WEIGHT
)

writer.add_hparams(model_config, {})

print('Starting training ...')

for epoch in range(EPOCHS):
    for i, images in enumerate(dl):
        images = images.cuda()

        loss, recons = vae(
            images,
            return_loss = True,
            return_recons = True,
            temp = temp
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        logs = {}

        if i % 100 == 0:
         
            k = NUM_IMAGES_SAVE

            with torch.no_grad():
                codes = vae.get_codebook_indices(images[:k])
                hard_recons = vae.decode(codes)

            images, recons = map(lambda t: t[:k], (images, recons))
            images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (images, recons, hard_recons, codes))

            images = make_grid(images, nrow = int(sqrt(k)), normalize = True, range = (0, 1))
            recons, hard_recons = map(lambda t: make_grid(t.float(), nrow = int(sqrt(k)), normalize = True, range = (-1, 1)), (recons, hard_recons))

            writer.add_image('sample_images', images, global_step)
            writer.add_image('reconstructions', recons, global_step)
            writer.add_image('hard_reconstructions', hard_recons, global_step)
            writer.add_histogram('codebook_indices', codes, global_step)
            writer.add_scalar('temperature', temp, global_step)

            save_model(save_path.joinpath(f'vae-{epoch}-{global_step}.pt'))

            # temperature anneal

            temp = max(temp * math.exp(-ANNEAL_RATE * global_step), TEMP_MIN)

            # lr decay

            # Do not advance schedulers from `deepspeed_config`.
            sched.step()

        # Collective loss, averaged
        avg_loss = loss.mean()

   
        if i % 10 == 0:
            lr = sched.get_last_lr()[0]
            print(f'epoch: {epoch}, step: {i}, lr - {lr:6f} loss - {avg_loss.item()}')

            writer.add_scalar('loss', avg_loss, global_step)
            writer.add_scalar('lr', lr, global_step)
        
        global_step += 1

    save_model(save_path.joinpath(f'vae-{epoch}.pt'))

writer.close()
save_model('./vae-final.pt')