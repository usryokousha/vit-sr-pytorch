import PIL.Image as Image
import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torchvision.transforms.functional_tensor as FT


from typing import Callable, Tuple, Union, List, Optional, Union, Dict

AUGMENTATIONS = [
    "TranslateX",
    "TranslateY",
    "Posterize",
    "Solarize",
    "AutoContrast",
    "Equalize",
    "Brightness",
    "Contrast",
    "Sharpness",
]

def exists(x):
    return x is not None

def to_tuple(x: Union[int, Tuple[int, ...]], repeats=1) -> Tuple[int, ...]:
    if isinstance(x, (int, float)):
        return (x, ) * repeats
    return x

def get_min_size(imgs, scales):
    max_scale = max(scales)
    h = min([int(TF.get_image_size(i)[1] * max_scale / s) for i, s in zip(imgs, scales)])
    w = min([int(TF.get_image_size(i)[0] * max_scale / s) for i, s in zip(imgs, scales)])
    return h, w

class RandomCropRelative(T.RandomCrop):
    def __init__(self, size, scale, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.scale = scale

    @staticmethod
    def get_params(
        img_small: Union[torch.Tensor, Image.Image],
        img_large: Union[torch.Tensor, Image.Image],
        scale: int, 
        output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        _, h, w = TF.get_dimensions(img_small)
        _, lh, lw = TF.get_dimensions(img_large)

        th, tw = output_size

        vh, vw = [min(s, int(l / scale)) for s, l in zip((h, w), (lh, lw))]

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        hdiff = max(vh - th, 0)
        wdiff = max(vw - tw, 0)

        ci = torch.randint(-hdiff // 2, hdiff // 2, size=(1,)).item()
        cj = torch.randint(-wdiff // 2, hdiff // 2, size=(1,)).item()
        
        i = (h - th) // 2 + ci
        j = (w - tw) // 2 + cj

        ci_ = ci * scale
        cj_ = cj * scale

        i_ = (lh - th * scale) // 2 + ci_
        j_ = (lw - tw * scale) // 2 + cj_

        return (i, j, th, tw), (i_, j_, th * scale, tw * scale)


    def forward(self, img_small, img_large):
        """
        Args:
            img_small (PIL Image, Tensor): Small image to be cropped.
            img_large (PIL Image, Tensor): Large image to be cropped.
        Returns:
            (PIL Image, Tensor)
        """ 
        if self.padding is not None:
            img_large = TF.pad(img_large, self.padding, self.fill, self.padding_mode)

        _, height, width = TF.get_dimensions(img_large)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img_large = TF.pad(img_large, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img_large = TF.pad(img_large, padding, self.fill, self.padding_mode)

        params_small, params_large = self.get_params(img_small, img_large, self.scale, self.size)

        return TF.crop(img_small, *params_small), TF.crop(img_large, *params_large)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, scale={self.scale}, padding={self.padding})"
        

class RandomFlipRelative(nn.Module):
    def __init__(self, p=0.5, horizontal=True, vertical=True):
        super().__init__()
        self.p = p
        self.horizontal = horizontal
        self.vertical = vertical

    def forward(self, img_small, img_large):
        if self.p < torch.rand(1).item() and self.horizontal:
            img_small, img_large = tuple(TF.hflip(img) for img in (img_small, img_large))

        if self.p < torch.rand(1).item() and self.vertical:
            img_small, img_large = tuple(TF.vflip(img) for img in (img_small, img_large))

        return img_small, img_large

    def __str__(self) -> str:
        return self.__class__.__name__ + '(p={}, horizontal={}, vertical={})'.format(self.p, self.horizontal, self.vertical)


def gaussian_blur(img: torch.Tensor, kernel: torch.Tensor, scale_factor: int) -> torch.Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    FT._assert_image_tensor(img)

    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])
    kernel = FT.interpolate(kernel, scale_factor=scale_factor, mode='bicubic')
    kernel /= kernel.sum(dim=(-2, -1), keepdim=True)
    kernel_size = kernel.shape[-2:]
    img, need_cast, need_squeeze, out_dtype = FT._cast_squeeze_in(
        img,
        [
            kernel.dtype,
        ],
    )

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = FT.torch_pad(img, padding, mode="reflect")
    img = FT.conv2d(img, kernel, groups=img.shape[-3])

    img = FT._cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img

def _check_relative_scales(x, kernel_size):
    """Ensure that kernel_size // scale is not less than one."""
    if isinstance(x, (int, float)):
        x = (x, )
    for i in x:
        assert i > 0, 'relative_scale must be greater than zero'
        assert kernel_size * i > 0, 'relative kernel must be greater than zero'
    return x

class RandomBlurRelative(nn.Module):
    def __init__(self, p=0.25, kernel_size=3, scale = 4, sigma = (0.1, 2)):
        super().__init__()
        self.scales = _check_relative_scales((1, scale), kernel_size)   
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min, sigma_max):
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img_small, img_large):
        if self.p < torch.rand(1).item():
            sigma = to_tuple(self.get_params(*self.sigma), 2)
            dtype = img_small.dtype if torch.is_floating_point(img_small) else torch.float32
            kernel = FT._get_gaussian_kernel2d(to_tuple(self.kernel_size, 2), sigma, dtype=dtype, device=img_small.device)
            img_small, img_large = tuple(gaussian_blur(img, kernel, scale) for img, scale in zip((img_small, img_large), self.scales))
        return img_small, img_large

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, kernel_size={self.kernel_size}, scale={self.scales}, sigma={self.sigma})"
        return s

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *x):
        return x

    def __str__(self):
        return self.__class__.__name__

class JoinTransform(nn.Module):
    def __init__(self, transforms: List[Callable]):
        super().__init__()
        if all([isinstance(t, nn.Module) for t in transforms]):
            self.transforms = nn.ModuleList(transforms)
        else:
            self.transforms = transforms

    def forward(self, *imgs):
        imgs_out = []
        for t, img in zip(self.transforms, imgs):
            imgs_out.append(t(img))
        return tuple(imgs_out)

    def __str__(self):
        return self.__class__.__name__ + f'({self.transforms})'

class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            TF._log_api_usage_once(self)
        self.transforms = transforms

    def __call__(self, *img):
        for t in self.transforms:
            img = t(*img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class MaskedImageTransform:
    def __init__(self, target_size, scale=4, p=0.5, horizontal=True, vertical=True, blur=True, eval=False):
        input_size = target_size // scale
        if eval:
            self.transform = Compose([
                JoinTransform([T.CenterCrop(input_size), T.CenterCrop(target_size)]),
                JoinTransform([T.ToTensor(), T.ToTensor()])
            ])
        else:
            self.transform = Compose([
                RandomCropRelative(target_size // scale, scale),
                RandomFlipRelative(p, horizontal, vertical),
                JoinTransform([T.ToTensor(), T.ToTensor()]),
                RandomBlurRelative(scale=scale, kernel_size=scale, sigma=(0.1, scale)) if blur else Identity(),
            ])

    def __call__(self, img_small, img_large):
        return self.transform(img_small, img_large)

    def __str__(self) -> str:
        format_string = self.__class__.__name__ + '(\n'
        format_string += '    ' + str(self.transform) + '\n'
        format_string += ')'

class AugMixLite(T.AugMix):
    def __init__(
        self,
        severity: int = 3,
        mixture_width: int = 3,
        chain_depth: int = -1,
        alpha: float = 1.0,
        all_ops: bool = True,
        interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__(severity, mixture_width, chain_depth, alpha, all_ops, interpolation, fill)

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        augs = super()._augmentation_space(num_bins, image_size)
        for k in augs.keys():
            if k not in AUGMENTATIONS:
                augs.pop(k)
        return augs

class Replicate(nn.Module):
    def __init__(self, num_copies):
        super().__init__()
        self.num_copies = num_copies

    def forward(self, x):
        output = []
        return output + [x.clone() for _ in range(self.num_copies)]

    def __str__(self):
        return self.__class__.__name__ + f'({self.num_copies})'

class FineTuneTransform:
    def __init__(self, image_size, eval=False, mean=(0.,) * 3, std=(1.,) * 3, no_jsd=True, **kwargs):
        if eval:
            self.preprocess = T.Compose([
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        else:
            augmix = AugMixLite(**kwargs)
            augmix_transform = augmix if not no_jsd else nn.Sequential(
                Replicate(2),
                JoinTransform([Identity(), augmix, augmix])
            )
            self.preprocess = T.Compose([
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                augmix_transform])

    def __call__(self, img):
        return self.preprocess(img)
    
class VAETransform:
    def __init__(self, image_size, is_train = True):
        self.image_size = image_size
        if is_train:
            self.transform = T.Compose([
                            T.Lambda(lambda x: x.convert('RGB')),
                            T.RandomCrop(image_size),
                            T.RandomHorizontalFlip(),
                            T.RandomVerticalFlip(),
                            T.ToTensor()
                        ])
        else:
            self.transform = T.Compose([
                            T.Lambda(lambda x: x.convert('RGB')),
                            T.CenterCrop(image_size),
                            T.ToTensor()
                        ])
            
    def __call__(self, x):
        return self.transform(x)

        
        
