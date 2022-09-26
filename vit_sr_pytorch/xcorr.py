from turtle import xcor
from urllib import response
import torch
import torch.nn.functional as F
import numpy as np

def _window_sum_2d(image, window_shape):
    window_sum = torch.cumsum(image, dim=-2)
    window_sum = (window_sum[..., window_shape[0]:-1, :]
                  - window_sum[..., :-window_shape[0] - 1, :])

    window_sum = window_sum = torch.cumsum(window_sum, dim=-1)
    window_sum = (window_sum[..., window_shape[1]:-1]
                  - window_sum[..., :-window_shape[1] - 1])

    return window_sum

def normxcorr(fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
    out_shape = map(lambda x: sum(x) - 1, (fixed.shape[-2:], moving.shape[-2:]))
    f_fixed = torch.fft.fft2(fixed, s=out_shape, dims=(-2, -1))
    f_moving = torch.fft.fft2(moving, s=out_shape, dims=(-2, -1))

    xcorr = torch.fft.ifft2(f_fixed * f_moving.conj(), s=out_shape, dims=(-2, -1))

    fixed_mean = fixed.mean(dim=(-2, -1), keepdim=True)
    fixed_volume = np.prod(fixed.shape[-2:])
    fixed_ssd = torch.sum((fixed - fixed_mean) ** 2, dim=(-2, -2), keepdim=True)

    moving_window_sum = _window_sum_2d(moving, fixed.shape[-2, -1])
    moving_window_sumsq = _window_sum_2d(moving ** 2, fixed.shape[-2, -1])

    numerator = xcorr - moving_window_sum * fixed_mean
    denominator = moving_window_sumsq

    torch.multiply(moving_window_sum, moving_window_sum, out=moving_window_sum)
    torch.divide(moving_window_sum, fixed_volume, out=moving_window_sum)
    denominator -= moving_window_sum
    denominator *= fixed_ssd

    torch.maximum(denominator, 0, out=denominator)
    torch.sqrt(denominator, out=denominator)

    response = torch.zeros_like(xcorr)
    mask = denominator > torch.finfo(moving.dtype).eps
    response[mask] = numerator[mask] / denominator[mask]

    slices = []
    for i in range(2):
        idx = i - 2
        d0 = (fixed.shape[idx] - 1) // 2
        d1 = d0 + moving.shape[idx]
        slices.append(slice(d0, d1))

    return response[..., slices[0], slices[1]]