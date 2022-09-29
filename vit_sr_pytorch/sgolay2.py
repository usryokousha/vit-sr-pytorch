import collections
import stat
import typing as t

import numpy as np
import torch
import torch.nn.functional as F

from fft_conv_pytorch import fft_conv


class SGolay2d(torch.nn.Module):
    def __init__(self, window_size, order) -> None:
        super().__init__()
        self.window_size = window_size
        self.register_buffer('kernel', self.build_kernel(window_size, order))

    @staticmethod
    def build_kernel(window_size, order):
        # exponents of the polynomial. 
        # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ... 
        # this line gives a list of two item tuple. Each tuple contains 
        # the exponents of the k-th term. First element of tuple is for x
        # second element for y.
        # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
        exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

        # coordinates of points
        half_size = window_size // 2
        idx = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
        dx = idx.repeat_interleave(window_size, dim=0)
        dy = idx.repeat(window_size)

        # build matrix of system of equation
        A = torch.empty((window_size ** 2 , len(exps)), dtype=torch.float32)
        for i, exp in enumerate( exps ):
            A[:,i] = (dx ** exp[0]) * (dy ** exp[1])

        return torch.linalg.pinv(A)[0].reshape((window_size, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.window_size // 2
        return fft_conv(x, self.kernel, padding=padding, padding_mode='reflect', stride=1)

        

    