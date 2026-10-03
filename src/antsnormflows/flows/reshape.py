import torch
import torch.nn as nn
import numpy as np

from .base import Flow, zero_log_det_like_z


# Flow layers to reshape the latent features

class Split(Flow):
    """
    Split features into two sets
    """

    def __init__(self, mode="channel"):
        super().__init__()
        self.mode = mode

    def forward(self, z):
        if self.mode == "channel":
            z1, z2 = z.chunk(2, dim=1)
        elif self.mode == "channel_inv":
            z2, z1 = z.chunk(2, dim=1)
        elif "checkerboard" in self.mode:
            n_dims = z.dim()
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z.size(n_dims - i))]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z.size(n_dims - i))]
            cb = cb1 if "inv" in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(len(z), *((n_dims - 1) * [1]))
            cb = cb.to(z.device)
            z_size = z.size()
            z1 = z.reshape(-1)[torch.nonzero(cb.view(-1), as_tuple=False)].view(
                *z_size[:-1], -1
            )
            z2 = z.reshape(-1)[torch.nonzero((1 - cb).view(-1), as_tuple=False)].view(
                *z_size[:-1], -1
            )
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = zero_log_det_like_z(z)
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        if self.mode == "channel":
            z = torch.cat([z1, z2], 1)
        elif self.mode == "channel_inv":
            z = torch.cat([z2, z1], 1)
        elif "checkerboard" in self.mode:
            n_dims = z1.dim()
            z_size = list(z1.size())
            z_size[-1] *= 2
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z_size[n_dims - i])]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z_size[n_dims - i])]
            cb = cb1 if "inv" in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(z_size[0], *((n_dims - 1) * [1]))
            cb = cb.to(z1.device)
            z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z = cb * z1 + (1 - cb) * z2
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = zero_log_det_like_z(z)
        return z, log_det


class Merge(Split):
    """
    Same as Split but with forward and backward pass interchanged
    """

    def __init__(self, mode="channel"):
        super().__init__(mode)

    def forward(self, z):
        z_out, log_det = super().inverse(z)
        return z_out, log_det

    def inverse(self, z):
        z_out, log_det = super().forward(z)
        return z_out, log_det


class Squeeze2d(Flow):
    """
    Squeeze operation of multi-scale architecture (RealNVP / Glow),
    which trades spatial resolution for more channels.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z):
        s = z.size()
        z = z.view(s[0], s[1] // 4, 2, 2, s[2], s[3])
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        z = z.view(s[0], s[1] // 4, 2 * s[2], 2 * s[3])
        log_det = zero_log_det_like_z(z)
        return z, log_det

    def inverse(self, z):
        s = z.size()
        z = z.view(s[0], s[1], s[2] // 2, 2, s[3] // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4).contiguous()
        z = z.view(s[0], 4 * s[1], s[2] // 2, s[3] // 2)
        log_det = zero_log_det_like_z(z)
        return z, log_det

class Squeeze3d(Flow):
    """
    Squeeze operation for 3D images, extending the concept from Glow.

    ``factors`` gives the squeeze factor of each of the three spatial axes (default ``(2, 2, 2)``, the
    historical behavior: 2x2x2 neighborhoods are packed into channels). A factor of 1 leaves that axis
    untouched, e.g. ``(1, 2, 2)`` squeezes only the last two axes, which keeps the first axis (time, for
    video-like windows) at full resolution; channels are multiplied by the product of the factors.
    """

    def __init__(self, factors=(2, 2, 2)):
        """
        Constructor

        Args:
          factors: squeeze factor (>= 1) of each spatial axis
        """
        super().__init__()
        factors = tuple(int(f) for f in factors)
        if len(factors) != 3 or any(f < 1 for f in factors):
            raise ValueError(f"Squeeze3d: factors must be three integers >= 1, got {factors}.")
        self.factors = factors
        self.num = factors[0] * factors[1] * factors[2]

    def forward(self, z):
        ft, fh, fw = self.factors
        s = z.size()
        if s[1] % self.num:
            raise ValueError(f"Squeeze3d{self.factors}: {s[1]} channels are not divisible by {self.num}.")
        z = z.view(s[0], s[1] // self.num, ft, fh, fw, s[2], s[3], s[4])
        z = z.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        z = z.view(s[0], s[1] // self.num, s[2] * ft, s[3] * fh, s[4] * fw)
        log_det = zero_log_det_like_z(z)
        return z, log_det

    def inverse(self, z):
        ft, fh, fw = self.factors
        s = z.size()
        if s[2] % ft or s[3] % fh or s[4] % fw:
            raise ValueError(f"Squeeze3d{self.factors}: spatial shape {tuple(s[2:])} is not divisible by the factors.")
        z = z.view(s[0], s[1], s[2] // ft, ft, s[3] // fh, fh, s[4] // fw, fw)
        z = z.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        z = z.view(s[0], self.num * s[1], s[2] // ft, s[3] // fh, s[4] // fw)
        log_det = zero_log_det_like_z(z)
        return z, log_det
    
