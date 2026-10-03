from torch import nn
from .. import utils

class ConvNet2d(nn.Module):
    """
    2D Convolutional Neural Network with leaky ReLU nonlinearities
    """

    def __init__(
        self,
        channels,
        kernel_size,
        leaky=0.0,
        init_zeros=True,
        actnorm=False,
        weight_std=None,
    ):
        super().__init__()
        net = nn.ModuleList([])
        for i in range(len(kernel_size) - 1):
            conv = nn.Conv2d(
                channels[i],
                channels[i + 1],
                kernel_size[i],
                padding=kernel_size[i] // 2,
                bias=(not actnorm),
            )
            if weight_std is not None:
                conv.weight.data.normal_(mean=0.0, std=weight_std)
            net.append(conv)
            if actnorm:
                net.append(utils.ActNorm((channels[i + 1], 1, 1)))
            net.append(nn.LeakyReLU(leaky))

        # Final conv layer
        net.append(
            nn.Conv2d(
                channels[-2],
                channels[-1],
                kernel_size[-1],
                padding=kernel_size[-1] // 2,
            )
        )
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

        self.net = nn.Sequential(*net)

    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
        return x


def _as_kernel_3d(k):
    """Kernel entry of ConvNet3d: an int (isotropic, as before) or a 3-sequence (one size per axis)."""
    if isinstance(k, int):
        return k
    k = tuple(int(v) for v in k)
    if len(k) != 3:
        raise ValueError(f"A 3D kernel needs 3 sizes (one per axis), got {k}.")
    if any(v % 2 == 0 for v in k):
        raise ValueError(f"Per-axis kernel sizes must be odd to keep the shape ('same' padding), got {k}.")
    return k


def _same_padding_3d(k):
    """'same' padding of a kernel entry. Ints keep the historical behavior (k // 2)."""
    return k // 2 if isinstance(k, int) else tuple(v // 2 for v in k)


class ConvNet3d(nn.Module):
    """
    3D Convolutional Neural Network with leaky ReLU nonlinearities

    Each entry of ``kernel_size`` is either an int (isotropic kernel, the historical behavior) or a
    3-sequence with one odd size per axis, e.g. ``(1, 3, 3)`` for a purely spatial kernel or ``(3, 1, 1)``
    for a purely temporal one when the first spatial axis is time.
    """

    def __init__(
        self,
        channels,
        kernel_size,
        leaky=0.0,
        init_zeros=True,
        actnorm=False,
        weight_std=None,
    ):
        super().__init__()
        kernel_size = [_as_kernel_3d(k) for k in kernel_size]
        net = nn.ModuleList([])
        for i in range(len(kernel_size) - 1):
            conv = nn.Conv3d(
                channels[i],
                channels[i + 1],
                kernel_size[i],
                padding=_same_padding_3d(kernel_size[i]),
                bias=(not actnorm),
            )
            if weight_std is not None:
                conv.weight.data.normal_(mean=0.0, std=weight_std)
            net.append(conv)
            if actnorm:
                net.append(utils.ActNorm((channels[i + 1], 1, 1, 1)))
            net.append(nn.LeakyReLU(leaky))

        # Final conv layer
        net.append(
            nn.Conv3d(
                channels[-2],
                channels[-1],
                kernel_size[-1],
                padding=_same_padding_3d(kernel_size[-1]),
            )
        )
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

        self.net = nn.Sequential(*net)

    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
        return x


class ConvNet1d(nn.Module):
    """
    1D Convolutional Neural Network with leaky ReLU nonlinearities

    Args:
      channels: Number of channels of each layer (len(kernel_size) + 1 entries)
      kernel_size: Kernel size of each layer (odd sizes keep the length unchanged)
      leaky: Leaky ReLU slope
      init_zeros: Initialize the last layer with zeros (identity coupling at init)
      actnorm: Use ActNorm (instead of bias) after the hidden convolutions
      weight_std: Optional std for normal initialization of hidden weights
      padding_mode: Padding for every convolution, passed to ``nn.Conv1d``
        ("zeros", "circular", "reflect" or "replicate"). Use "circular" for
        periodic signals (e.g. a phase-normalized gait cycle) so that the
        first and last time points are treated as neighbours instead of
        being padded with artificial zeros.
    """

    def __init__(
        self,
        channels,
        kernel_size,
        leaky=0.0,
        init_zeros=True,
        actnorm=False,
        weight_std=None,
        padding_mode="zeros",
    ):
        super().__init__()
        net = nn.ModuleList([])
        for i in range(len(kernel_size) - 1):
            conv = nn.Conv1d(
                channels[i],
                channels[i + 1],
                kernel_size[i],
                padding=kernel_size[i] // 2,
                bias=(not actnorm),
                padding_mode=padding_mode,
            )
            if weight_std is not None:
                conv.weight.data.normal_(mean=0.0, std=weight_std)
            net.append(conv)
            if actnorm:
                net.append(utils.ActNorm((channels[i + 1], 1)))
            net.append(nn.LeakyReLU(leaky))

        # Final conv layer
        net.append(
            nn.Conv1d(
                channels[-2],
                channels[-1],
                kernel_size[-1],
                padding=kernel_size[-1] // 2,
                padding_mode=padding_mode,
            )
        )
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

        self.net = nn.Sequential(*net)

    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
        return x
