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


class ConvNet3d(nn.Module):
    """
    3D Convolutional Neural Network with leaky ReLU nonlinearities
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
            conv = nn.Conv3d(
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
                net.append(utils.ActNorm((channels[i + 1], 1, 1, 1)))
            net.append(nn.LeakyReLU(leaky))

        # Final conv layer
        net.append(
            nn.Conv3d(
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
