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
