import torch
from torch import nn

from ..base import Flow
from .coupling import AffineCouplingBlock
from ..mixing import Invertible1x1Conv, Invertible1x1x1Conv
from ..normalization import ActNorm
from ... import nets

class GlowBlock2d(Flow):
    """Glow: Generative Flow with Invertible 1×1 Convolutions, [arXiv: 1807.03039]"""

    def __init__(
        self,
        channels,
        hidden_channels,
        scale=True,
        scale_map="tanh",
        split_mode="channel",
        leaky=0.1,
        init_zeros=True,
        use_lu=True,
        net_actnorm=True,
        s_cap=2.0
    ):
        super().__init__()
        self.flows = nn.ModuleList([])
        self.channels = channels

        # Coupling layer
        kernel_size = (3, 1, 3)
        num_param = 2 if scale else 1

        if "channel" == split_mode:
            channels_ = ((channels + 1) // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * (channels // 2),)
        elif "channel_inv" == split_mode:
            channels_ = (channels // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * ((channels + 1) // 2),)
        elif "checkerboard" in split_mode:
            channels_ = (channels,) + 2 * (hidden_channels,)
            channels_ += (num_param * channels,)
        else:
            raise NotImplementedError("Mode " + split_mode + " is not implemented.")
        param_map = nets.ConvNet2d(
            channels_, kernel_size, leaky, init_zeros, actnorm=net_actnorm
        )

        self.flows.append(ActNorm((channels,) + (1, 1)))
        if channels > 1:
            self.flows.append(Invertible1x1Conv(channels, use_lu))
        self.flows.append(AffineCouplingBlock(param_map, scale, scale_map, split_mode, s_cap))

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

        for idx, flow in enumerate(self.flows):
            z, log_det = flow(z)
            log_det_tot += log_det

        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

        for idx, flow in enumerate(reversed(self.flows)):
            z, log_det = flow.inverse(z)
            log_det_tot += log_det

        return z, log_det_tot


class GlowBlock3d(Flow):
    """Glow: Generative Flow with Invertible 1×1x1 Convolutions, [arXiv: 1807.03039](https://arxiv.org/abs/1807.03039)

    One Block of the Glow model, comprised of

    - MaskedAffineFlow (affine coupling layer)
    - Invertible1x1x1Conv (dropped if there is only one channel)
    - ActNorm (first batch used for initialization)
    """

    def __init__(
        self,
        channels,
        hidden_channels,
        scale=True,
        scale_map="tanh",
        split_mode="channel",
        leaky=0.1,
        init_zeros=True,
        use_lu=True,
        net_actnorm=True,
        s_cap=2.0
    ):
        """Constructor

        Args:
          channels: Number of channels of the data
          hidden_channels: number of channels in the hidden layer of the ConvNet
          scale: Flag, whether to include scale in affine coupling layer
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow
          split_mode: Splitting mode, for possible values see Split class
          leaky: Leaky parameter of LeakyReLUs of ConvNet2d
          init_zeros: Flag whether to initialize last conv layer with zeros
          use_lu: Flag whether to parametrize weights through the LU decomposition in invertible 1x1 convolution layers
          logscale_factor: Factor which can be used to control the scale of the log scale factor, see [source](https://github.com/openai/glow)
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Coupling layer
        kernel_size = (3, 1, 3)
        num_param = 2 if scale else 1
        if "channel" == split_mode:
            channels_ = ((channels + 1) // 2,) + 3 * (hidden_channels,)
            channels_ += (num_param * (channels // 2),)
        elif "channel_inv" == split_mode:
            channels_ = (channels // 2,) + 3 * (hidden_channels,)
            channels_ += (num_param * ((channels + 1) // 2),)
        elif "checkerboard" in split_mode:
            channels_ = (channels,) + 3 * (hidden_channels,)
            channels_ += (num_param * channels,)
        else:
            raise NotImplementedError("Mode " + split_mode + " is not implemented.")
        param_map = nets.ConvNet3d(
            channels_, kernel_size, leaky, init_zeros, actnorm=net_actnorm
        )

        self.flows += [ActNorm((channels,) + (1, 1, 1))]
        if channels > 1:
            self.flows += [Invertible1x1x1Conv(channels, use_lu)]
        self.flows += [AffineCouplingBlock(param_map, scale, scale_map, split_mode, s_cap)]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

        for idx, flow in enumerate(self.flows):
            z, log_det = flow(z)
            log_det_tot += log_det

        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

        for idx, flow in enumerate(reversed(self.flows)):
            z, log_det = flow.inverse(z)
            log_det_tot += log_det

        return z, log_det_tot
