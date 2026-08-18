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
        s_cap=2.0,
        conv_s_cap=None,
        actnorm_s_cap=None,
        gen_clamp=1.0e4,
    ):
        """
        conv_s_cap: optional override for the Invertible1x1Conv's own log-scale
        clamp. If None (default), the conv uses `s_cap` (matching the caller's
        requested scale_cap). Pass an explicit value (e.g. 2.5) to reproduce
        the pre-fix behavior for checkpoints trained before the invertible
        conv correctly received `s_cap` -- those checkpoints' weights were
        calibrated against the conv's old hardcoded default, not against the
        configured `scale_cap`.
        actnorm_s_cap: optional override for this block's ActNorm log-scale
        clamp. If None (default), ActNorm uses `s_cap` (matching the caller's
        requested scale_cap) instead of its own hardcoded default (5.0). Pass
        an explicit value to reproduce the pre-fix behavior for checkpoints
        trained before ActNorm correctly received `s_cap`.
        gen_clamp: symmetric bound applied to the block's output tensor
        (via nan_to_num + clamp) after each sub-flow, in BOTH forward() and
        inverse() -- a local blowup in one channel would otherwise
        contaminate the whole tensor within a couple of blocks, since
        Invertible1x1Conv mixes all channels linearly. Default 1.0e4
        matches the value this was previously hardcoded to; pass an
        explicit value to reproduce checkpoints trained under a different
        hardcoded default, or to tighten/loosen the safety margin. This is
        a safety net, not a tuning knob for accuracy -- if training hits
        this bound often, that is a signal to investigate the underlying
        instability (e.g. via the same s_cap/conv_s_cap/actnorm_s_cap
        parameters) rather than to raise gen_clamp further.
        """
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

        # Bind this ActNorm's log-scale clamp to the caller's requested
        # scale_cap instead of silently using ActNorm's own hardcoded
        # default (5.0), unless actnorm_s_cap explicitly overrides it.
        self.flows.append(ActNorm(
            (channels,) + (1, 1),
            log_s_cap=(actnorm_s_cap if actnorm_s_cap is not None else s_cap),
        ))
        if channels > 1:
            # Pass s_cap through so the invertible 1x1 conv's log-scale clamp
            # matches the caller's requested scale_cap instead of silently
            # using Invertible1x1Conv's own hardcoded default (2.5), unless
            # conv_s_cap explicitly overrides it (legacy-checkpoint mode).
            self.flows.append(Invertible1x1Conv(
                channels, use_lu,
                s_cap=(conv_s_cap if conv_s_cap is not None else s_cap),
            ))
        self.flows.append(AffineCouplingBlock(param_map, scale, scale_map, split_mode, s_cap))

        # Bound applied in BOTH directions (forward()/generation AND
        # inverse()/training), after each sub-flow, to stop a local
        # blowup from cascading through the remaining blocks --
        # Invertible1x1Conv mixes all channels linearly, so a single
        # non-finite channel otherwise contaminates the whole tensor
        # within a couple of blocks. Configurable via `gen_clamp`; see
        # its docstring above for checkpoint-compatibility notes.
        self._gen_clamp = float(gen_clamp)

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

        for idx, flow in enumerate(self.flows):
            z, log_det = flow(z)
            z = torch.nan_to_num(z, nan=0.0, posinf=self._gen_clamp, neginf=-self._gen_clamp)
            z = torch.clamp(z, -self._gen_clamp, self._gen_clamp)
            log_det_tot += log_det

        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

        for idx, flow in enumerate(reversed(self.flows)):
            z, log_det = flow.inverse(z)
            z = torch.nan_to_num(z, nan=0.0, posinf=self._gen_clamp, neginf=-self._gen_clamp)
            z = torch.clamp(z, -self._gen_clamp, self._gen_clamp)
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
        s_cap=2.0,
        conv_s_cap=None,
        actnorm_s_cap=None,
        gen_clamp=1.0e4,
        shift_cap=None,
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
          conv_s_cap: optional override for the Invertible1x1x1Conv's own log-scale
            clamp. If None (default), the conv uses `s_cap`. Pass an explicit
            value (e.g. 2.5) to reproduce the pre-fix behavior for checkpoints
            trained before the invertible conv correctly received `s_cap`.
          actnorm_s_cap: optional override for this block's ActNorm log-scale
            clamp. If None (default), ActNorm uses `s_cap` instead of its own
            hardcoded default (5.0). Pass an explicit value to reproduce the
            pre-fix behavior for checkpoints trained before ActNorm correctly
            received `s_cap`.
          shift_cap: optional tanh clamp on the affine coupling's additive
            shift term (see AffineCoupling's docstring for why this exists
            separately from s_cap/conv_s_cap/actnorm_s_cap -- shift has no
            bounding nonlinearity of its own and can compound block-to-block
            once inputs drift out of the training distribution, e.g. when
            sampling at temperature > 1). None (default) reproduces exact
            prior behavior/checkpoints; pass a value generous relative to
            typical in-distribution activation magnitude to add a safety
            margin without perturbing normal-range sampling/reconstruction.
          gen_clamp: symmetric bound applied to the block's output tensor
            (via nan_to_num + clamp) after each sub-flow, in BOTH forward()
            and inverse() -- a local blowup in one channel would otherwise
            contaminate the whole tensor within a couple of blocks, since
            Invertible1x1x1Conv mixes all channels linearly. Default 1.0e4
            matches the value this was previously hardcoded to; pass an
            explicit value to reproduce checkpoints trained under a
            different hardcoded default, or to tighten/loosen the safety
            margin. This is a safety net, not a tuning knob for accuracy --
            if training hits this bound often, that is a signal to
            investigate the underlying instability (e.g. via
            s_cap/conv_s_cap/actnorm_s_cap) rather than to raise gen_clamp
            further.
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

        # Bind this ActNorm's log-scale clamp to the caller's requested
        # scale_cap instead of silently using ActNorm's own hardcoded
        # default (5.0), unless actnorm_s_cap explicitly overrides it.
        self.flows += [ActNorm(
            (channels,) + (1, 1, 1),
            log_s_cap=(actnorm_s_cap if actnorm_s_cap is not None else s_cap),
        )]
        if channels > 1:
            # Pass s_cap through so the invertible 1x1x1 conv's log-scale
            # clamp matches the caller's requested scale_cap instead of
            # silently using Invertible1x1x1Conv's own hardcoded default (2.5),
            # unless conv_s_cap explicitly overrides it (legacy-checkpoint mode).
            self.flows += [Invertible1x1x1Conv(
                channels, use_lu,
                s_cap=(conv_s_cap if conv_s_cap is not None else s_cap),
            )]
        self.flows += [AffineCouplingBlock(param_map, scale, scale_map, split_mode, s_cap, t_cap=shift_cap)]

        # Bound applied in BOTH directions (forward()/generation AND
        # inverse()/training), after each sub-flow, to stop a local
        # blowup from cascading through the remaining blocks --
        # Invertible1x1x1Conv mixes all channels linearly, so a single
        # non-finite channel otherwise contaminates the whole tensor
        # within a couple of blocks. Configurable via `gen_clamp`; see
        # its docstring above for checkpoint-compatibility notes.
        self._gen_clamp = float(gen_clamp)

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

        for idx, flow in enumerate(self.flows):
            z, log_det = flow(z)
            z = torch.nan_to_num(z, nan=0.0, posinf=self._gen_clamp, neginf=-self._gen_clamp)
            z = torch.clamp(z, -self._gen_clamp, self._gen_clamp)
            log_det_tot += log_det

        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

        for idx, flow in enumerate(reversed(self.flows)):
            z, log_det = flow.inverse(z)
            z = torch.nan_to_num(z, nan=0.0, posinf=self._gen_clamp, neginf=-self._gen_clamp)
            z = torch.clamp(z, -self._gen_clamp, self._gen_clamp)
            log_det_tot += log_det

        return z, log_det_tot