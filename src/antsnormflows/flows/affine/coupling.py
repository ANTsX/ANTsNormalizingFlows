
import numpy as np
import torch
from torch import nn

from ..base import Flow, zero_log_det_like_z
from ..reshape import Split, Merge

# -----------------------------------------------------------------------------
# Small helpers used by stabilized heads (kept local to this file)
# -----------------------------------------------------------------------------

def _last_affine(module: nn.Module | None):
    if module is None or not isinstance(module, nn.Module):
        return None
    last = None
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            last = m
    return last


def _zero_init_last_linear_or_conv(module: nn.Module | None):
    last = _last_affine(module)
    if last is not None:
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)


def _apply_spectral_norm_to_last(module: nn.Module | None):
    last = _last_affine(module)
    if last is not None:
        nn.utils.spectral_norm(last)


# -----------------------------------------------------------------------------
# Affine flows (unchanged behavior by default)
# -----------------------------------------------------------------------------

class AffineConstFlow(Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a
    scaling layer which is a special case of this where t is None
    """

    def __init__(self, shape, scale=True, shift=True):
        """Constructor

        Args:
          shape: Shape of the coupling layer
          scale: Flag whether to apply scaling
          shift: Flag whether to apply shift
          logscale_factor: Optional factor which can be used to control the scale of the log scale factor
        """
        super().__init__()
        if scale:
            self.s = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("s", torch.zeros(shape)[None])
        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer("t", torch.zeros(shape)[None])
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(
            torch.tensor(self.s.shape) == 1, as_tuple=False
        )[:, 0].tolist()

    def forward(self, z):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(self.s)
        return z_, log_det

    def inverse(self, z):
        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s)
        return z_, log_det


class CCAffineConst(Flow):
    """
    Affine constant flow layer with class-conditional parameters
    """

    def __init__(self, shape, num_classes):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.s = nn.Parameter(torch.zeros(shape)[None])
        self.t = nn.Parameter(torch.zeros(shape)[None])
        self.s_cc = nn.Parameter(torch.zeros(num_classes, int(np.prod(shape))))
        self.t_cc = nn.Parameter(torch.zeros(num_classes, int(np.prod(shape))))
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(
            torch.tensor(self.s.shape) == 1, as_tuple=False
        )[:, 0].tolist()

    def forward(self, z, y):
        s = self.s + (y @ self.s_cc).view(-1, *self.shape)
        t = self.t + (y @ self.t_cc).view(-1, *self.shape)
        z_ = z * torch.exp(s) + t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det

    def inverse(self, z, y):
        s = self.s + (y @ self.s_cc).view(-1, *self.shape)
        t = self.t + (y @ self.t_cc).view(-1, *self.shape)
        z_ = (z - t) * torch.exp(-s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det


class AffineCoupling(Flow):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    """

    def __init__(self, param_map, scale=True, scale_map="exp", s_cap=None):
        """Constructor

        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow,
                     'sigmoid_inv' uses multiplicative sigmoid scale when sampling from the model
          s_cap: Optional symmetric cap for scale_ via tanh clamp (stability); if None, no clamp applied.
        """
        super().__init__()
        self.add_module("param_map", param_map)
        self.scale = scale
        self.scale_map = scale_map
        self.s_cap = s_cap

    def _bound(self, s_raw):
        if self.s_cap is None:
            return s_raw
        return self.s_cap * torch.tanh(s_raw / self.s_cap)

    def forward(self, z):
        z1, z2 = z
        param = self.param_map(z1)

        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = self._bound(param[:, 1::2, ...])
            reduce_dims = list(range(1, shift.dim()))

            if self.scale_map in ("exp", "tanh"):
                log_s32 = scale_.float()
                with torch.amp.autocast('cuda', enabled=False):
                    s32 = torch.exp(log_s32)
                z2 = z2 * s32.to(z2.dtype) + shift
                log_det = torch.sum(log_s32, dim=reduce_dims).to(z2.dtype)

            elif self.scale_map == "sigmoid":
                scale32 = torch.sigmoid((scale_ + 2).float())
                z2 = z2 / scale32.to(z2.dtype) + shift
                log_det = -torch.sum(torch.log(scale32), dim=reduce_dims).to(z2.dtype)

            elif self.scale_map == "sigmoid_inv":
                scale32 = torch.sigmoid((scale_ + 2).float())
                z2 = z2 * scale32.to(z2.dtype) + shift
                log_det = torch.sum(torch.log(scale32), dim=reduce_dims).to(z2.dtype)

            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 = z2 + param
            log_det = zero_log_det_like_z(z2)

        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        param = self.param_map(z1)

        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = self._bound(param[:, 1::2, ...])
            reduce_dims = list(range(1, shift.dim()))

            if self.scale_map in ("exp", "tanh"):
                log_s32 = scale_.float()
                with torch.amp.autocast('cuda', enabled=False):
                    inv_s32 = torch.exp(-log_s32)
                z2 = (z2 - shift) * inv_s32.to(z2.dtype)
                log_det = -torch.sum(log_s32, dim=reduce_dims).to(z2.dtype)

            elif self.scale_map == "sigmoid":
                scale32 = torch.sigmoid((scale_ + 2).float())
                z2 = (z2 - shift) * scale32.to(z2.dtype)
                log_det = torch.sum(torch.log(scale32), dim=reduce_dims).to(z2.dtype)

            elif self.scale_map == "sigmoid_inv":
                scale32 = torch.sigmoid((scale_ + 2).float())
                z2 = (z2 - shift) / scale32.to(z2.dtype)
                log_det = -torch.sum(torch.log(scale32), dim=reduce_dims).to(z2.dtype)

            else:
                raise NotImplementedError("The scale map " + self.scale_map + " is not implemented.")
        else:
            z2 = z2 - param
            log_det = zero_log_det_like_z(z2)

        return [z1, z2], log_det


# -----------------------------------------------------------------------------
# Stabilized MaskedAffineFlow (drop-in compatible, with extra kwargs)
# -----------------------------------------------------------------------------

class MaskedAffineFlow(Flow):
    """RealNVP as introduced in [arXiv: 1605.08803](https://arxiv.org/abs/1605.08803)

    Masked affine flow:

        f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)

    - class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    - NICE is AffineFlow with only shifts (volume preserving)

    This implementation adds *bounded* log-scales and safe numeric handling.
    """

    def __init__(
        self,
        b,
        t=None,
        s=None,
        *,
        s_cap: float | None = 2.0,
        sanitize_nonfinite: bool = True,
        zero_init_heads: bool = False,
        spectral_norm_last: bool = False,
    ):
        """Constructor (drop-in: b, t=None, s=None; extra kwargs are optional)

        Args:
          b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
          t: translation mapping, i.e. neural network, where first input dimension is batch dim, if None no translation is applied
          s: scale mapping, i.e. neural network, where first input dimension is batch dim, if None no scale is applied
          s_cap: symmetric cap for s via tanh clamp (set None to disable). Typical values 1.5–2.0.
          sanitize_nonfinite: if True, replaces NaN/Inf from s/t with 0 (neutral) before use.
          zero_init_heads: if True, zero-inits the last linear/conv of s and t heads (identity start).
          spectral_norm_last: if True, applies spectral_norm to the last linear/conv of s and t heads.
        """
        super().__init__()
        # ensure float mask and keep a broadcastable buffer
        b = b.float()
        self.register_buffer("b", b.view(1, *b.size()))

        if s is None:
            self.s = torch.zeros_like  # type: ignore[assignment]
        else:
            self.add_module("s", s)

        if t is None:
            self.t = torch.zeros_like  # type: ignore[assignment]
        else:
            self.add_module("t", t)

        self.s_cap = s_cap
        self.sanitize_nonfinite = sanitize_nonfinite

        # Optional head conditioning for stability
        if zero_init_heads:
            _zero_init_last_linear_or_conv(getattr(self, "s", None))
            _zero_init_last_linear_or_conv(getattr(self, "t", None))
        if spectral_norm_last:
            _apply_spectral_norm_to_last(getattr(self, "s", None))
            _apply_spectral_norm_to_last(getattr(self, "t", None))

    # ---- internals ---------------------------------------------------------
    def _bound_scale(self, s_raw):
        if self.s_cap is None:
            return s_raw
        # Smooth symmetric clamp so exp(s) never explodes
        return self.s_cap * torch.tanh(s_raw / self.s_cap)

    def _safe(self, x):
        if not self.sanitize_nonfinite:
            return x
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- forward/inverse ---------------------------------------------------
    def forward(self, z):
        b = self.b.to(dtype=z.dtype, device=z.device)
        z_masked = b * z
        scale_raw = self.s(z_masked)
        trans_raw = self.t(z_masked)
        s = self._safe(self._bound_scale(scale_raw))
        t = self._safe(trans_raw)
        invb = 1.0 - b
        z_ = z_masked + invb * (z * torch.exp(s) + t)
        log_det = torch.sum(invb * s, dim=list(range(1, b.dim())))
        return z_, log_det

    def inverse(self, z):
        b = self.b.to(dtype=z.dtype, device=z.device)
        z_masked = b * z
        scale_raw = self.s(z_masked)
        trans_raw = self.t(z_masked)
        s = self._safe(self._bound_scale(scale_raw))
        t = self._safe(trans_raw)
        invb = 1.0 - b
        z_ = z_masked + invb * (z - t) * torch.exp(-s)
        log_det = -torch.sum(invb * s, dim=list(range(1, b.dim())))
        return z_, log_det

class AffineCouplingBlock(Flow):
    """
    Affine Coupling layer including split and merge operation
    """

    def __init__(self, param_map, scale=True, scale_map="tanh", split_mode="channel", s_cap=3.0):
        """Constructor

        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow
          split_mode: Splitting mode, for possible values see Split class
          s_cap: optional tanh clamp for the internal AffineCoupling's scale head
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [Split(split_mode)]
        # Affine coupling layer (optionally bounded)
        self.flows += [AffineCoupling(param_map, scale, scale_map, s_cap=s_cap)]
        # Merge layer
        self.flows += [Merge(split_mode)]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot
