import torch
import math
import torch.nn as nn

from .base import Flow, stable_fp_dtype
from .affine.coupling import AffineConstFlow

class ActNorm(AffineConstFlow):
    """
    Glow-style ActNorm with data-dependent init.
    Numerics: (a) bound log-scale, (b) FP32 exp/logdet, (c) correct init flag handling.
    Works for ``N, C, *spatial`` (2D, 3D, ...).
    """

    def __init__(self, *args, log_s_cap: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        # registered buffer so it moves with device/checkpoints
        self.register_buffer("data_dep_init_done", torch.tensor(0.0))
        # cap for log-scale (None disables bounding)
        self.log_s_cap = float(log_s_cap)

    @staticmethod
    def _bound_log_s(log_s: torch.Tensor, cap: float | None) -> torch.Tensor:
        if cap is None:
            return log_s
        return cap * torch.tanh(log_s / cap)

    @staticmethod
    def _num_spatial_elems(z: torch.Tensor) -> int:
        # product of all spatial dims after N and C
        return int(math.prod(z.shape[2:])) if len(z.shape) > 2 else 1

    @staticmethod
    def _check_finite(name: str, t: torch.Tensor, context: str) -> None:
        if not torch.isfinite(t).all():
            n_bad = int((~torch.isfinite(t)).sum().item())
            raise RuntimeError(
                f"[ActNorm] non-finite values in '{name}' during {context}: "
                f"{n_bad}/{t.numel()} elements non-finite "
                f"(min={t[torch.isfinite(t)].min().item() if n_bad < t.numel() else float('nan')}, "
                f"shape={tuple(t.shape)}). This is the true origin of the NaN -- "
                f"upstream of wherever autograd's anomaly detector first reports it."
            )

    def _data_dep_init_forward(self, z: torch.Tensor):
        """
        Initialize so that (roughly) out = (z - mean)/std at first batch:
            s := -log(std),   t := -mean * exp(s)
        """
        assert self.s is not None and self.t is not None
        with torch.no_grad():
            self._check_finite("z", z, "forward data-dependent init (input)")
            # compute stats at a stable precision (>= fp32, preserving fp64)
            stable_dtype = stable_fp_dtype(z)
            mean = z.to(stable_dtype).mean(dim=self.batch_dims, keepdim=True)
            std  = z.to(stable_dtype).std( dim=self.batch_dims, keepdim=True) + 1e-6
            self._check_finite("std", std, "forward data-dependent init")
            s_init = -torch.log(std)                              # log-scale
            s_init = self._bound_log_s(s_init, self.log_s_cap)    # bound at init
            self._check_finite("s_init", s_init, "forward data-dependent init")
            self.s.copy_(s_init.to(self.s.dtype))                 # store log-scale

            # t = -mean * exp(s)  (compute exp at stable precision)
            with torch.amp.autocast('cuda', enabled=False):
                exp_s = torch.exp(self.s.to(stable_dtype))
            self.t.copy_((-mean * exp_s).to(self.t.dtype))
            self._check_finite("t", self.t, "forward data-dependent init")

            # IMPORTANT: update buffer in-place (don’t rebind the attribute)
            self.data_dep_init_done.fill_(1.0)

    def forward(self, z: torch.Tensor):
        # one-time data-dependent init
        if not (self.data_dep_init_done > 0.0):
            self._data_dep_init_forward(z)

        # use bounded log-scale and a stable (>= fp32, fp64-preserving) exp/logdet
        stable_dtype = stable_fp_dtype(z)
        log_s = self._bound_log_s(self.s, self.log_s_cap)
        log_s32 = log_s.to(stable_dtype)
        with torch.amp.autocast('cuda', enabled=False):
            s32 = torch.exp(log_s32)  # per-channel scale at stable precision

        z_out = z * s32.to(z.dtype) + self.t

        # log|det J| per sample: (#spatial elements) * sum_c log_s
        num_spatial = self._num_spatial_elems(z)
        # self.s has shape (1, C, *1s) → sum over all but batch gives sum over channels
        logdet_scalar = num_spatial * log_s32.flatten(1).sum(dim=1)  # shape: (1,)
        log_det = logdet_scalar.expand(z.size(0)).to(z.dtype)        # shape: (N,)
        return z_out, log_det

    def inverse(self, z: torch.Tensor):
        stable_dtype = stable_fp_dtype(z)
        # NOTE: for density-estimation training (inverse_and_log_det called
        # on real data), THIS is the path that actually performs the
        # data-dependent init on iteration 1 -- forward() is only exercised
        # by sample()/generation. Despite the old "(rare path)" comment,
        # it is the primary init path in this training setup, so it gets
        # the same finiteness instrumentation as _data_dep_init_forward().
        if not (self.data_dep_init_done > 0.0):
            # mirror of forward init (kept for completeness)
            assert self.s is not None and self.t is not None
            with torch.no_grad():
                self._check_finite("z", z, "inverse data-dependent init (input)")
                mean = z.to(stable_dtype).mean(dim=self.batch_dims, keepdim=True)
                std  = z.to(stable_dtype).std( dim=self.batch_dims, keepdim=True) + 1e-6
                self._check_finite("std", std, "inverse data-dependent init")
                s_init = torch.log(std)
                s_init = self._bound_log_s(s_init, self.log_s_cap)
                self._check_finite("s_init", s_init, "inverse data-dependent init")
                self.s.copy_(s_init.to(self.s.dtype))
                self.t.copy_(mean.to(self.t.dtype))
                self._check_finite("t", self.t, "inverse data-dependent init")
                self.data_dep_init_done.fill_(1.0)

        log_s = self._bound_log_s(self.s, self.log_s_cap)
        log_s32 = log_s.to(stable_dtype)
        with torch.amp.autocast('cuda', enabled=False):
            inv_s32 = torch.exp(-log_s32)

        z_out = (z - self.t) * inv_s32.to(z.dtype)

        num_spatial = self._num_spatial_elems(z)
        logdet_scalar = num_spatial * log_s32.flatten(1).sum(dim=1)
        log_det = (-logdet_scalar).expand(z.size(0)).to(z.dtype)
        return z_out, log_det

class BatchNorm(Flow):
    """
    RealNVP-style BatchNorm that ignores gradients through batch stats.
    Works for inputs shaped ``(N, C, *spatial)``. Computes per-channel stats over
    batch and spatial dimensions.
    """

    def __init__(self, eps: float = 1e-6, detach_stats: bool = True):
        super().__init__()
        self.register_buffer("eps", torch.tensor(float(eps)))
        self.detach_stats = bool(detach_stats)

    @staticmethod
    def _reduce_dims(z: torch.Tensor):
        # reduce over batch and all spatial dims; keep channel
        return (0,) + tuple(range(2, z.dim()))

    @staticmethod
    def _num_spatial(z: torch.Tensor) -> int:
        return int(math.prod(z.shape[2:])) if z.dim() > 2 else 1

    def forward(self, z: torch.Tensor):
        # per-channel mean/var over batch+spatial, at a stable precision
        # (>= fp32, preserving fp64)
        stable_dtype = stable_fp_dtype(z)
        dims = self._reduce_dims(z)
        z_stable = z.to(stable_dtype)
        mean = z_stable.mean(dim=dims, keepdim=True)
        var  = z_stable.var( dim=dims, keepdim=True, unbiased=False)
        if self.detach_stats:
            mean = mean.detach()
            var  = var.detach()

        eps_stable = self.eps.to(stable_dtype)

        # normalize at stable precision (AMP-safe), cast result back
        with torch.amp.autocast('cuda', enabled=False):
            inv_std_stable = torch.rsqrt(var + eps_stable)  # = 1/sqrt(var+eps)
        z_hat = (z - mean.to(z.dtype)) * inv_std_stable.to(z.dtype)

        # log|det J| per sample: (#spatial) * sum_c log(inv_std) = -0.5 * (#spatial) * sum_c log(var+eps)
        with torch.amp.autocast('cuda', enabled=False):
            logdet_per_channel = (-0.5) * torch.log(var + eps_stable).flatten(1).sum(dim=1)  # (1,)
        log_det = (self._num_spatial(z) * logdet_per_channel).to(z.dtype)                    # (N,)
        return z_hat, log_det

    def inverse(self, z: torch.Tensor):
        # recompute the same batch stats (detach if desired)
        stable_dtype = stable_fp_dtype(z)
        dims = self._reduce_dims(z)
        z_stable = z.to(stable_dtype)
        mean = z_stable.mean(dim=dims, keepdim=True)
        var  = z_stable.var( dim=dims, keepdim=True, unbiased=False)
        if self.detach_stats:
            mean = mean.detach()
            var  = var.detach()

        eps_stable = self.eps.to(stable_dtype)
        with torch.amp.autocast('cuda', enabled=False):
            std_stable = torch.sqrt(var + eps_stable)

        z_out = z * std_stable.to(z.dtype) + mean.to(z.dtype)

        # inverse logdet is the negative of forward's
        with torch.amp.autocast('cuda', enabled=False):
            logdet_per_channel = (+0.5) * torch.log(var + eps_stable).flatten(1).sum(dim=1)
        log_det = (self._num_spatial(z) * logdet_per_channel).to(z.dtype)
        return z_out, log_det
