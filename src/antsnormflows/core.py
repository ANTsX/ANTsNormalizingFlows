import torch
import torch.nn as nn
import contextlib
import numpy as np

from typing import List, Optional, Sequence, Tuple

from . import distributions
from . import utils

import torch.utils.checkpoint as checkpoint


def _apply_flow_sequence(flows, z, call_flow, reverse=False, use_checkpoint=False,
                          checkpoint_args=()):
    """Applies a sequence of flows (or their inverse) to a batch, accumulating
    the total log-determinant in float32.

    Factors out the checkpointing/accumulation logic shared by
    `NormalizingFlow` and `ConditionalNormalizingFlow`'s
    `forward_and_log_det`/`inverse_and_log_det` methods.

    Args:
      flows: nn.ModuleList of flows
      z: Batch to transform
      call_flow: Callable(flow, z, *checkpoint_args) -> (new_z, log_det)
        that invokes `flow`/`flow.inverse` with whatever extra arguments
        it needs (e.g. context)
      reverse: Iterate the flows back-to-front (used for the inverse
        direction)
      use_checkpoint: Wrap each flow call in `torch.utils.checkpoint` to
        trade compute for memory
      checkpoint_args: Extra tensor arguments (e.g. context) forwarded to
        `call_flow`. Passed explicitly through `checkpoint.checkpoint` (as
        opposed to only via closure) so that gradients w.r.t. these
        tensors are tracked correctly when checkpointing is enabled.

    Returns:
      Tuple of (transformed batch, total log-determinant)
    """
    log_det = torch.zeros(len(z), device=z.device, dtype=torch.float32)
    indices = range(len(flows) - 1, -1, -1) if reverse else range(len(flows))

    for i in indices:
        flow = flows[i]
        if use_checkpoint:
            def _run(latent, *extra, f=flow):
                return call_flow(f, latent, *extra)
            z, log_d = checkpoint.checkpoint(_run, z, *checkpoint_args, use_reentrant=False)
        else:
            z, log_d = call_flow(flow, z, *checkpoint_args)
        log_det += log_d.float()

    return z, log_det


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """

    def __init__(self, q0: nn.Module, flows: Sequence[nn.Module], p: Optional[nn.Module] = None):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
          p: Target distribution
        """
        super().__init__()
        if q0 is None:
            raise ValueError("NormalizingFlow: q0 (base distribution) must not be None")
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Transforms latent variable z to the flow variable x

        Note:
          Calling the model as `model(z)` performs this z -> x transform;
          it does NOT compute a training loss. This differs from
          `MultiscaleFlow.forward`, which returns a negative
          log-likelihood. Use `forward_kld`/`reverse_kld` for losses.

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z)
        return z

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x)
        return x

    def forward_and_log_det(self, z, dtype=None):
        if dtype is not None:
            ctx = torch.amp.autocast(device_type=z.device.type, dtype=dtype)
        else:
            ctx = contextlib.nullcontext()

        use_checkpoint = self.training and len(self.flows) > 10

        with ctx:
            z, log_det = _apply_flow_sequence(
                self.flows, z,
                call_flow=lambda flow, zz: flow(zz),
                use_checkpoint=use_checkpoint,
            )

        return z, log_det

    def inverse_and_log_det(self, x, dtype=None):
        if dtype is not None:
            ctx = torch.amp.autocast(device_type=x.device.type, dtype=dtype)
        else:
            ctx = contextlib.nullcontext()

        use_checkpoint = self.training and len(self.flows) > 10

        with ctx:
            x, log_det = _apply_flow_sequence(
                self.flows, x,
                call_flow=lambda flow, xx: flow.inverse(xx),
                reverse=True,
                use_checkpoint=use_checkpoint,
            )

        return x, log_det

    def forward_kld(self, x: torch.Tensor) -> torch.Tensor:
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples: int = 1, beta: float = 1.0, score_fn: bool = True) -> torch.Tensor:
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          num_samples: Number of samples to draw from base distribution
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q_ = self.q0(num_samples)
        if score_fn:
            log_q = log_q_.clone()
            for flow in self.flows:
                z, log_det = flow(z)
                log_q -= log_det
        else:
            # log_q is recomputed below via the inverse pass with gradients
            # disabled, so we skip accumulating it during the forward pass
            # here (it would just be discarded).
            for flow in self.flows:
                z, _ = flow(z)
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            grad_states = utils.get_requires_grad_states(self)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            utils.restore_requires_grad(self, grad_states)
        log_p = self.p.log_prob(z)
        return torch.mean(log_q) - beta * torch.mean(log_p)

    def reverse_alpha_div(self, num_samples: int = 1, alpha: float = 1, dreg: bool = False) -> torch.Tensor:
        """Alpha divergence when sampling from q

        Args:
          num_samples: Number of samples to draw
          dreg: Flag whether to use Double Reparametrized Gradient estimator, see [arXiv 1810.04152](https://arxiv.org/abs/1810.04152)

        Returns:
          Alpha divergence
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.p.log_prob(z)
        if dreg:
            # Note: unlike reverse_kld, the forward-pass log_q above is not
            # wasted here even when dreg=True: w_const depends on it before
            # log_q gets recomputed via the inverse pass below.
            w_const = torch.exp(log_p - log_q).detach()
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            grad_states = utils.get_requires_grad_states(self)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            utils.restore_requires_grad(self, grad_states)
            w = torch.exp(log_p - log_q)
            w_alpha = w_const**alpha
            w_alpha = w_alpha / torch.mean(w_alpha)
            weights = (1 - alpha) * w_alpha + alpha * w_alpha**2
            loss = -alpha * torch.mean(weights * torch.log(w))
        else:
            loss = np.sign(alpha - 1) * torch.logsumexp(alpha * (log_p - log_q), 0)
        return loss

    def sample(self, num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Get log probability for batch

        Args:
          x: Batch

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q

    def save(self, path: str) -> None:
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location="cpu") -> None:
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
          map_location: Device to map the loaded tensors to (see
            `torch.load`). Defaults to "cpu" for portability across
            machines/devices; call `.to(device)` on the model afterwards
            if needed.
        """
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict)

class ConditionalNormalizingFlow(NormalizingFlow):
    """
    Conditional normalizing flow model, providing condition,
    which is also called context, to both the base distribution
    and the flow layers
    """
    def forward(self, z: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transforms latent variable z to the flow variable x

        Note:
          Like `NormalizingFlow.forward`, calling the model as
          `model(z, context=...)` performs this z -> x transform, not a
          loss computation. Use `forward_kld`/`reverse_kld` for losses.

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z, context=context)
        return z

    def inverse(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x, context=context)
        return x

    def forward_and_log_det(self, z, context=None, dtype=None):
        if dtype is not None:
            ctx = torch.amp.autocast(device_type=z.device.type, dtype=dtype)
        else:
            ctx = contextlib.nullcontext()

        use_checkpoint = self.training and len(self.flows) > 10

        with ctx:
            z, log_det = _apply_flow_sequence(
                self.flows, z,
                call_flow=lambda flow, zz, ctx_val: flow(zz, context=ctx_val),
                checkpoint_args=(context,),
                use_checkpoint=use_checkpoint,
            )

        return z, log_det

    def inverse_and_log_det(self, x, context=None, dtype=None):
        if dtype is not None:
            ctx = torch.amp.autocast(device_type=x.device.type, dtype=dtype)
        else:
            ctx = contextlib.nullcontext()

        use_checkpoint = self.training and len(self.flows) > 10

        with ctx:
            x, log_det = _apply_flow_sequence(
                self.flows, x,
                call_flow=lambda flow, xx, ctx_val: flow.inverse(xx, context=ctx_val),
                reverse=True,
                checkpoint_args=(context,),
                use_checkpoint=use_checkpoint,
            )

        return x, log_det

    def sample(self, num_samples: int = 1, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          context: Batch of conditions/context

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples, context=context)
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get log probability for batch

        Args:
          x: Batch
          context: Batch of conditions/context

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return log_q

    def forward_kld(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          context: Batch of conditions/context

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples: int = 1, context: Optional[torch.Tensor] = None,
                    beta: float = 1.0, score_fn: bool = True) -> torch.Tensor:
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          num_samples: Number of samples to draw from base distribution
          context: Batch of conditions/context
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q_ = self.q0(num_samples, context=context)
        if score_fn:
            log_q = log_q_.clone()
            for flow in self.flows:
                z, log_det = flow(z, context=context)
                log_q -= log_det
        else:
            # log_q is recomputed below via the inverse pass with gradients
            # disabled, so we skip accumulating it during the forward pass
            # here (it would just be discarded).
            for flow in self.flows:
                z, _ = flow(z, context=context)
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            grad_states = utils.get_requires_grad_states(self)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_, context=context)
                log_q += log_det
            log_q += self.q0.log_prob(z_, context=context)
            utils.restore_requires_grad(self, grad_states)
        log_p = self.p.log_prob(z, context=context)
        return torch.mean(log_q) - beta * torch.mean(log_p)

class ClassCondFlow(nn.Module):
    """
    Class conditional normalizing Flow model, providing the
    class to be conditioned on only to the base distribution,
    as done e.g. in [Glow](https://arxiv.org/abs/1807.03039)

    Note:
      This class does not define `forward()` (calling `model(x)` directly
      will raise a `NotImplementedError` from `nn.Module`). Use
      `forward_kld`, `log_prob`, or `sample` instead.
    """

    def __init__(self, q0: nn.Module, flows: Sequence[nn.Module]):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
        """
        super().__init__()
        if q0 is None:
            raise ValueError("ClassCondFlow: q0 (base distribution) must not be None")
        self.q0 = q0
        self.flows = nn.ModuleList(flows)

    def forward_kld(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, y)
        return -torch.mean(log_q)

    def sample(self, num_samples: int = 1, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          y: Classes to sample from, will be sampled uniformly if None

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples, y)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Get log probability for batch

        Args:
          x: Batch
          y: Classes of x

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, y)
        return log_q

    def save(self, path: str) -> None:
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location="cpu") -> None:
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
          map_location: Device to map the loaded tensors to (see
            `torch.load`). Defaults to "cpu" for portability across
            machines/devices; call `.to(device)` on the model afterwards
            if needed.
        """
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict)

class MultiscaleFlow(nn.Module):
    """
    Normalizing Flow model with multiscale architecture, see RealNVP or Glow paper
    """

    def __init__(self, q0: Sequence[nn.Module], flows: Sequence[Sequence[nn.Module]],
                 merges: Sequence[nn.Module], transform: Optional[nn.Module] = None,
                 class_cond: bool = True):
        """Constructor

        Args:

          q0: List of base distribution
          flows: List of flows for each level
          merges: List of merge/split operations (forward pass must do merge)
          transform: Initial transformation of inputs
          class_cond: Flag, indicated whether model has class conditional
        base distributions
        """
        super().__init__()
        if len(flows) != len(q0):
            raise ValueError(
                f"MultiscaleFlow: got {len(q0)} base distribution(s) (q0) but "
                f"{len(flows)} level(s) of flows; these must have the same length."
            )
        if len(merges) != len(q0) - 1:
            raise ValueError(
                f"MultiscaleFlow: expected {len(q0) - 1} merge operation(s) for "
                f"{len(q0)} level(s), got {len(merges)}."
            )
        self.q0 = nn.ModuleList(q0)
        self.num_levels = len(self.q0)
        self.flows = nn.ModuleList([nn.ModuleList(flow) for flow in flows])
        self.merges = nn.ModuleList(merges)
        self.transform = transform
        self.class_cond = class_cond
        self._latent_shapes = None
        self._x_shape = None

    def forward_kld(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          y: Batch of classes to condition on, if applicable

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        return -torch.mean(self.log_prob(x, y))

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get negative log-likelihood for maximum likelihood training

        Note:
          Unlike `NormalizingFlow.forward`, calling this model as
          `model(x, y)` returns a scalar-per-batch loss (the negative
          log-likelihood), not a transformed sample. Use `sample()` to
          draw new samples from the model.

        Args:
          x: Batch of data
          y: Batch of classes to condition on, if applicable

        Returns:
            Negative log-likelihood of the batch
        """
        return -self.log_prob(x, y)

    def forward_and_log_det(self, z: List[torch.Tensor], dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        device_type = z[0].device.type if isinstance(z, list) else z.device.type
        
        if dtype is not None:
            ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)
        else:
            ctx = contextlib.nullcontext()
            
        with ctx:
            log_det = torch.zeros(len(z[0]), dtype=torch.float32, device=z[0].device)

            for i in range(self.num_levels):
                if i == 0:
                    z_ = z[0]
                else:
                    z_, log_det_ = self.merges[i - 1]([z_, z[i]])
                    log_det += log_det_.float() if torch.is_tensor(log_det_) else log_det_

                for flow in self.flows[i]:
                    z_, log_det_ = flow(z_)
                    log_det += log_det_.float() if torch.is_tensor(log_det_) else log_det_

            if self.transform is not None:
                z_, log_det_ = self.transform(z_)
                log_det += log_det_.float() if torch.is_tensor(log_det_) else log_det_

        return z_, log_det

    def inverse_and_log_det(self, x: torch.Tensor, dtype=None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if dtype is not None:
            ctx = torch.amp.autocast(device_type=x.device.type, dtype=dtype)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            log_det = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)

            if self.transform is not None:
                x, log_det_ = self.transform.inverse(x)
                log_det += log_det_.float() if torch.is_tensor(log_det_) else log_det_

            z = [None] * self.num_levels

            for i in range(self.num_levels - 1, -1, -1):
                for flow in reversed(self.flows[i]):
                    x, log_det_ = flow.inverse(x)
                    log_det += log_det_.float() if torch.is_tensor(log_det_) else log_det_

                if i == 0:
                    z[i] = x
                else:
                    [x, z[i]], log_det_ = self.merges[i - 1].inverse(x)
                    log_det += log_det_.float() if torch.is_tensor(log_det_) else log_det_

            if self._latent_shapes is None:
                zs = z if isinstance(z, (list, tuple)) else [z]
                self._latent_shapes = [tuple(zi.shape[1:]) for zi in zs] 
                self._x_shape = tuple(x.shape[1:])

        return z, log_det

    @torch.no_grad()
    def sample(self, num_samples: int = 1, y: Optional[torch.Tensor] = None,
               temperature: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw per-level latents with cached shapes and rebuild x via forward().
        Returns: (x, log_prob(x))
        """
        dev = next(self.parameters()).device
        dty = next(self.parameters()).dtype

        if temperature is not None:
            self.set_temperature(temperature)

        # Need shapes observed at least once (training/inference calls inverse/log_prob)
        if self._latent_shapes is None:
            raise RuntimeError(
                "MultiscaleFlow.sample: latent shapes unknown. "
                "Call log_prob/inverse_and_log_det once (e.g., during training) so shapes can be cached."
            )

        latent_shapes = self._latent_shapes  # [(C_i,H_i,W_i), ...]
        bases = list(self.q0)
        if len(bases) == 1 and len(latent_shapes) > 1:
            bases = bases * len(latent_shapes)

        z_list = []
        log_q = torch.zeros(num_samples, device=dev, dtype=dty)

        for lvl, (base, event_shape) in enumerate(zip(bases, latent_shapes)):
            event_shape = tuple(event_shape)             # e.g., (C,H,W) or (C,D,H,W)
            need        = (num_samples, *event_shape)
            flat_event  = np.prod(event_shape)

            if self.class_cond:
                z_i, log_q_i = base(num_samples, y)
            else:
                z_i, log_q_i = base(num_samples)

            ok = (z_i.dim() == len(event_shape)+1 and tuple(z_i.shape[1:]) == event_shape)
            if not ok and (z_i.dim() == 2 and z_i.shape[1] == flat_event):
                z_i = z_i.view(need)
                ok = True

            if not ok:
                raise RuntimeError(
                    f"q0[{lvl}] sample has shape {tuple(z_i.shape[1:])} but expected {event_shape}"
                )

            z_list.append(z_i)
            log_q = log_q + log_q_i.to(device=dev, dtype=dty)


        # Reconstruct x using the graph (merges + flows) at the right spatial sizes
        x, log_det = self.forward_and_log_det(z_list)
        log_q = log_q - log_det

        if temperature is not None:
            self.reset_temperature()
        return x, log_q

    def log_prob(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get log probability for batch

        Args:
          x: Batch
          y: Classes of x. Must be passed in if `class_cond` is True.

        Returns:
          log probability
        """
        log_q = 0
        z = x
        if self.transform is not None:
            z, log_det = self.transform.inverse(z)
            log_q += log_det
        for i in range(self.num_levels - 1, -1, -1):
            for j in range(len(self.flows[i]) - 1, -1, -1):
                z, log_det = self.flows[i][j].inverse(z)
                log_q += log_det
            if i > 0:
                [z, z_], log_det = self.merges[i - 1].inverse(z)
                log_q += log_det
            else:
                z_ = z
            if self.class_cond:
                log_q += self.q0[i].log_prob(z_, y)
            else:
                log_q += self.q0[i].log_prob(z_)
        return log_q

    def save(self, path: str) -> None:
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location="cpu") -> None:
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
          map_location: Device to map the loaded tensors to (see
            `torch.load`). Defaults to "cpu" for portability across
            machines/devices; call `.to(device)` on the model afterwards
            if needed.
        """
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict)

    def set_temperature(self, temperature: Optional[float]) -> None:
        """Set temperature for temperature a annealed sampling

        Args:
          temperature: Temperature parameter
        """
        for q0 in self.q0:
            if hasattr(q0, "temperature"):
                q0.temperature = temperature
            else:
                raise NotImplementedError(
                    "One base function does not "
                    "support temperature annealed sampling"
                )

    def reset_temperature(self) -> None:
        """
        Set temperature values of base distributions back to None
        """
        self.set_temperature(None)


class NormalizingFlowVAE(nn.Module):
    """
    VAE using normalizing flows to express approximate distribution
    """

    def __init__(self, prior: nn.Module, q0: Optional[nn.Module] = None,
                 flows: Optional[Sequence[nn.Module]] = None, decoder: Optional[nn.Module] = None):
        """Constructor of normalizing flow model

        Args:
          prior: Prior distribution of te VAE, i.e. Gaussian
          decoder: Optional decoder
          flows: Flows to transform output of base encoder
          q0: Base Encoder, defaults to a new `distributions.Dirac()` instance
        """
        super().__init__()
        # Note: q0 is instantiated here rather than as a mutable default
        # argument (`q0=distributions.Dirac()`), which would otherwise
        # create a single shared module reused by every instance that
        # doesn't pass its own q0.
        self.prior = prior
        self.decoder = decoder
        self.flows = nn.ModuleList(flows)
        self.q0 = q0 if q0 is not None else distributions.Dirac()

    def forward(self, x: torch.Tensor, num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Takes data batch, samples num_samples for each data point from base distribution

        Args:
          x: data batch
          num_samples: number of samples to draw for each data point

        Returns:
          latent variables for each batch and sample, log_q, and log_p
        """
        z, log_q = self.q0(x, num_samples=num_samples)
        # Flatten batch and sample dim
        z = z.view(-1, *z.size()[2:])
        log_q = log_q.view(-1, *log_q.size()[2:])
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.prior.log_prob(z)
        if self.decoder is not None:
            log_p += self.decoder.log_prob(x, z)
        # Separate batch and sample dimension again
        z = z.view(-1, num_samples, *z.size()[1:])
        log_q = log_q.view(-1, num_samples, *log_q.size()[1:])
        log_p = log_p.view(-1, num_samples, *log_p.size()[1:])
        return z, log_q, log_p
