"""Principal component flow (PCF) penalty for any bijective flow.

Cunningham, Cobb and Jha, "Principal Component Flows", ICML 2022.
"""
from typing import Optional

import torch
import torch.nn as nn


def _batch_coupling_modules(model: nn.Module):
    """Modules that couple the samples of a batch while training (batch statistics)."""
    from ..flows.normalization import BatchNorm as FlowBatchNorm
    found = []
    for name, m in model.named_modules():
        if not m.training:
            continue
        # The flow BatchNorm with detach_stats=True treats the batch statistics as
        # constants for autograd, so each sample's Jacobian rows stay its own.
        if isinstance(m, FlowBatchNorm) and not getattr(m, "detach_stats", False):
            found.append(name or type(m).__name__)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            found.append(name or type(m).__name__)
    return found


def pcf_penalty(model: nn.Module, x: torch.Tensor, num_dims: int = 1,
                context: Optional[torch.Tensor] = None,
                create_graph: Optional[bool] = None) -> torch.Tensor:
    """Principal component flow penalty -I_hat(x) >= 0, estimated stochastically.

    With the latent space partitioned into single coordinates, the surrogate of
    the paper is

        I_hat(x) = log|det dz/dx| - sum_k log ||d z_k / dx|| <= 0,

    zero exactly when the rows of the encoder Jacobian dz/dx are orthogonal
    (Hadamard's inequality), i.e. when the flow's contours are its principal
    manifolds. Add it to the negative log-likelihood with a weight alpha:
    loss = -log p(x) + alpha * pcf_penalty(model, x).

    Works for any flow whose ``inverse_and_log_det`` maps data to latents and
    returns (z, log|det dz/dx|): ``NormalizingFlow``, ``ConditionalNormalizingFlow``
    (pass ``context``; the Jacobian is taken with respect to x at fixed context),
    ``MultiscaleFlow`` and ``ConvFlow1d``. A list of latents (multiscale levels) is
    sampled level by level.

    The sum over latent coordinates is estimated without bias by stratified
    sampling: ``num_dims`` coordinates are drawn uniformly within each level
    (independently for each batch element), and each level's contribution is
    scaled by (level size / num_dims). Each draw costs one vector-Jacobian
    product through the encoder. When ``num_dims`` is at least the size of a
    level, that level's sum is computed exactly (no variance).

    One vector-Jacobian product serves the whole batch, which requires the flow
    to treat samples independently: layers that use batch statistics while
    training (BatchNorm) are rejected.

    Args:
      model: Flow with ``inverse_and_log_det``
      x: Batch of data
      num_dims: Latent coordinates sampled per level and per batch element
      context: Conditioning tensor for conditional flows
      create_graph: Keep the graph of the vector-Jacobian products so that the
        penalty can be differentiated with respect to the parameters (double
        backward). Defaults to ``model.training``.

    Returns:
      Penalty -I_hat for each batch element (shape (batch,)), >= 0 up to Monte
      Carlo error
    """
    if num_dims < 1:
        raise ValueError(f"pcf_penalty: num_dims must be >= 1, got {num_dims}.")
    coupled = _batch_coupling_modules(model)
    if coupled:
        raise ValueError(
            "pcf_penalty: layers using batch statistics in training mode couple the samples of a "
            f"batch, so the per-sample Jacobian rows would be wrong: {coupled}. Put them in eval "
            "mode or use ActNorm.")
    if create_graph is None:
        create_graph = model.training
    x = x.detach().requires_grad_(True)
    if context is None:
        z, log_det = model.inverse_and_log_det(x)
    else:
        z, log_det = model.inverse_and_log_det(x, context=context)
    return pcf_penalty_from_latents(x, z, log_det, num_dims=num_dims, create_graph=create_graph)


def pcf_penalty_from_latents(x: torch.Tensor, z, log_det: torch.Tensor, num_dims: int = 1,
                             create_graph: bool = True) -> torch.Tensor:
    """PCF penalty from an encoding that has already been computed.

    Same estimator as `pcf_penalty`, for callers that already ran
    ``z, log_det = model.inverse_and_log_det(x)`` with ``x.requires_grad`` set
    (e.g. inside a training forward pass, so that the likelihood and the
    penalty share one encoder pass, or inside a DistributedDataParallel
    forward so that gradient synchronization covers the penalty). The flow
    must treat the samples of the batch independently (see `pcf_penalty`).

    Args:
      x: Input batch, the leaf (requires_grad=True) the encoding was computed from
      z: Latent tensor, or list of latent tensors (one per level)
      log_det: log|det dz/dx| for each batch element
      num_dims: Latent coordinates sampled per level and per batch element
      create_graph: Keep the graph for differentiation with respect to the parameters

    Returns:
      Penalty -I_hat for each batch element (shape (batch,))
    """
    from ..flows.base import stable_fp_dtype
    if num_dims < 1:
        raise ValueError(f"pcf_penalty_from_latents: num_dims must be >= 1, got {num_dims}.")
    if not x.requires_grad:
        raise ValueError("pcf_penalty_from_latents: x must require grad and be the input z was computed from.")
    levels = z if isinstance(z, (list, tuple)) else [z]
    batch_size = x.shape[0]
    accum_dtype = stable_fp_dtype(x)
    rows = torch.arange(batch_size, device=x.device)
    total = torch.zeros(batch_size, dtype=accum_dtype, device=x.device)
    for z_level in levels:
        z_flat = z_level.reshape(batch_size, -1)
        size = z_flat.shape[1]
        exact = num_dims >= size
        n_draws = size if exact else num_dims
        weight = 1.0 if exact else size / num_dims
        for draw in range(n_draws):
            if exact:
                idx = torch.full((batch_size,), draw, device=x.device, dtype=torch.long)
            else:
                idx = torch.randint(size, (batch_size,), device=x.device)
            selector = torch.zeros_like(z_flat)
            selector[rows, idx] = 1.0
            row = torch.autograd.grad(z_flat, x, grad_outputs=selector,
                                      create_graph=create_graph, retain_graph=True)[0]
            row_norm = row.reshape(batch_size, -1).to(accum_dtype).norm(dim=1)
            total = total + weight * torch.log(row_norm.clamp_min(1e-30))
    return total - log_det.to(accum_dtype)


def latent_stretch(model: nn.Module, x: torch.Tensor, num_probes: int = 64,
                   context: Optional[torch.Tensor] = None, eps: float = 1e-12):
    """Image-space stretch of every latent coordinate, by Hutchinson diagonal estimation.

    For the encoder Jacobian G = dz/dx with rows g_k = dz_k/dx, a random
    Rademacher probe v in data space gives E[(G v)_k^2] = ||g_k||^2 for every k
    at once. Averaging over ``num_probes`` probes estimates all row norms with
    one Jacobian-vector product per probe (computed as a double vector-Jacobian
    product, so any flow with ``inverse_and_log_det`` works).

    When the rows are orthogonal (the goal of the principal component flow
    penalty), the decoder column dx/dz_k equals g_k / ||g_k||^2, so its norm,
    the image-space stretch of coordinate k, is 1 / ||g_k||. Ranking latent
    coordinates by their stretch averaged over a population gives the flow's
    principal ordering (the latent variances themselves are ~1 after whitening
    and carry no ordering). Without the penalty the stretch is only an
    approximation of the column norm.

    Args:
      model: Flow with ``inverse_and_log_det``
      x: Batch of data
      num_probes: Number of Rademacher probes (relative error ~ sqrt(2 / num_probes))
      context: Conditioning tensor for conditional flows
      eps: Floor on the squared row norm

    Returns:
      (z, stretch): the latents (list, one tensor per level, detached) and, for
      each level, a tensor (batch, level size) of estimated stretches 1/||g_k||
    """
    if num_probes < 1:
        raise ValueError(f"latent_stretch: num_probes must be >= 1, got {num_probes}.")
    coupled = _batch_coupling_modules(model)
    if coupled:
        raise ValueError(f"latent_stretch: layers using batch statistics in training mode: {coupled}.")
    x = x.detach().requires_grad_(True)
    if context is None:
        z, _ = model.inverse_and_log_det(x)
    else:
        z, _ = model.inverse_and_log_det(x, context=context)
    levels = z if isinstance(z, (list, tuple)) else [z]
    batch_size = x.shape[0]
    sizes = [zl[0].numel() for zl in levels]
    z_flat = torch.cat([zl.reshape(batch_size, -1) for zl in levels], dim=1)
    u = torch.zeros_like(z_flat, requires_grad=True)
    gt_u = torch.autograd.grad(z_flat, x, grad_outputs=u, create_graph=True)[0]   # G^T u, linear in u
    sq = torch.zeros_like(z_flat, dtype=torch.float64)
    for _ in range(num_probes):
        v = torch.randint(0, 2, x.shape, device=x.device).to(x.dtype) * 2 - 1
        gv = torch.autograd.grad(gt_u, u, grad_outputs=v, retain_graph=True)[0]  # G v
        sq += gv.detach().double() ** 2
    stretch = (sq / num_probes).clamp_min(eps).rsqrt().to(x.dtype)
    out_z, out_s, start = [], [], 0
    for zl, n in zip(levels, sizes):
        out_z.append(zl.detach())
        out_s.append(stretch[:, start:start + n])
        start += n
    return out_z, out_s
