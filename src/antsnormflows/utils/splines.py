import torch
from torch.nn import functional as F
import math
from typing import Tuple

import numpy as np

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


import os

# Ne compiler que si on n'est pas dans l'environnement CI
if os.environ.get("CI") == "true":
    def conditional_compile(fn):
        return fn
else:
    conditional_compile = torch.compile

@conditional_compile
def search_sorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = bin_locations.shape[-1]
    adjust = torch.zeros(dims, device=bin_locations.device, dtype=bin_locations.dtype)
    adjust[-1] = eps
    bl = bin_locations + adjust
    return torch.sum(inputs[..., None] >= bl, dim=-1) - 1

@conditional_compile
def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # 1. Traitement des limites (tails) sans indexation booléenne dynamique
    if tails == "linear":
        unnormalized_derivatives_ = F.pad(unnormalized_derivatives, pad=[1, 1])
        constant = math.log(math.exp(1.0 - min_derivative) - 1.0)
        unnormalized_derivatives_[..., 0] = constant
        unnormalized_derivatives_[..., -1] = constant
    elif tails == "circular":
        unnormalized_derivatives_ = F.pad(unnormalized_derivatives, pad=[0, 1])
        unnormalized_derivatives_[..., -1] = unnormalized_derivatives_[..., 0]
    elif isinstance(tails, (list, tuple)):
        unnormalized_derivatives_ = unnormalized_derivatives.clone()
        constant = math.log(math.exp(1.0 - min_derivative) - 1.0)
        # Boucle Python simple : parfaitement supportée par torch.compile
        for i, t in enumerate(tails):
            if t == "linear":
                unnormalized_derivatives_[..., i, 0] = constant
                unnormalized_derivatives_[..., i, -1] = constant
            elif t == "circular":
                unnormalized_derivatives_[..., i, -1] = unnormalized_derivatives_[..., i, 0]
    else:
        raise RuntimeError(f"{tails} tails are not implemented.")

    # 2. Remplacement du masque par un "Clamp" pour éviter les formes dynamiques
    if torch.is_tensor(tail_bound):
        tail_bound_ = torch.broadcast_to(tail_bound, inputs.shape)
        left = -tail_bound_
        right = tail_bound_
        bottom = -tail_bound_
        top = tail_bound_
        # Équivalent de torch.clamp pour des tenseurs
        inputs_clamped = torch.max(torch.min(inputs, tail_bound_), -tail_bound_)
    else:
        left = -tail_bound
        right = tail_bound
        bottom = -tail_bound
        top = tail_bound
        inputs_clamped = torch.clamp(inputs, min=-tail_bound, max=tail_bound)

    # 3. Exécution de la spline sur l'ensemble du tenseur
    outputs_spline, logabsdet_spline = rational_quadratic_spline(
        inputs=inputs_clamped,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives_,
        inverse=inverse,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    # 4. Fusion propre avec torch.where (élimine l'avertissement IndexPutBackward0)
    outputs = torch.where(inside_interval_mask, outputs_spline.to(inputs.dtype), inputs)
    logabsdet = torch.where(inside_interval_mask, logabsdet_spline.to(inputs.dtype), torch.zeros_like(inputs))

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    num_bins = unnormalized_widths.shape[-1]

    if torch.is_tensor(left):
        lim_tensor = True
    else:
        lim_tensor = False

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    if lim_tensor:
        cumwidths = (right[..., None] - left[..., None]) * cumwidths + left[..., None]
    else:
        cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    if lim_tensor:
        cumheights = (top[..., None] - bottom[..., None]) * cumheights + bottom[..., None]
    else:
        cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = search_sorted(cumheights, inputs)[..., None]
    else:
        bin_idx = search_sorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet
