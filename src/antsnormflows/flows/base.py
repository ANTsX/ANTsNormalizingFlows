import torch
import torch.nn as nn


class Flow(nn.Module):
    """
    Generic class for flow functions
    """

    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        Args:
          z: input variable, first dimension is batch dim

        Returns:
          transformed z and log of absolute determinant
        """
        raise NotImplementedError("Forward pass has not been implemented.")

    def inverse(self, z):
        raise NotImplementedError("This flow has no algebraic inverse.")


class Reverse(Flow):
    """
    Switches the forward transform of a flow layer with its inverse and vice versa
    """

    def __init__(self, flow):
        """Constructor

        Args:
          flow: Flow layer to be reversed
        """
        super().__init__()
        self.flow = flow

    def forward(self, z):
        return self.flow.inverse(z)

    def inverse(self, z):
        return self.flow.forward(z)


class Composite(Flow):
    """
    Composes several flows into one, in the order they are given.
    """

    def __init__(self, flows):
        """Constructor

        Args:
          flows: Iterable of flows to composite
        """
        super().__init__()
        self._flows = nn.ModuleList(flows)

    @staticmethod
    def _cascade(inputs, funcs):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = torch.zeros(batch_size, dtype=inputs.dtype, device=inputs.device)
        for func in funcs:
            outputs, logabsdet = func(outputs)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs):
        funcs = self._flows
        return self._cascade(inputs, funcs)

    def inverse(self, inputs):
        funcs = (flow.inverse for flow in self._flows[::-1])
        return self._cascade(inputs, funcs)


def zero_log_det_like_z(z):
    return torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)


def stable_fp_dtype(x: torch.Tensor) -> torch.dtype:
    """
    Return a numerically stable floating-point dtype for exp/log-scale
    computations: at least float32, but preserves float64 when the input is
    already double precision.

    Several flow layers (ActNorm, AffineCoupling, Invertible1x1Conv /
    Invertible1x1x1Conv, and MultiscaleFlow's log_det accumulators) upcast
    scale/log-scale tensors with a hardcoded ``.float()`` before calling
    ``exp``/``log``. That upcast is correct and intentional when running
    under fp16/bf16 autocast (it guarantees at least float32 precision for
    numerically sensitive ops), but it silently *downgrades* precision when
    the surrounding computation is already float64 -- e.g. a model cast to
    ``.double()`` for a high-precision invertibility check would still
    accumulate log-determinants (or scale factors) at float32 precision,
    making such a check unable to distinguish genuine bugs from ordinary
    float32 roundoff. Use ``x.to(stable_fp_dtype(x))`` in place of
    ``x.float()`` wherever the goal is "at least fp32", not "always fp32".
    """
    return torch.float64 if x.dtype == torch.float64 else torch.float32