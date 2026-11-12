import unittest

import torch
from torch.testing import assert_close

from antsnormflows import MultiscaleFlow
from antsnormflows.flows import GlowBlock2d, Merge, Squeeze2d, Invertible1x1Conv
from antsnormflows.distributions.base import DiagGaussian


def _glow2d(num_levels=2, num_blocks=2, input_shape=(2, 4, 4), hidden=8, grad_checkpoint=None):
    """Small 2-D multiscale Glow, built as in core_test.test_multiscale_flow (no class conditioning)."""
    flows, merges, base = [], [], []
    for i in range(num_levels):
        flows_ = [GlowBlock2d(input_shape[0] * 2 ** (num_levels + 1 - i), hidden, split_mode="channel", scale=True)
                  for _ in range(num_blocks)]
        flows_ += [Squeeze2d()]
        flows += [flows_]
        if i > 0:
            merges += [Merge()]
            latent_shape = (input_shape[0] * 2 ** (num_levels - i), input_shape[1] // 2 ** (num_levels - i),
                            input_shape[2] // 2 ** (num_levels - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (num_levels + 1), input_shape[1] // 2 ** num_levels,
                            input_shape[2] // 2 ** num_levels)
        base += [DiagGaussian(latent_shape)]
    model = MultiscaleFlow(base, flows, merges, class_cond=False, grad_checkpoint=grad_checkpoint)
    model.inverse_and_log_det(torch.randn(8, *input_shape))   # data-dependent ActNorm initialization
    return model


def _penalty_by_jacobian(model, x):
    """-I_hat from the full encoder Jacobian: sum_k log ||row_k|| - log|det J|."""
    out = []
    for xi in x:
        def enc(v):
            z, _ = model.inverse_and_log_det(v.unsqueeze(0))
            return torch.cat([zl.reshape(-1) for zl in z])
        J = torch.autograd.functional.jacobian(enc, xi).reshape(-1, xi.numel())
        out.append(J.norm(dim=1).log().sum() - torch.linalg.slogdet(J)[1])
    return torch.stack(out)


class PCFRegularizerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_exact_mode_matches_full_jacobian(self):
        model = _glow2d().double().eval()
        x = torch.randn(3, 2, 4, 4, dtype=torch.float64)
        penalty = model.pcf_regularizer(x, num_dims=10 ** 6)   # >= every level size: exact
        assert penalty.shape == (3,)
        assert_close(penalty, _penalty_by_jacobian(model, x), rtol=1e-6, atol=1e-6)
        assert (penalty >= -1e-8).all()                       # Hadamard: -I_hat >= 0

    def test_stochastic_estimate_is_unbiased(self):
        model = _glow2d().double().eval()
        x = torch.randn(1, 2, 4, 4, dtype=torch.float64)
        exact = model.pcf_regularizer(x, num_dims=10 ** 6)
        draws = torch.stack([model.pcf_regularizer(x, num_dims=2) for _ in range(3000)]).squeeze(1)
        se = draws.std() / draws.numel() ** 0.5
        assert abs(draws.mean() - exact[0]) < 4 * se

    def test_zero_for_orthogonal_encoder(self):
        # A single 1x1 convolution with an orthogonal weight has an orthogonal (block-diagonal) Jacobian.
        conv = Invertible1x1Conv(4, use_lu=False)
        with torch.no_grad():
            q, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
            conv.W.copy_(q.to(conv.W.dtype))
        model = MultiscaleFlow([DiagGaussian((4, 3, 3))], [[conv]], [], class_cond=False).double().eval()
        x = torch.randn(2, 4, 3, 3, dtype=torch.float64)
        assert_close(model.pcf_regularizer(x, num_dims=10 ** 6), torch.zeros(2, dtype=torch.float64),
                     rtol=0, atol=1e-8)

    def test_penalty_is_differentiable(self):
        for grad_checkpoint in (None, True):
            with self.subTest(grad_checkpoint=grad_checkpoint):
                model = _glow2d(grad_checkpoint=grad_checkpoint).train()
                x = torch.randn(4, 2, 4, 4)
                loss = (-model.log_prob(x) + 0.1 * model.pcf_regularizer(x, num_dims=2)).mean()
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert len(grads) > 0
                assert all(torch.isfinite(g).all() for g in grads)
                assert any(g.abs().sum() > 0 for g in grads)

    def test_invalid_num_dims(self):
        model = _glow2d().eval()
        with self.assertRaises(ValueError):
            model.pcf_regularizer(torch.randn(2, 2, 4, 4), num_dims=0)


if __name__ == "__main__":
    unittest.main()
