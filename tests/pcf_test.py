import unittest

import torch
from torch.testing import assert_close

from antsnormflows import MultiscaleFlow, NormalizingFlow, ConditionalNormalizingFlow, ConvFlow1d
from antsnormflows.flows import GlowBlock2d, Merge, Squeeze2d, Invertible1x1Conv, MaskedAffineFlow, \
    MaskedAffineAutoregressive, BatchNorm
from antsnormflows.nets import MLP
from antsnormflows.distributions.base import DiagGaussian
from antsnormflows.utils import latent_stretch, pcf_penalty, pcf_penalty_from_latents


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


def _real_nvp(dim=4, n_layers=4):
    layers = []
    for i in range(n_layers):
        b = torch.tensor([float(j % 2 == i % 2) for j in range(dim)])
        layers.append(MaskedAffineFlow(b, MLP([dim, 16, dim], init_zeros=False), MLP([dim, 16, dim], init_zeros=False)))
    return NormalizingFlow(DiagGaussian(dim), layers)


def _jacobian_penalty(encode, x):
    """-I_hat from the full Jacobian of encode (one sample -> flat latent)."""
    out = []
    for xi in x:
        J = torch.autograd.functional.jacobian(encode, xi).reshape(-1, xi.numel())
        out.append(J.norm(dim=1).log().sum() - torch.linalg.slogdet(J)[1])
    return torch.stack(out)


class GenericPCFPenaltyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_normalizing_flow_exact(self):
        model = _real_nvp().double().eval()
        x = torch.randn(5, 4, dtype=torch.float64)
        penalty = pcf_penalty(model, x, num_dims=4)
        expected = _jacobian_penalty(lambda v: model.inverse_and_log_det(v.unsqueeze(0))[0].reshape(-1), x)
        assert_close(penalty, expected, rtol=1e-6, atol=1e-6)
        assert (penalty >= -1e-8).all()

    def test_conditional_flow_exact_at_fixed_context(self):
        layers = [MaskedAffineAutoregressive(features=3, hidden_features=16, context_features=2, num_blocks=1,
                                             random_mask=False) for _ in range(2)]
        model = ConditionalNormalizingFlow(DiagGaussian(3), layers).double().eval()
        x = torch.randn(4, 3, dtype=torch.float64)
        c = torch.randn(4, 2, dtype=torch.float64)
        penalty = pcf_penalty(model, x, num_dims=3, context=c)
        expected = torch.cat([
            _jacobian_penalty(lambda v, ci=c[i:i + 1]: model.inverse_and_log_det(v.unsqueeze(0), context=ci)[0].reshape(-1),
                              x[i:i + 1]) for i in range(4)])
        assert_close(penalty, expected, rtol=1e-6, atol=1e-6)

    def test_conv_flow_1d_exact_and_unbiased(self):
        model = ConvFlow1d(channels=2, length=8, K=2, hidden_channels=8).double()
        model.inverse_and_log_det(torch.randn(8, 2, 8, dtype=torch.float64))   # ActNorm initialization
        with torch.no_grad():                                                  # leave the near-identity init
            for p in model.parameters():
                p.add_(0.3 * torch.randn_like(p))
        model.eval()
        x = torch.randn(2, 2, 8, dtype=torch.float64)
        exact = pcf_penalty(model, x, num_dims=10 ** 6)
        assert (exact > 1e-6).all()                                            # non-trivial penalty
        expected = _jacobian_penalty(lambda v: model.inverse_and_log_det(v.unsqueeze(0))[0].reshape(-1), x)
        assert_close(exact, expected, rtol=1e-6, atol=1e-6)
        draws = torch.stack([pcf_penalty(model, x[:1], num_dims=2) for _ in range(3000)]).squeeze(1)
        assert abs(draws.mean() - exact[0]) < 4 * draws.std() / draws.numel() ** 0.5

    def test_multiscale_method_delegates(self):
        model = _glow2d().double().eval()
        x = torch.randn(2, 2, 4, 4, dtype=torch.float64)
        assert_close(model.pcf_regularizer(x, num_dims=10 ** 6), pcf_penalty(model, x, num_dims=10 ** 6))

    def test_batch_statistics_rejected_in_training(self):
        dim = 4
        coupled = NormalizingFlow(DiagGaussian(dim), [BatchNorm(detach_stats=False), _real_nvp(dim, 2).flows[0]])
        with self.assertRaises(ValueError):
            pcf_penalty(coupled.train(), torch.randn(6, dim))
        pcf_penalty(coupled.eval(), torch.randn(6, dim))                     # eval mode: running statistics
        detached = NormalizingFlow(DiagGaussian(dim), [BatchNorm(detach_stats=True), _real_nvp(dim, 2).flows[0]])
        pcf_penalty(detached.train(), torch.randn(6, dim))                   # constants for autograd: allowed
        mlp_bn = _real_nvp(dim, 2)
        mlp_bn.flows[0].s.net.insert(1, torch.nn.BatchNorm1d(16))
        with self.assertRaises(ValueError):
            pcf_penalty(mlp_bn.train(), torch.randn(6, dim))

    def test_training_gradient_through_penalty(self):
        model = _real_nvp().train()
        x = torch.randn(16, 4)
        loss = (-model.log_prob(x) + 0.5 * pcf_penalty(model, x, num_dims=4)).mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


    def test_from_latents_matches_and_requires_grad(self):
        model = _glow2d().double().eval()
        x = torch.randn(2, 2, 4, 4, dtype=torch.float64).requires_grad_(True)
        z, log_det = model.inverse_and_log_det(x)
        assert_close(pcf_penalty_from_latents(x, z, log_det, num_dims=10 ** 6),
                     pcf_penalty(model, x, num_dims=10 ** 6))
        x0 = torch.randn(2, 2, 4, 4, dtype=torch.float64)
        z0, ld0 = model.inverse_and_log_det(x0)
        with self.assertRaises(ValueError):
            pcf_penalty_from_latents(x0, z0, ld0)


    def test_latent_stretch_matches_jacobian_rows(self):
        model = _glow2d().double().eval()
        x = torch.randn(2, 2, 4, 4, dtype=torch.float64)
        z, stretch = latent_stretch(model, x, num_probes=6000)
        assert [s.shape for s in stretch] == [zl.reshape(2, -1).shape for zl in z]
        for i in range(2):
            J = torch.autograd.functional.jacobian(
                lambda v: torch.cat([zl.reshape(-1) for zl in model.inverse_and_log_det(v.unsqueeze(0))[0]]), x[i])
            exact = 1.0 / J.reshape(-1, x[i].numel()).norm(dim=1)
            est = torch.cat([s[i] for s in stretch])
            assert ((est / exact - 1).abs().median() < 0.03) and ((est / exact - 1).abs().max() < 0.15)

    def test_latent_stretch_on_conditional_flow(self):
        layers = [MaskedAffineAutoregressive(features=3, hidden_features=16, context_features=2, num_blocks=1,
                                             random_mask=False) for _ in range(2)]
        model = ConditionalNormalizingFlow(DiagGaussian(3), layers).double().eval()
        z, s = latent_stretch(model, torch.randn(4, 3, dtype=torch.float64), num_probes=8,
                              context=torch.randn(4, 2, dtype=torch.float64))
        assert len(z) == 1 and s[0].shape == (4, 3) and (s[0] > 0).all()


    def test_level_sampling_is_unbiased(self):
        model = _glow2d().double().eval()
        x = torch.randn(1, 2, 4, 4, dtype=torch.float64)
        exact = pcf_penalty(model, x, num_dims=10 ** 6)
        draws = torch.stack([pcf_penalty(model, x, num_dims=10 ** 6, num_levels=1) for _ in range(2000)]).squeeze(1)
        assert abs(draws.mean() - exact[0]) < 4 * draws.std() / draws.numel() ** 0.5
        assert_close(pcf_penalty(model, x, num_dims=10 ** 6, num_levels=5), exact)   # >= levels: all


if __name__ == "__main__":
    unittest.main()
