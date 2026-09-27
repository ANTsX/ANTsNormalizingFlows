import unittest
import torch

from antsnormflows.flows import CouplingBlock1d
from tests.flows.flow_test import FlowTest


def _perturb(module, std=0.1, seed=0):
    # The last layer of the coupling network is zero-initialized, which makes
    # a fresh block the identity. Perturb all parameters so that the tests
    # exercise a non-trivial transformation.
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(std * torch.randn(p.shape, generator=gen, dtype=p.dtype))


def _jacobian_log_det(flow, x):
    # Brute-force log|det J| of flow.forward at a single sample x of shape (1, C, L).
    shape = x.shape
    f = lambda v: flow(v.view(shape))[0].reshape(-1)
    J = torch.autograd.functional.jacobian(f, x.reshape(-1))
    return torch.linalg.slogdet(J)[1]


class CouplingBlock1dTest(FlowTest):
    def test_coupling_block_1d(self):
        length = 16
        hidden_channels = 16
        for batch_size, channels, scale, split_mode, kernel_size, padding_mode in [
            (2, 4, True, "channel", 3, "zeros"),
            (1, 4, True, "channel_inv", 3, "circular"),
            (3, 8, False, "channel", 5, "zeros"),
            (2, 6, True, "channel_inv", 5, "circular"),
            (2, 5, True, "channel", 3, "circular"),
            (2, 5, True, "channel_inv", 3, "zeros"),
        ]:
            with self.subTest(
                batch_size=batch_size,
                channels=channels,
                scale=scale,
                split_mode=split_mode,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
            ):
                torch.manual_seed(0)
                inputs = torch.randn(batch_size, channels, length, dtype=torch.float64)
                flow = CouplingBlock1d(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    scale=scale,
                    split_mode=split_mode,
                    s_cap=1.0,
                    padding_mode=padding_mode,
                ).double()
                _perturb(flow)

                # Non-trivial transform
                outputs, log_det = flow(inputs)
                self.assertGreater((outputs - inputs).abs().max().item(), 1e-3)

                self.checkForwardInverse(flow, inputs, atol=1e-8, rtol=1e-8)

                # log-det matches the brute-force Jacobian
                x1 = inputs[:1]
                _, ld = flow(x1)
                self.assertClose(ld[0], _jacobian_log_det(flow, x1), atol=1e-8, rtol=1e-8)

    def test_circular_padding_equivariance(self):
        # With circular padding the block commutes with a circular time shift,
        # i.e. the gait-cycle start has no special role.
        torch.manual_seed(0)
        channels, length = 6, 16
        x = torch.randn(3, channels, length, dtype=torch.float64)
        for padding_mode, equivariant in [("circular", True), ("zeros", False)]:
            with self.subTest(padding_mode=padding_mode):
                flow = CouplingBlock1d(
                    channels=channels, hidden_channels=8, s_cap=1.0,
                    padding_mode=padding_mode,
                ).double()
                _perturb(flow, std=0.3)
                y_shift = flow(torch.roll(x, 5, dims=2))[0]
                shift_y = torch.roll(flow(x)[0], 5, dims=2)
                self.assertEqual(torch.allclose(y_shift, shift_y, atol=1e-10), equivariant)


if __name__ == "__main__":
    unittest.main()
