import unittest
import torch

from antsnormflows.flows import GlowBlock1d
from tests.flows.flow_test import FlowTest
from tests.flows.affine.coupling_block_1d_test import _perturb, _jacobian_log_det


class Glow1dTest(FlowTest):
    def test_glow_1d(self):
        length = 16
        hidden_channels = 16
        for batch_size, channels, scale, split_mode, use_lu, net_actnorm, kernel_size, padding_mode in [
            (2, 4, True, "channel", True, False, (3, 1, 3), "zeros"),
            (1, 4, True, "channel_inv", True, True, (3, 1, 3), "circular"),
            (2, 6, True, "channel", False, False, 5, "circular"),
            (1, 4, False, "channel_inv", True, False, (5, 1, 5), "zeros"),
            (2, 5, True, "channel", True, False, 3, "circular"),
        ]:
            with self.subTest(
                batch_size=batch_size,
                channels=channels,
                scale=scale,
                split_mode=split_mode,
                use_lu=use_lu,
                net_actnorm=net_actnorm,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
            ):
                torch.manual_seed(0)
                inputs = torch.randn(batch_size, channels, length, dtype=torch.float64)
                flow = GlowBlock1d(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    scale=scale,
                    split_mode=split_mode,
                    use_lu=use_lu,
                    net_actnorm=net_actnorm,
                    s_cap=1.0,
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                ).double()
                # Trigger the data-dependent ActNorm init, then perturb all
                # parameters (the coupling's last layer starts at zero).
                flow(inputs)
                _perturb(flow, std=0.05)

                outputs, _ = flow(inputs)
                self.assertGreater((outputs - inputs).abs().max().item(), 1e-3)

                self.checkForwardInverse(flow, inputs, atol=1e-8, rtol=1e-8)

                x1 = inputs[:1]
                _, ld = flow(x1)
                self.assertClose(ld[0], _jacobian_log_det(flow, x1), atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
