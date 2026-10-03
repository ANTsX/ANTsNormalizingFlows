"""Tests of the (2+1)D building blocks: per-axis Squeeze3d factors, per-axis ConvNet3d kernels and
GlowBlock3d with factorized kernels / temporal identity initialization.

The historical behavior (2x2x2 squeeze, isotropic 3x3x3, 1x1x1, 3x3x3 kernels) must stay unchanged.
"""
import unittest

import torch

from antsnormflows.flows import GlowBlock3d, Squeeze3d
from antsnormflows.nets import ConvNet3d

FACTORIZED = ((1, 3, 3), (3, 1, 1), (1, 3, 3))
SPATIAL_ONLY = ((1, 3, 3), (1, 1, 1), (1, 3, 3))


def _legacy_squeeze3d_forward(z):
    """The implementation of Squeeze3d.forward before per-axis factors were added."""
    s = z.size()
    z = z.view(s[0], s[1] // 8, 2, 2, 2, s[2], s[3], s[4])
    z = z.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return z.view(s[0], s[1] // 8, s[2] * 2, s[3] * 2, s[4] * 2)


def _legacy_squeeze3d_inverse(z):
    s = z.size()
    z = z.view(s[0], s[1], s[2] // 2, 2, s[3] // 2, 2, s[4] // 2, 2)
    z = z.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    return z.view(s[0], 8 * s[1], s[2] // 2, s[3] // 2, s[4] // 2)


def _perturb_(module, scale=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))


class Squeeze3dFactorsTest(unittest.TestCase):
    def test_default_matches_legacy(self):
        g = torch.Generator().manual_seed(0)
        z = torch.randn(2, 16, 4, 6, 8, generator=g)
        sq = Squeeze3d()
        self.assertEqual(sq.factors, (2, 2, 2))
        self.assertTrue(torch.equal(sq.forward(z)[0], _legacy_squeeze3d_forward(z)))
        self.assertTrue(torch.equal(sq.inverse(z)[0], _legacy_squeeze3d_inverse(z)))

    def test_roundtrip_and_shapes(self):
        g = torch.Generator().manual_seed(1)
        for factors in [(2, 2, 2), (1, 2, 2), (2, 1, 1), (1, 1, 2), (1, 1, 1)]:
            n = factors[0] * factors[1] * factors[2]
            x = torch.randn(2, 3, 4, 8, 8, generator=g)
            sq = Squeeze3d(factors)
            y, ld = sq.inverse(x)                       # data -> latent direction: space to channels
            self.assertEqual(tuple(y.shape), (2, 3 * n, 4 // factors[0], 8 // factors[1], 8 // factors[2]))
            self.assertTrue(torch.all(ld == 0))
            back, ld2 = sq.forward(y)
            self.assertTrue(torch.equal(back, x), factors)
            self.assertTrue(torch.all(ld2 == 0))

    def test_time_axis_untouched_when_factor_is_one(self):
        """With factors (1, 2, 2) the output at time index t only contains input frame t."""
        g = torch.Generator().manual_seed(2)
        x = torch.randn(1, 2, 5, 8, 8, generator=g)
        x2 = x.clone()
        x2[:, :, 3] += 1.0
        y, _ = Squeeze3d((1, 2, 2)).inverse(x)
        y2, _ = Squeeze3d((1, 2, 2)).inverse(x2)
        diff = (y - y2).abs().amax(dim=(0, 1, 3, 4))
        self.assertTrue(diff[3] > 0)
        self.assertTrue(torch.all(diff[[0, 1, 2, 4]] == 0))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            Squeeze3d((0, 2, 2))
        with self.assertRaises(ValueError):
            Squeeze3d((2, 2))
        with self.assertRaises(ValueError):
            Squeeze3d((1, 2, 2)).forward(torch.zeros(1, 6, 2, 2, 2))      # 6 channels, 4 needed
        with self.assertRaises(ValueError):
            Squeeze3d((1, 2, 2)).inverse(torch.zeros(1, 2, 2, 3, 4))      # odd height


class ConvNet3dKernelsTest(unittest.TestCase):
    def test_int_kernels_unchanged(self):
        net = ConvNet3d((4, 8, 8, 8, 6), (3, 1, 3), leaky=0.0)
        convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv3d)]
        self.assertEqual([tuple(c.kernel_size) for c in convs], [(3, 3, 3), (1, 1, 1), (3, 3, 3)])
        self.assertEqual([tuple(c.padding) for c in convs], [(1, 1, 1), (0, 0, 0), (1, 1, 1)])

    def test_per_axis_kernels(self):
        net = ConvNet3d((4, 8, 8, 8, 6), FACTORIZED, leaky=0.0)
        convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv3d)]
        self.assertEqual([tuple(c.kernel_size) for c in convs], [(1, 3, 3), (3, 1, 1), (1, 3, 3)])
        self.assertEqual([tuple(c.padding) for c in convs], [(0, 1, 1), (1, 0, 0), (0, 1, 1)])
        self.assertEqual(tuple(net(torch.zeros(2, 4, 5, 6, 7)).shape), (2, 6, 5, 6, 7))

    def test_json_style_lists_and_even_sizes(self):
        ConvNet3d((4, 8, 8, 8, 6), [[1, 3, 3], [3, 1, 1], [1, 3, 3]])
        with self.assertRaises(ValueError):
            ConvNet3d((4, 8, 8, 8, 6), ((1, 2, 2), 1, 3))
        with self.assertRaises(ValueError):
            ConvNet3d((4, 8, 8, 8, 6), ((1, 3), 1, 3))


class GlowBlock3dFactorizedTest(unittest.TestCase):
    def _block(self, **kw):
        torch.manual_seed(0)
        block = GlowBlock3d(4, 3, **kw).double()
        _perturb_(block, 0.2)
        return block

    def test_default_kernels(self):
        block = GlowBlock3d(4, 3)
        convs = [m for m in block.modules() if isinstance(m, torch.nn.Conv3d)]
        self.assertEqual([tuple(c.kernel_size) for c in convs], [(3, 3, 3), (1, 1, 1), (3, 3, 3)])

    def test_roundtrip_and_logdet_against_autograd(self):
        for kw in ({}, dict(kernel_size=FACTORIZED), dict(kernel_size=FACTORIZED, temporal_init="identity"),
                   dict(kernel_size=SPATIAL_ONLY, net_actnorm=False)):
            block = self._block(**kw)
            g = torch.Generator().manual_seed(3)
            x = torch.randn(1, 4, 2, 3, 3, generator=g, dtype=torch.double)
            with torch.no_grad():
                block(x)                                       # data-dependent ActNorm initialization
            z, ld = block(x)
            back, ld_inv = block.inverse(z)
            self.assertLess(float((back - x).abs().max()), 1e-9, kw)
            self.assertLess(float((ld + ld_inv).abs().max()), 1e-9, kw)

            def f(v):
                return block(v.view(1, 4, 2, 3, 3))[0].reshape(-1)

            jac = torch.autograd.functional.jacobian(f, x.reshape(-1))
            sign, logabsdet = torch.linalg.slogdet(jac)
            self.assertGreater(float(sign), 0.0)
            self.assertLess(abs(float(logabsdet) - float(ld[0])), 1e-7, (kw, float(logabsdet), float(ld[0])))

    def test_temporal_identity_init(self):
        block = GlowBlock3d(8, 6, kernel_size=FACTORIZED, temporal_init="identity")
        convs = [m for m in block.modules() if isinstance(m, torch.nn.Conv3d)]
        mid = convs[1]
        self.assertEqual(tuple(mid.kernel_size), (3, 1, 1))
        w = mid.weight.detach()
        expected = torch.zeros_like(w)
        for c in range(6):
            expected[c, c, 1, 0, 0] = 1.0
        self.assertTrue(torch.equal(w, expected))
        # spatial convs and the zero-initialised last layer are untouched
        self.assertTrue(torch.all(convs[2].weight == 0))
        self.assertGreater(float(convs[0].weight.abs().sum()), 0.0)
        # default initialization leaves the temporal conv alone
        plain = GlowBlock3d(8, 6, kernel_size=FACTORIZED)
        self.assertFalse(torch.equal(w, [m for m in plain.modules() if isinstance(m, torch.nn.Conv3d)][1].weight.detach()))

    def test_frames_are_independent_without_temporal_kernel(self):
        block = self._block(kernel_size=SPATIAL_ONLY)
        g = torch.Generator().manual_seed(4)
        x = torch.randn(2, 4, 5, 4, 4, generator=g, dtype=torch.double)
        with torch.no_grad():
            block(x)                                           # initialize ActNorm
            z1, _ = block(x)
            x2 = x.clone()
            x2[:, :, 2] += 0.5
            z2, _ = block(x2)
        diff = (z1 - z2).abs().amax(dim=(0, 1, 3, 4))
        self.assertGreater(float(diff[2]), 0.0)
        self.assertTrue(torch.all(diff[[0, 1, 3, 4]] == 0))

    def test_temporal_kernel_couples_neighbouring_frames(self):
        block = self._block(kernel_size=FACTORIZED)
        g = torch.Generator().manual_seed(5)
        x = torch.randn(2, 4, 5, 4, 4, generator=g, dtype=torch.double)
        with torch.no_grad():
            block(x)
            z1, _ = block(x)
            x2 = x.clone()
            x2[:, :, 2] += 0.5
            z2, _ = block(x2)
        diff = (z1 - z2).abs().amax(dim=(0, 1, 3, 4))
        self.assertGreater(float(diff[1]) + float(diff[3]), 0.0)     # the (3,1,1) conv reaches +-1 frame
        self.assertTrue(torch.all(diff[[0, 4]] == 0))                # but not +-2 within a single block

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            GlowBlock3d(4, 3, kernel_size=((1, 3, 3), (3, 1, 1)))
        with self.assertRaises(ValueError):
            GlowBlock3d(4, 3, temporal_init="dirac")


if __name__ == "__main__":
    unittest.main()
