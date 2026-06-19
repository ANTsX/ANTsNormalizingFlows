import time
import unittest
import torch
from torch import nn

from antsnormflows.utils.splines import search_sorted
from antsnormflows.flows.affine.glow import GlowBlock3d
from antsnormflows.core import ConditionalNormalizingFlow
from antsnormflows.distributions.base import DiagGaussian, ConditionalDiagGaussian
from antsnormflows.flows.neural_spline.wrapper import AutoregressiveRationalQuadraticSpline

class TestSearchsortedCorrectness(unittest.TestCase):
    def test_binary_search_correctness(self):
        """Vérifie que search_sorted retourne les bons indices (Fix GB-4)."""
        bin_locations = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
        inputs = torch.tensor([[0.0, 0.1, 0.3, 0.6, 0.99]])
        expected = torch.tensor([[0, 0, 1, 2, 3]])
        
        result = search_sorted(bin_locations, inputs)
        torch.testing.assert_close(result, expected)
    
    def test_edge_cases(self):
        """Test des cas limites (valeurs exactement sur les bornes)."""
        bin_locations = torch.tensor([[0.0, 0.5, 1.0]])
        inputs = torch.tensor([[1.0]])
        result = search_sorted(bin_locations, inputs)
        self.assertEqual(result.item(), 1)


class TestNeuroImagingIntegration(unittest.TestCase):
    """Tests spécifiques aux cas d'usage neuro-imagerie 3D."""
    
    def test_glow_block_3d_forward_inverse(self):
        """Teste GlowBlock3d sur un volume IRM synthétique miniature."""
        torch.manual_seed(42)
        C, D, H, W = 2, 16, 16, 16 
        
        block = GlowBlock3d(
            channels=C,
            hidden_channels=8,
            split_mode='channel',
            scale=True
        )
        
        x = torch.randn(2, C, D, H, W)
        x_fwd, log_det_fwd = block(x)
        x_rec, log_det_inv = block.inverse(x_fwd)
        
        # Tolérance augmentée pour le float32
        torch.testing.assert_close(x_rec, x, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(log_det_fwd + log_det_inv,
                                   torch.zeros_like(log_det_fwd), atol=1e-4, rtol=1e-4)
    
    def test_conditional_flow_on_morphometric_features(self):
        """Simule un conditionnement sur l'âge/sexe du patient."""
        torch.manual_seed(42)
        n_dims = 10         
        n_context = 2       
        n_samples = 50
        
        layer = AutoregressiveRationalQuadraticSpline(
            num_input_channels=n_dims,
            num_blocks=2,
            num_hidden_channels=32,
            num_context_channels=n_context
        )
        base = DiagGaussian(n_dims)
        
        # Ajout de la forme et de l'encodeur de contexte (mapping N -> 2 * D)
        context_encoder = torch.nn.Linear(n_context, 2 * n_dims)
        target = ConditionalDiagGaussian(shape=(n_dims,), context_encoder=context_encoder)
        
        model = ConditionalNormalizingFlow(base, [layer], target)
        
        x = torch.randn(n_samples, n_dims)          
        context = torch.randn(n_samples, n_context) 
        
        log_p = model.log_prob(x, context=context)
        
        self.assertEqual(log_p.shape, (n_samples,))
        self.assertFalse(torch.isnan(log_p).any())
        self.assertFalse(torch.isinf(log_p).any())

class TestPerformanceRegression(unittest.TestCase):
    """Tests de non-régression de performance sur GPU."""
    
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Tests de performance ignorés : GPU requis")
        cls.device = torch.device("cuda:0")
    
    def test_nsf_memory_footprint(self):
        """Vérifie la croissance linéaire de la mémoire avec le batch size."""
        n_dims = 8
        layer = AutoregressiveRationalQuadraticSpline(
            num_input_channels=n_dims, num_blocks=2, num_hidden_channels=32
        ).to(self.device)
        
        # Test petit batch
        torch.cuda.reset_peak_memory_stats()
        z_small = torch.randn(64, n_dims, device=self.device)
        _ = layer(z_small)
        mem_small = torch.cuda.max_memory_allocated() / 1e6
        
        # Test grand batch
        torch.cuda.reset_peak_memory_stats()
        z_large = torch.randn(256, n_dims, device=self.device)
        _ = layer(z_large)
        mem_large = torch.cuda.max_memory_allocated() / 1e6
        
        memory_ratio = mem_large / mem_small if mem_small > 0 else 1.0
        batch_ratio = 256 / 64  # = 4.0
        
        # Vérifie que la mémoire ne croît pas de manière quadratique
        self.assertLess(memory_ratio, batch_ratio * 1.5)