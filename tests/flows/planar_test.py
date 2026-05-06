import unittest
import torch

from antsnormflows.flows import Planar
from tests.flows.flow_test import FlowTest

class PlanarTest(FlowTest):
    def test_planar(self):
        batch_size = 3
        for latent_size in [(2,), (5,), (2, 3, 4)]:
            for act in ["tanh", "leaky_relu"]:
                with self.subTest(latent_size=latent_size, act=act):
                    flow = Planar(latent_size, act=act)
                    inputs = torch.randn((batch_size, *latent_size))
                    if act == "leaky_relu":
                        self.checkForwardInverse(flow, inputs)
                    else:
                        self.checkForward(flow, inputs)


if __name__ == "__main__":
    unittest.main()