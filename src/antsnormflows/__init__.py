from ._version import __version__  # light import

# Avoid importing torch-consuming modules at build time
try:
    from .core import NormalizingFlow, MultiscaleFlow, ClassCondFlow, NormalizingFlowVAE, ConditionalNormalizingFlow
except Exception:
    NormalizingFlow = None
    MultiscaleFlow = None
    ClassCondFlow = None
    NormalizingFlowVAE = None
    ConditionalNormalizingFlow = None
   

from . import flows, distributions, nets, sampling, utils  # optional; guard similarly if needed

__all__ = [
    "NormalizingFlow", "MultiscaleFlow", "ClassCondFlow", "NormalizingFlowVAE", "ConditionalNormalizingFlow",
    "flows", "distributions", "nets", "sampling", "utils",
    "__version__",
]