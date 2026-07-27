import warnings

from ._version import __version__  # light import

# Avoid importing torch-consuming modules at build time (e.g. when torch is
# not installed, such as during packaging/metadata introspection). Any other
# failure is unexpected, so warn rather than silently returning None for
# every class - otherwise a real bug in core.py just becomes a confusing
# "NoneType is not callable" later.
try:
    from .core import NormalizingFlow, MultiscaleFlow, ClassCondFlow, NormalizingFlowVAE, ConditionalNormalizingFlow
except Exception as exc:
    warnings.warn(
        f"antsnormflows.core failed to import ({exc!r}); NormalizingFlow, "
        "MultiscaleFlow, ClassCondFlow, NormalizingFlowVAE and "
        "ConditionalNormalizingFlow will be unavailable (set to None)."
    )
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