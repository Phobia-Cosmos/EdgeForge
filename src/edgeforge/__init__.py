"""EdgeForge heterogeneous edge runtime."""

__version__ = "0.17.0"

# The envelope converter has no optional runtime dependencies and is safe to
# expose at the package boundary.  Keep the import small so importing
# ``edgeforge`` does not pull in PyTorch.
from .lop_envelope import (  # noqa: E402,F401
    diagnostic_to_metrics,
    diagnostics_to_metrics,
    edgeforge_bundle_from_diagnostic,
    to_edgeforge_metrics,
)
from .lop_analysis import (  # noqa: E402,F401
    DEFAULT_LOP_OUTCOME,
    evaluate_lop_gate,
)

__all__ = [
    "__version__",
    "diagnostic_to_metrics",
    "diagnostics_to_metrics",
    "edgeforge_bundle_from_diagnostic",
    "to_edgeforge_metrics",
    "DEFAULT_LOP_OUTCOME",
    "evaluate_lop_gate",
]
