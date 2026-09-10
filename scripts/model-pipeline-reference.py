#!/usr/bin/env python3
"""Compatibility entry point for the packaged reference adapter."""

import sys
from pathlib import Path

try:
    from edgeforge.reference_model_pipeline import main
except ModuleNotFoundError:
    # Keep the repository script runnable before an editable/package install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from edgeforge.reference_model_pipeline import main


if __name__ == "__main__":
    main()
