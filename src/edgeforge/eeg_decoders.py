"""Backward-compatible flat exports for standalone EEG decoders.

Prefer ``edgeforge.eeg_models`` for new code; this module keeps a concise
import path for experiment scripts.
"""

from .eeg_models import *  # noqa: F401,F403

