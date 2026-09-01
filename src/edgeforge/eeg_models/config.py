"""Configuration objects for the reusable EEG decoder components.

The diagnostic scripts historically constructed small networks inline.  This
module keeps architecture choices in a serialisable dataclass so the same
decoder can be used by training, LoP probes and deployment code.  The config
is deliberately conservative and does not import PyTorch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass
class EEGModelConfig:
    """Common options shared by the EEG decoder implementations.

    Inputs normally use ``[batch, channels, samples]``.  A four-dimensional
    ``[batch, sequence, channels, samples]`` tensor is also accepted by all
    built-in decoders; the sequence axis is flattened during feature
    extraction and restored in the returned logits.

    ``name`` selects a registered architecture.  Extra architecture-specific
    values are kept here rather than in ad-hoc dictionaries so experiment
    manifests can be dumped with :meth:`to_dict`.
    """

    name: str = "eegnet"
    in_channels: int = 8
    input_length: int = 128
    num_classes: int = 4
    feature_dim: int = 64
    dropout: float = 0.1

    # CNN/TCN widths and depth.
    width: int = 24
    depth: int = 3
    kernel_size: int = 15
    dilations: tuple[int, ...] = (1, 2, 4)

    # Transformer options.
    patch_size: int = 8
    layers: int = 2
    heads: int = 4
    mlp_ratio: float = 4.0

    # BrainUICL-style dual-branch frontend.  ``eog_channels=0`` selects the
    # single-branch FACED-like variant; a positive value enables the ISRUC
    # EEG+EOG fusion branch.  When ``eeg_channels`` is omitted in the latter
    # form, it is inferred as ``in_channels - eog_channels``.
    eeg_channels: int | None = None
    eog_channels: int = 0
    sequence_length: int | None = None
    attention_normalization_axis: int = 1

    # Optional model-specific knobs accepted by custom builders.  Keeping a
    # small metadata mapping avoids proliferating fields while remaining
    # serialisable and validated by ``from_mapping``.
    options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Convert list values loaded from JSON to the immutable tuple used by
        # the implementation, while preserving user supplied values.
        if not isinstance(self.dilations, tuple):
            self.dilations = tuple(int(item) for item in self.dilations)
        if self.sequence_length is not None:
            self.sequence_length = int(self.sequence_length)
        self.name = str(self.name).strip().lower()
        for field_name in ("in_channels", "input_length", "num_classes", "feature_dim", "width", "depth", "patch_size", "layers", "heads"):
            value = int(getattr(self, field_name))
            if value < 1:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        self.eog_channels = int(self.eog_channels)
        if self.eeg_channels is not None:
            self.eeg_channels = int(self.eeg_channels)
        if self.eeg_channels is None:
            # ``in_channels`` denotes the total concatenated width when an
            # EOG branch is enabled.  Inferring the EEG remainder makes the
            # concise ``in_channels=8, eog_channels=2`` form safe while still
            # preserving the explicit ``eeg_channels`` override used by
            # existing ISRUC manifests.
            self.eeg_channels = self.in_channels - int(self.eog_channels)
        if self.eeg_channels < 1:
            raise ValueError("eeg_channels must be positive")
        if self.eog_channels < 0:
            raise ValueError("eog_channels must be non-negative")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not self.dilations:
            raise ValueError("dilations must not be empty")
        if self.options is None:
            self.options = {}
        elif not isinstance(self.options, Mapping):
            raise TypeError("options must be a mapping")
        else:
            self.options = dict(self.options)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None, **overrides: Any) -> "EEGModelConfig":
        """Create a config from JSON/YAML-like values.

        Unknown keys are placed in ``options`` rather than silently dropped;
        this makes typoed architecture options visible in experiment output
        while allowing custom registry entries to consume their own values.
        """

        merged: dict[str, Any] = dict(values or {})
        merged.update(overrides)
        known = {field.name for field in fields(cls)}
        extras = {key: merged.pop(key) for key in list(merged) if key not in known}
        existing_options = merged.get("options")
        if extras:
            if existing_options is None:
                merged["options"] = extras
            elif isinstance(existing_options, Mapping):
                merged["options"] = {**dict(existing_options), **extras}
            else:
                raise TypeError("options must be a mapping when supplied")
        return cls(**merged)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of this config."""

        result = asdict(self)
        result["dilations"] = list(self.dilations)
        # Return a copy so callers cannot mutate the config through metadata.
        result["options"] = dict(self.options or {})
        return result


# A descriptive alias used by some callers and older experiment manifests.
EEGDecoderConfig = EEGModelConfig
