"""EEGNet-style depthwise/separable convolutional decoder."""

from __future__ import annotations

from typing import Any

from .common import EEGDecoderBase, EEGForwardBundle, nn, require_torch, torch
from .config import EEGModelConfig


class EEGNet(EEGDecoderBase):
    """Compact EEGNet variant suitable for epoch-level decoding.

    The first convolution learns temporal filters.  A depthwise temporal
    convolution followed by a pointwise projection provides the separable
    stage used by EEGNet while keeping the input layout ``[B,C,S]``.  Both
    regular epochs and ``[B,T,C,S]`` streams are accepted.
    """

    decoder_name = "eegnet"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        config = config or EEGModelConfig(name=self.decoder_name, **kwargs)
        self.config = config
        channels = int(config.in_channels)
        width = int(config.width)
        kernel = max(3, int(config.kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        options = dict(config.options or {})
        # Canonical EEGNet performs temporal filtering independently at each
        # channel and then a depthwise spatial convolution over the channel
        # axis.  Keep an explicit opt-out for legacy 1-D probes that already
        # mix channels in the first convolution.
        self.canonical_spatial = bool(options.get("canonical_spatial", True))
        if self.canonical_spatial:
            self.temporal_conv = nn.Conv2d(1, width, kernel_size=(1, kernel), padding=(0, kernel // 2), bias=False)
            self.temporal_norm = nn.BatchNorm2d(width)
        else:
            self.temporal_conv = nn.Conv1d(channels, width, kernel_size=kernel, padding=kernel // 2, bias=False)
            self.temporal_norm = nn.BatchNorm1d(width)
        self.temporal_activation = nn.ReLU()
        depthwise_kernel = max(3, min(15, kernel // 2 if kernel > 5 else 5))
        if depthwise_kernel % 2 == 0:
            depthwise_kernel += 1
        if self.canonical_spatial:
            # Spatial depthwise filters span all input channels.  The
            # temporal dimension is left untouched before pointwise mixing.
            self.depthwise_conv = nn.Conv2d(width, width, kernel_size=(channels, 1), groups=width, bias=False)
            self.pointwise_conv = nn.Conv2d(width, 2 * width, kernel_size=1, bias=False)
            self.separable_norm = nn.BatchNorm2d(2 * width)
        else:
            self.depthwise_conv = nn.Conv1d(width, width, kernel_size=depthwise_kernel, padding=depthwise_kernel // 2, groups=width, bias=False)
            self.pointwise_conv = nn.Conv1d(width, 2 * width, kernel_size=1, bias=False)
            self.separable_norm = nn.BatchNorm1d(2 * width)
        self.separable_activation = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), ceil_mode=True) if self.canonical_spatial else nn.AvgPool1d(kernel_size=4, stride=4, ceil_mode=True)
        self.dropout_layer = nn.Dropout(config.dropout)
        self.embedding_projection = nn.Linear(2 * width, config.feature_dim)
        self.classifier_input = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        require_torch()
        flat, sequence_shape = self._flatten_epochs(values)
        if int(flat.shape[1]) != int(self.config.in_channels):
            raise ValueError(f"EEGNet expected {self.config.in_channels} channels, got {int(flat.shape[1])}")
        representations: dict[str, Any] = {}
        x = self.temporal_conv(flat.unsqueeze(1) if self.canonical_spatial else flat)
        x = self.temporal_activation(self.temporal_norm(x))
        representations["temporal"] = x
        x = self.depthwise_conv(x)
        x = self.separable_activation(self.separable_norm(self.pointwise_conv(x)))
        representations["separable"] = x
        x = self.dropout_layer(self.pool(x))
        representations["pooled"] = x
        pooled = x.mean(dim=(-1, -2)) if self.canonical_spatial else x.mean(dim=-1)
        embedding = self.embedding_projection(pooled)
        representations["embedding"] = embedding
        classifier_input = self.classifier_input(embedding)
        representations["classifier_input"] = classifier_input
        logits = self.classifier(classifier_input)
        logits = self._restore_epochs(logits, sequence_shape)
        for key, item in list(representations.items()):
            representations[key] = self._restore_epochs(item, sequence_shape)
        return EEGForwardBundle(
            logits=logits,
            representations=representations,
            metadata={
                "architecture": self.decoder_name,
                "input_layout": "[B,C,S] or [B,T,C,S]",
                "canonical_spatial_mixing": self.canonical_spatial,
                "spatial_mixing_caveat": "Conv2d depthwise channel filter" if self.canonical_spatial else "legacy Conv1d channel-mixed frontend",
            },
        )

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            "temporal": {"feature_axis": 1, "kind": "conv", "sequence_axis_policy": "prepend"},
            "separable": {"feature_axis": 1, "kind": "elu", "sequence_axis_policy": "prepend"},
            "pooled": {"feature_axis": 1, "kind": "pool", "sequence_axis_policy": "prepend"},
            "embedding": {"feature_axis": -1, "kind": "linear"},
            "classifier_input": {"feature_axis": -1, "kind": "gelu"},
        }


# Explicit alias used in a number of EEG decoding repositories.
EEGNetDecoder = EEGNet
