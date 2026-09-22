"""ShallowConvNet and DeepConvNet EEG decoders.

These blocks follow the broad Braindecode conventions (temporal filtering,
channel mixing, square/log features for the shallow model and stacked ELU
convolutions for the deep model) while exposing intermediate activations for
LoP diagnostics.
"""

from __future__ import annotations

from typing import Any

from .common import EEGDecoderBase, EEGForwardBundle, nn, require_torch, torch
from .config import EEGModelConfig


class ShallowConvNet(EEGDecoderBase):
    """Compact square/log convolutional decoder inspired by ShallowConvNet."""

    decoder_name = "shallowconvnet"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        config = config or EEGModelConfig(name=self.decoder_name, **kwargs)
        self.config = config
        width = int(config.width)
        temporal_kernel = max(3, min(int(config.kernel_size), int(config.input_length)))
        if temporal_kernel % 2 == 0:
            temporal_kernel -= 1
        self.temporal_conv = nn.Conv1d(config.in_channels, width, temporal_kernel, padding=temporal_kernel // 2, bias=False)
        self.temporal_norm = nn.BatchNorm1d(width)
        # Grouped 1x1 mixing acts as a depthwise spatial filter while retaining
        # a stable shape for arbitrary channel counts.
        self.spatial_conv = nn.Conv1d(width, width, kernel_size=1, groups=width, bias=False)
        self.spatial_norm = nn.BatchNorm1d(width)
        self.pool = nn.AvgPool1d(kernel_size=8, stride=4, padding=4, ceil_mode=True)
        self.dropout_layer = nn.Dropout(config.dropout)
        self.embedding_projection = nn.Linear(width, config.feature_dim)
        self.classifier_input = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        torch_mod = require_torch()
        flat, sequence_shape = self._flatten_epochs(values)
        if int(flat.shape[1]) != int(self.config.in_channels):
            raise ValueError(f"ShallowConvNet expected {self.config.in_channels} channels, got {int(flat.shape[1])}")
        representations: dict[str, Any] = {}
        x = self.temporal_norm(self.temporal_conv(flat))
        representations["temporal"] = x
        x = self.spatial_norm(self.spatial_conv(x))
        # The square/log transform is the characteristic shallow-net feature;
        # clamp keeps the diagnostic finite for silent EEG windows.
        x = torch_mod.log(torch_mod.clamp(self.dropout_layer(self.pool(x.square())), min=1e-6))
        representations["log_power"] = x
        pooled = x.mean(dim=-1)
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
            metadata={"architecture": self.decoder_name, "nonlinearity": "square+log"},
        )

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            "temporal": {"feature_axis": 1, "kind": "linear", "sequence_axis_policy": "prepend"},
            "log_power": {"feature_axis": 1, "kind": "log", "sequence_axis_policy": "prepend"},
            "embedding": {"feature_axis": -1, "kind": "linear"},
            "classifier_input": {"feature_axis": -1, "kind": "gelu"},
        }


class _DeepBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        require_torch()
        super().__init__()
        kernel = max(3, int(kernel_size))
        if kernel % 2 == 0:
            kernel -= 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Any) -> Any:
        return self.dropout(self.pool(self.activation(self.norm(self.conv(values)))))


class DeepConvNet(EEGDecoderBase):
    """Stacked convolutional EEG decoder inspired by DeepConvNet."""

    decoder_name = "deepconvnet"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        config = config or EEGModelConfig(name=self.decoder_name, **kwargs)
        self.config = config
        width = int(config.width)
        blocks = []
        in_channels = int(config.in_channels)
        for index in range(max(1, int(config.depth))):
            out_channels = width * (2**index)
            blocks.append(_DeepBlock(in_channels, out_channels, config.kernel_size, config.dropout))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_projection = nn.Linear(in_channels, config.feature_dim)
        self.classifier_input = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        require_torch()
        flat, sequence_shape = self._flatten_epochs(values)
        if int(flat.shape[1]) != int(self.config.in_channels):
            raise ValueError(f"DeepConvNet expected {self.config.in_channels} channels, got {int(flat.shape[1])}")
        representations: dict[str, Any] = {}
        x = flat
        for index, block in enumerate(self.blocks):
            x = block(x)
            representations[f"block{index + 1}"] = x
        representations["pooled"] = self.pool(x)
        embedding = self.embedding_projection(representations["pooled"].squeeze(-1))
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
            metadata={"architecture": self.decoder_name, "blocks": len(self.blocks)},
        )

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            **{
                f"block{index + 1}": {
                    "feature_axis": 1,
                    "kind": "elu",
                    "sequence_axis_policy": "prepend",
                }
                for index in range(len(self.blocks))
            },
            "pooled": {"feature_axis": 1, "kind": "pool", "sequence_axis_policy": "prepend"},
            "embedding": {"feature_axis": -1, "kind": "linear"},
            "classifier_input": {"feature_axis": -1, "kind": "gelu"},
        }


ShallowConvNetDecoder = ShallowConvNet
DeepConvNetDecoder = DeepConvNet
