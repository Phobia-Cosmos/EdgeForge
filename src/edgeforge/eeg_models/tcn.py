"""Temporal convolutional network (TCN) EEG decoder."""

from __future__ import annotations

from typing import Any, Sequence

from .common import EEGDecoderBase, EEGForwardBundle, nn, require_torch, torch
from .config import EEGModelConfig


class TemporalResidualBlock(nn.Module):
    """A causal-compatible (same-length) dilated residual block."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float) -> None:
        require_torch()
        super().__init__()
        dilation = max(1, int(dilation))
        padding = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, values: Any) -> Any:
        residual = self.skip(values)
        values = self.dropout1(self.activation1(self.norm1(self.conv1(values))))
        values = self.dropout2(self.activation2(self.norm2(self.conv2(values))))
        return values + residual


class TCN(EEGDecoderBase):
    """Dilated temporal CNN with named layer representations."""

    decoder_name = "tcn"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        config = config or EEGModelConfig(name=self.decoder_name, **kwargs)
        self.config = config
        width = int(config.width)
        dilations: Sequence[int] = config.dilations
        blocks = []
        in_channels = int(config.in_channels)
        for dilation in dilations[: max(1, int(config.depth))]:
            blocks.append(TemporalResidualBlock(in_channels, width, int(dilation), float(config.dropout)))
            in_channels = width
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_projection = nn.Linear(width, config.feature_dim)
        self.classifier_input = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        require_torch()
        flat, sequence_shape = self._flatten_epochs(values)
        if int(flat.shape[1]) != int(self.config.in_channels):
            raise ValueError(f"TCN expected {self.config.in_channels} channels, got {int(flat.shape[1])}")
        representations: dict[str, Any] = {}
        x = flat
        for index, block in enumerate(self.blocks):
            x = block(x)
            representations[f"block{index + 1}"] = x
        pooled_map = self.pool(x)
        representations["pooled"] = pooled_map
        embedding = self.embedding_projection(pooled_map.squeeze(-1))
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
            metadata={"architecture": self.decoder_name, "dilations": [int(item) for item in self.config.dilations]},
        )

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            **{
                f"block{index + 1}": {
                    "feature_axis": 1,
                    "kind": "gelu",
                    "sequence_axis_policy": "prepend",
                }
                for index in range(len(self.blocks))
            },
            "pooled": {"feature_axis": 1, "kind": "pool", "sequence_axis_policy": "prepend"},
            "embedding": {"feature_axis": -1, "kind": "linear"},
            "classifier_input": {"feature_axis": -1, "kind": "gelu"},
        }


TCNDecoder = TCN
