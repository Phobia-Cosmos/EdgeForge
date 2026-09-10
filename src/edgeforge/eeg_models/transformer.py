"""Patch-token Transformer decoder for multichannel EEG epochs."""

from __future__ import annotations

from typing import Any

from .common import EEGDecoderBase, EEGForwardBundle, InspectableTransformerBlock, F, nn, require_torch, torch
from .config import EEGModelConfig


class EEGTransformer(EEGDecoderBase):
    """Temporal patch Transformer with exposed per-head attention maps."""

    decoder_name = "transformer"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        config = config or EEGModelConfig(name=self.decoder_name, **kwargs)
        self.config = config
        width = int(config.width)
        heads = int(config.heads)
        if width % heads:
            raise ValueError("transformer width must be divisible by heads")
        # Keep the learned patch kernel valid for the configured epoch.  A
        # runtime calibration window may still be shorter than that epoch, so
        # ``forward_bundle`` pads it before applying the convolution.
        patch = min(max(1, int(config.patch_size)), max(1, int(config.input_length)))
        self.patch_size = patch
        self.patch_embed = nn.Conv1d(config.in_channels, width, kernel_size=patch, stride=patch)
        token_count = max(1, (int(config.input_length) - patch) // patch + 1)
        self.position = nn.Parameter(torch.zeros(1, token_count, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.token_norm = nn.LayerNorm(width)
        self.encoder = nn.ModuleList(
            [InspectableTransformerBlock(width, heads, dropout=config.dropout, mlp_ratio=config.mlp_ratio) for _ in range(int(config.layers))]
        )
        self.embedding_projection = nn.Linear(width, config.feature_dim)
        self.classifier_input = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)

    def _position_for(self, token_count: int, device: Any, dtype: Any) -> Any:
        position = self.position
        if token_count == int(position.shape[1]):
            return position.to(device=device, dtype=dtype)
        # Interpolate along the token axis so a decoder configured for one
        # epoch length can still instrument another length.
        resized = F.interpolate(position.transpose(1, 2), size=token_count, mode="linear", align_corners=False)
        return resized.transpose(1, 2).to(device=device, dtype=dtype)

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        require_torch()
        flat, sequence_shape = self._flatten_epochs(values)
        if int(flat.shape[1]) != int(self.config.in_channels):
            raise ValueError(f"EEGTransformer expected {self.config.in_channels} channels, got {int(flat.shape[1])}")
        representations: dict[str, Any] = {}
        if int(flat.shape[-1]) < self.patch_size:
            flat = F.pad(flat, (0, self.patch_size - int(flat.shape[-1])))
        tokens = self.patch_embed(flat).transpose(1, 2)
        tokens = tokens + self._position_for(int(tokens.shape[1]), tokens.device, tokens.dtype)
        tokens = self.token_norm(tokens)
        representations["tokens"] = tokens
        attention: dict[str, Any] = {}
        for index, block in enumerate(self.encoder):
            tokens = block(tokens)
            representations[f"encoder.{index}"] = tokens
            if block.last_attention is not None:
                attention[f"encoder.{index}"] = block.last_attention
        pooled = tokens.mean(dim=1)
        representations["pooled"] = pooled
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
            attention=attention,
            metadata={
                "architecture": self.decoder_name,
                "attention_normalization_axis": -1,
                "input_layout": "[B,C,S] or [B,T,C,S]",
            },
        )

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            "tokens": {"feature_axis": -1, "kind": "transformer"},
            **{f"encoder.{index}": {"feature_axis": -1, "kind": "transformer"} for index in range(len(self.encoder))},
            "pooled": {"feature_axis": -1, "kind": "pool"},
            "embedding": {"feature_axis": -1, "kind": "linear"},
            "classifier_input": {"feature_axis": -1, "kind": "gelu"},
        }


# More explicit names are useful when multiple Transformer families are
# registered in an experiment manifest.
TransformerDecoder = EEGTransformer
