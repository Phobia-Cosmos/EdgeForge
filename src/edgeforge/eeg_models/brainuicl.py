"""BrainUICL-compatible EEG frontend and Transformer decoder.

The original BrainUICL repository keeps its feature extractor, attention
encoder and sleep-stage MLP in separate files and requires dataset-specific
argument objects.  ``BrainUICLDecoder`` provides the same high-level graph
behind the common EdgeForge config/bundle interface, without importing that
repository.  It supports both ISRUC-style EEG+EOG input and FACED-style
single-branch input.
"""

from __future__ import annotations

from typing import Any

from .common import BrainUICLAttentionBlock, EEGDecoderBase, EEGForwardBundle, nn, require_torch, torch
from .config import EEGModelConfig


class BrainUICLFrontend(nn.Module):
    """Convolutional branch corresponding to BrainUICL's FeatureExtractor."""

    def __init__(self, channels: int, d_model: int, dropout: float) -> None:
        require_torch()
        super().__init__()
        # The first layer keeps the characteristic wide/strided filter from
        # BrainUICL.  Padding and adaptive pooling make the component usable
        # for short synthetic epochs as well as 2,500--3,000 sample epochs.
        self.layers = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=50, stride=6, padding=25, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=8, ceil_mode=True),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=8, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, d_model, kernel_size=8, padding=4),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        # The released BrainUICL frontend applies a final MaxPool1d(4, 4)
        # before global averaging.  Keep it as a named component; the short
        # synthetic-window fallback below skips it only when the feature map
        # is shorter than the kernel.
        self.final_pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, values: Any) -> Any:
        # For very short windows MaxPool1d(kernel=8) can be invalid.  The
        # normal EEG epochs are long; this fallback keeps smoke tests and
        # calibration probes deterministic without changing long-window math.
        if int(values.shape[-1]) < 8:
            first = self.layers[0:3](values)
            values = self.layers[4:](first)
        else:
            values = self.layers(values)
        if int(values.shape[-1]) >= 4:
            values = self.final_pool(values)
        return self.avg(values).squeeze(-1)


class BrainUICLDecoder(EEGDecoderBase):
    """BrainUICL-style dual-branch frontend + sequence Transformer.

    Parameters can be supplied directly or through ``config.options``:

    ``d_model`` (default 512), ``classifier_hidden`` (default
    ``config.feature_dim``), and ``attention_normalization_axis`` (default 1,
    matching the released BrainUICL implementation).
    """

    decoder_name = "brainuicl"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        config = config or EEGModelConfig(name=self.decoder_name, **kwargs)
        self.config = config
        options = dict(config.options or {})
        self.d_model = int(options.get("d_model", 512))
        self.classifier_hidden = int(options.get("classifier_hidden", config.feature_dim))
        self.eeg_channels = int(config.eeg_channels or config.in_channels)
        self.eog_channels = int(config.eog_channels)
        self.sequence_length = int(config.sequence_length or 0)
        # In the single-branch (FACED-like) form ``in_channels`` denotes the
        # total concatenated EEG/EOG channel count.  ``eeg_channels`` may be a
        # smaller bookkeeping value when callers provide the two tensors
        # separately, so build the frontend for the total width.
        frontend_channels = self.eeg_channels if self.eog_channels > 0 else int(config.in_channels)
        self.frontend_channels = frontend_channels
        self.eeg_frontend = BrainUICLFrontend(frontend_channels, self.d_model, config.dropout)
        self.eog_frontend = BrainUICLFrontend(self.eog_channels, self.d_model, config.dropout) if self.eog_channels > 0 else None
        self.fusion = nn.Linear(self.d_model * (2 if self.eog_frontend is not None else 1), self.d_model)
        # The released TransformerEncoder owns one attention/FFN block and
        # applies that same block ``layer_num`` times.  Keep the weight sharing
        # explicit instead of silently turning it into ``layer_num`` distinct
        # parameter sets.
        shared_block = BrainUICLAttentionBlock(
            self.d_model,
            config.heads if self.d_model % config.heads == 0 else self._compatible_heads(self.d_model, config.heads),
            dropout=config.dropout,
            mlp_ratio=config.mlp_ratio,
            normalization_axis=config.attention_normalization_axis,
        )
        self.encoder = nn.ModuleList([shared_block for _ in range(int(config.layers))])
        self.classifier_input = nn.Sequential(
            nn.Linear(self.d_model, self.classifier_hidden),
            nn.Dropout(config.dropout),
            nn.GELU(),
            nn.Linear(self.classifier_hidden, config.feature_dim),
            nn.Dropout(config.dropout),
            nn.GELU(),
        )
        self.classifier = nn.Linear(config.feature_dim, config.num_classes, bias=False)

    @staticmethod
    def _compatible_heads(width: int, requested: int) -> int:
        # Pick the largest divisor not exceeding the requested head count.
        for candidate in range(max(1, int(requested)), 0, -1):
            if width % candidate == 0:
                return candidate
        return 1

    def _flatten_branch(self, values: Any, expected_channels: int, name: str) -> tuple[Any, tuple[int, int] | None]:
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{name} input must be a torch.Tensor")
        if values.ndim == 4:
            batch, sequence, channels, samples = (int(item) for item in values.shape)
            flat = values.reshape(batch * sequence, channels, samples)
            shape: tuple[int, int] | None = (batch, sequence)
        elif values.ndim == 3:
            flat, shape = values, None
            if self.sequence_length > 1 and int(values.shape[0]) % self.sequence_length == 0:
                shape = (int(values.shape[0]) // self.sequence_length, self.sequence_length)
        else:
            raise ValueError(f"{name} input must have shape [B,C,S] or [B,T,C,S], got {tuple(values.shape)}")
        if int(flat.shape[1]) != int(expected_channels):
            raise ValueError(f"{name} expected {expected_channels} channels, got {int(flat.shape[1])}")
        return flat, shape

    @staticmethod
    def _restore_token(values: Any, shape: tuple[int, int] | None) -> Any:
        if shape is None:
            return values.squeeze(1)
        return values.reshape(shape[0], shape[1], *values.shape[2:])

    def _parse_inputs(self, values: Any, eog: Any | None) -> tuple[Any, Any | None, tuple[int, int] | None]:
        # Accept ``(eeg, eog)`` as well as separate ``forward(x, eog)``.
        if isinstance(values, (tuple, list)):
            if len(values) != 2:
                raise ValueError("BrainUICL tuple input must contain (eeg, eog)")
            values, eog = values
        # A combined tensor may carry EEG and EOG channels in one array.  Split
        # it before the per-branch channel validation.
        combined = False
        if self.eog_frontend is not None and eog is None and isinstance(values, torch.Tensor) and values.ndim in (3, 4):
            channel_count = int(values.shape[-2])
            if channel_count == self.eeg_channels + self.eog_channels:
                if values.ndim == 4:
                    values, eog = values[..., : self.eeg_channels, :], values[..., self.eeg_channels :, :]
                else:
                    values, eog = values[:, : self.eeg_channels], values[:, self.eeg_channels :]
                combined = True
        if self.eog_frontend is None and eog is None and isinstance(values, torch.Tensor):
            # Single-branch combined input; do not split or validate against a
            # bookkeeping EEG-only count.
            eeg, shape = self._flatten_branch(values, self.frontend_channels, "EEG")
        else:
            eeg, shape = self._flatten_branch(values, self.eeg_channels, "EEG")
        if self.eog_frontend is None:
            if eog is not None:
                # FACED callers often pass EOG and EEG separately; combine
                # them for the single branch, preserving the feature order.
                eog_flat, eog_shape = self._flatten_branch(eog, int(eog.shape[-2]) if eog.ndim >= 3 else 0, "EOG")
                if shape is not None and eog_shape not in (None, shape):
                    raise ValueError("EEG and EOG sequence shapes do not match")
                # Preserve the conventional combined layout: EEG channels
                # first, followed by auxiliary/EOG channels.  This keeps a
                # separately supplied pair numerically equivalent to the
                # corresponding concatenated tensor.
                eeg = torch.cat((eeg, eog_flat), dim=1)
            return eeg, None, shape
        if eog is None:
            # A combined tensor is convenient for dataloaders.  In that case
            # ``in_channels`` is expected to equal eeg+eog and we split here.
            if int(eeg.shape[1]) == self.eeg_channels + self.eog_channels and not combined:
                eeg, eog = eeg[:, : self.eeg_channels], eeg[:, self.eeg_channels :]
            else:
                raise ValueError("BrainUICL requires an EOG tensor or combined EEG+EOG channels")
        eog_flat, eog_shape = self._flatten_branch(eog, self.eog_channels, "EOG")
        if shape is not None and eog_shape not in (None, shape):
            raise ValueError("EEG and EOG sequence shapes do not match")
        return eeg, eog_flat, shape

    def forward_bundle(self, values: Any, eog: Any | None = None, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        require_torch()
        eeg, eog_flat, sequence_shape = self._parse_inputs(values, eog)
        eeg_feature = self.eeg_frontend(eeg)
        representations: dict[str, Any] = {}
        if self.eog_frontend is not None and eog_flat is not None:
            eog_feature = self.eog_frontend(eog_flat)
            fusion = self.fusion(torch.cat((eeg_feature, eog_feature), dim=-1))
        else:
            fusion = self.fusion(eeg_feature)
        # BrainUICL treats the T flattened epochs belonging to one subject as
        # Transformer tokens.  Keep that sequence together; using one token
        # per flattened epoch would silently remove all temporal attention.
        if sequence_shape is not None:
            tokens = fusion.reshape(sequence_shape[0], sequence_shape[1], self.d_model)
        else:
            tokens = fusion.unsqueeze(1)
        representations["fusion"] = tokens
        attention: dict[str, Any] = {}
        for index, block in enumerate(self.encoder):
            tokens = block(tokens)
            representations[f"transformer.{index}"] = tokens
            if block.last_attention is not None:
                attention[f"transformer.{index}"] = block.last_attention
        # For sequence inputs each flattened epoch is one token.  Rebuild the
        # token sequence before classification; for ordinary epochs the token
        # dimension is one and is removed in the returned bundle.
        tokens_for_output = tokens
        classifier_input = self.classifier_input(tokens_for_output)
        logits = self.classifier(classifier_input)
        if sequence_shape is None:
            logits = logits.squeeze(1)
        representations["classifier_input"] = classifier_input
        for key, item in list(representations.items()):
            representations[key] = self._restore_token(item, sequence_shape)
        return EEGForwardBundle(
            logits=logits,
            representations=representations,
            attention=attention,
            metadata={
                "architecture": self.decoder_name,
                "d_model": self.d_model,
                "attention_normalization_axis": int(self.config.attention_normalization_axis),
                "shared_attention_block": True,
                "input_layout": "[B,C,S], [B,T,C,S], (eeg,eog)",
                "eeg_channels": self.eeg_channels,
                "eog_channels": self.eog_channels,
            },
        )

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            "fusion": {"feature_axis": -1, "kind": "linear"},
            **{f"transformer.{index}": {"feature_axis": -1, "kind": "transformer"} for index in range(len(self.encoder))},
            "classifier_input": {"feature_axis": -1, "kind": "gelu"},
        }


BrainUICL = BrainUICLDecoder
