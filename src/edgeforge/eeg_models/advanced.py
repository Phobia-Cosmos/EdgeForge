"""Additional EEG baselines commonly used in decoder comparisons."""

from __future__ import annotations

from typing import Any, Sequence

from .common import EEGDecoderBase, EEGForwardBundle, nn, require_torch, torch
from .config import EEGModelConfig
from .transformer import EEGTransformer


class LoPMLP(EEGDecoderBase):
    """Fully-connected LoP baseline with configurable hidden widths.

    The Nature LoP experiments use fixed-capacity MLP/linear controls in
    addition to convolutional networks.  Set ``options.hidden_dims`` to the
    exact widths from a paper; the default ``(256, 256, 256)`` keeps an EEG
    smoke test lightweight while preserving the three-hidden-layer topology.
    """

    decoder_name = "lop_mlp"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        config = config or EEGModelConfig(name=self.decoder_name, **kwargs)
        self.config = config
        options = dict(config.options or {})
        hidden_values = options.get("hidden_dims", (256, 256, 256))
        if isinstance(hidden_values, int):
            hidden_values = (hidden_values,) * 3
        hidden_dims: tuple[int, ...] = tuple(max(1, int(value)) for value in hidden_values)
        if not hidden_dims:
            raise ValueError("LoPMLP hidden_dims must not be empty")
        self.hidden_dims = hidden_dims
        input_dim = int(config.in_channels) * int(config.input_length)
        layers: list[Any] = []
        in_features = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(config.dropout)))
            in_features = hidden
        self.hidden = nn.Sequential(*layers)
        self.embedding_projection = nn.Linear(in_features, config.feature_dim)
        self.classifier_input = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        flat, sequence_shape = self._flatten_epochs(values)
        if int(flat.shape[1]) != int(self.config.in_channels):
            raise ValueError(f"LoPMLP expected {self.config.in_channels} channels, got {int(flat.shape[1])}")
        # Interpolate a differently sized epoch to the configured input length
        # instead of silently changing the parameter count.
        if int(flat.shape[-1]) != int(self.config.input_length):
            flat = torch.nn.functional.adaptive_avg_pool1d(flat, int(self.config.input_length))
        x = flat.reshape(flat.shape[0], -1)
        representations: dict[str, Any] = {}
        for index, layer in enumerate(self.hidden):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                representations[f"hidden.{index // 3}"] = x
        embedding = self.embedding_projection(x)
        representations["embedding"] = embedding
        classifier_input = self.classifier_input(embedding)
        representations["classifier_input"] = classifier_input
        logits = self.classifier(classifier_input)
        logits = self._restore_epochs(logits, sequence_shape)
        for key, item in list(representations.items()):
            representations[key] = self._restore_epochs(item, sequence_shape)
        return EEGForwardBundle(logits=logits, representations=representations, metadata={"architecture": self.decoder_name, "hidden_dims": list(self.hidden_dims)})

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            **{f"hidden.{index}": {"feature_axis": -1, "kind": "relu"} for index in range(len(self.hidden_dims))},
            "embedding": {"feature_axis": -1, "kind": "linear"},
            "classifier_input": {"feature_axis": -1, "kind": "gelu"},
        }


class FBCNet(EEGDecoderBase):
    """Lightweight filter-bank CNN baseline inspired by FBCNet.

    Each branch acts as a temporal sub-band filter; concatenated branch
    features are pooled and projected for classification.  It is intentionally
    configurable rather than tied to a particular sampling rate.
    """

    decoder_name = "fbcnet"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        config = config or EEGModelConfig(name=self.decoder_name, **kwargs)
        self.config = config
        options = dict(config.options or {})
        kernels = options.get("filter_kernels", (7, 15, 31))
        if isinstance(kernels, int):
            kernels = (kernels,)
        self.filter_kernels: tuple[int, ...] = tuple(max(3, int(k)) for k in kernels)
        width = int(config.width)
        branches = []
        for kernel in self.filter_kernels:
            if kernel % 2 == 0:
                kernel += 1
            branches.append(
                nn.Sequential(
                    nn.Conv1d(config.in_channels, width, kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                )
            )
        self.filter_bank = nn.ModuleList(branches)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_projection = nn.Linear(width * len(branches), config.feature_dim)
        self.classifier_input = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        flat, sequence_shape = self._flatten_epochs(values)
        if int(flat.shape[1]) != int(self.config.in_channels):
            raise ValueError(f"FBCNet expected {self.config.in_channels} channels, got {int(flat.shape[1])}")
        representations: dict[str, Any] = {}
        branch_values = [branch(flat) for branch in self.filter_bank]
        for index, item in enumerate(branch_values):
            representations[f"band.{index}"] = item
        concatenated = torch.cat(branch_values, dim=1)
        representations["filter_bank"] = concatenated
        pooled = self.pool(concatenated).squeeze(-1)
        embedding = self.embedding_projection(pooled)
        representations["embedding"] = embedding
        classifier_input = self.classifier_input(embedding)
        representations["classifier_input"] = classifier_input
        logits = self.classifier(classifier_input)
        logits = self._restore_epochs(logits, sequence_shape)
        for key, item in list(representations.items()):
            representations[key] = self._restore_epochs(item, sequence_shape)
        return EEGForwardBundle(logits=logits, representations=representations, metadata={"architecture": self.decoder_name, "filter_kernels": list(self.filter_kernels)})

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            **{f"band.{index}": {"feature_axis": 1, "kind": "gelu"} for index in range(len(self.filter_bank))},
            "filter_bank": {"feature_axis": 1, "kind": "gelu"},
            "embedding": {"feature_axis": -1, "kind": "linear"},
            "classifier_input": {"feature_axis": -1, "kind": "gelu"},
        }


class EEGConformer(EEGTransformer):
    """Convolution-augmented Transformer baseline.

    A depthwise local temporal projection precedes the patch-token encoder;
    this captures the convolutional front-end used by common EEG Conformer
    implementations while retaining the shared Transformer instrumentation.
    """

    decoder_name = "conformer"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config or EEGModelConfig(name=self.decoder_name, **kwargs))
        kernel = max(3, min(int(self.config.kernel_size), int(self.config.input_length)))
        if kernel % 2 == 0:
            kernel -= 1
        self.local_projection = nn.Sequential(
            nn.Conv1d(self.config.in_channels, self.config.in_channels, kernel, padding=kernel // 2, groups=self.config.in_channels, bias=False),
            nn.BatchNorm1d(self.config.in_channels),
            nn.GELU(),
        )

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        flat, sequence_shape = self._flatten_epochs(values)
        local = self.local_projection(flat)
        local_values = self._restore_epochs(local, sequence_shape)
        bundle = super().forward_bundle(local_values, *args, **kwargs)
        # Keep this tap flattened as ``[B*T,C,S]`` for sequence inputs.  A
        # fixed feature axis of 1 then always denotes channels; restoring it
        # to ``[B,T,C,S]`` would make the axis depend on the input rank and
        # could silently measure sequence positions instead of channels.
        bundle.representations["local_projection"] = local
        return bundle

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        specs = super().representation_specs()
        specs["local_projection"] = {"feature_axis": 1, "kind": "gelu", "layout": "flattened_epoch_map"}
        return specs
