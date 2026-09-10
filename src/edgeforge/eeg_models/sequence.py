"""Optional multi-scale, recurrent and graph EEG decoder baselines.

The classes in this module intentionally expose a small, dependency-free
implementation of several frequently used EEG decoder families.  They are
component baselines for architecture/LoP comparisons, not claims that every
paper with the same name uses identical kernels or preprocessing.
"""

from __future__ import annotations

from typing import Any, Sequence

from .common import EEGDecoderBase, EEGForwardBundle, F, nn, require_torch, torch
from .config import EEGModelConfig


class TSception(EEGDecoderBase):
    """Multi-scale temporal inception + spatial mixing decoder.

    Each temporal branch uses a different odd kernel, after which a pointwise
    spatial mixer combines all channels.  The branch taps are useful for
    testing whether a domain shift affects short- or long-time filters.
    """

    decoder_name = "tsception"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        config = config or EEGModelConfig(name=self.decoder_name, **kwargs)
        self.config = config
        options = dict(config.options or {})
        kernels = options.get("temporal_kernels", (7, 15, 31))
        if isinstance(kernels, int):
            kernels = (kernels,)
        self.temporal_kernels: tuple[int, ...] = tuple(max(3, int(item)) for item in kernels)
        if not self.temporal_kernels:
            raise ValueError("TSception temporal_kernels must not be empty")
        width = int(config.width)
        branches: list[Any] = []
        for kernel in self.temporal_kernels:
            if kernel % 2 == 0:
                kernel += 1
            branches.append(
                nn.Sequential(
                    nn.Conv1d(config.in_channels, width, kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                )
            )
        self.temporal_branches = nn.ModuleList(branches)
        total = width * len(branches)
        self.spatial_mixer = nn.Sequential(
            nn.Conv1d(total, total, kernel_size=1, bias=False),
            nn.BatchNorm1d(total),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_projection = nn.Linear(total, config.feature_dim)
        self.classifier_input = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        require_torch()
        flat, sequence_shape = self._flatten_epochs(values)
        if int(flat.shape[1]) != int(self.config.in_channels):
            raise ValueError(f"TSception expected {self.config.in_channels} channels, got {int(flat.shape[1])}")
        branch_values = [branch(flat) for branch in self.temporal_branches]
        representations: dict[str, Any] = {
            f"temporal.{index}": item for index, item in enumerate(branch_values)
        }
        concatenated = torch.cat(branch_values, dim=1)
        representations["temporal_concat"] = concatenated
        spatial = self.spatial_mixer(concatenated)
        representations["spatial"] = spatial
        pooled = self.pool(spatial).squeeze(-1)
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
            metadata={
                "architecture": self.decoder_name,
                "temporal_kernels": list(self.temporal_kernels),
                "input_layout": "[B,C,S] or [B,T,C,S]",
            },
        )

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            **{
                f"temporal.{index}": {
                    "feature_axis": 1,
                    "kind": "gelu",
                    "sequence_axis_policy": "prepend",
                }
                for index in range(len(self.temporal_branches))
            },
            "temporal_concat": {"feature_axis": 1, "kind": "gelu", "sequence_axis_policy": "prepend"},
            "spatial": {"feature_axis": 1, "kind": "gelu", "sequence_axis_policy": "prepend"},
            "pooled": {"feature_axis": -1, "kind": "pool"},
            "embedding": {"feature_axis": -1, "kind": "linear"},
            "classifier_input": {"feature_axis": -1, "kind": "gelu"},
        }


class ATCNet(EEGDecoderBase):
    """Attention-temporal-convolution baseline inspired by ATCNet.

    A compact convolutional stem produces a fixed number of tokens.  Tokens
    are partitioned into overlapping-style windows (the interpolation makes
    the implementation length agnostic), each processed by key-axis MHA,
    followed by a temporal convolution over window summaries.
    """

    decoder_name = "atcnet"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        config = config or EEGModelConfig(name=self.decoder_name, **kwargs)
        self.config = config
        options = dict(config.options or {})
        self.windows = max(1, int(options.get("windows", 4)))
        self.tokens_per_window = max(1, int(options.get("tokens_per_window", 4)))
        width = int(config.width)
        stem_kernel = min(max(1, int(config.kernel_size)), max(1, int(config.input_length)))
        if stem_kernel > 1 and stem_kernel % 2 == 0:
            stem_kernel -= 1
        self.stem_kernel = stem_kernel
        self.stem = nn.Sequential(
            nn.Conv1d(config.in_channels, width, kernel_size=stem_kernel, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        heads = int(config.heads) if width % int(config.heads) == 0 else 1
        self.window_attention = nn.ModuleList(
            [nn.MultiheadAttention(width, heads, batch_first=True, dropout=config.dropout) for _ in range(self.windows)]
        )
        self.window_norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(self.windows)])
        self.temporal = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.embedding_projection = nn.Linear(width, config.feature_dim)
        self.classifier_input = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)
        self.last_attention: dict[str, Any] = {}
        self.attention_heads = heads

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        torch_mod = require_torch()
        flat, sequence_shape = self._flatten_epochs(values)
        if int(flat.shape[1]) != int(self.config.in_channels):
            raise ValueError(f"ATCNet expected {self.config.in_channels} channels, got {int(flat.shape[1])}")
        stem = self.stem(flat)
        representations: dict[str, Any] = {"stem": stem}
        total_tokens = self.windows * self.tokens_per_window
        tokens = torch_mod.nn.functional.adaptive_avg_pool1d(stem, total_tokens).transpose(1, 2)
        windows = tokens.reshape(tokens.shape[0], self.windows, self.tokens_per_window, tokens.shape[-1])
        summaries: list[Any] = []
        attention: dict[str, Any] = {}
        self.last_attention = {}
        for index, (attention_layer, norm) in enumerate(zip(self.window_attention, self.window_norm)):
            local = windows[:, index]
            attended, weights = attention_layer(local, local, local, need_weights=True, average_attn_weights=False)
            local = norm(local + attended)
            summaries.append(local.mean(dim=1))
            attention[f"window.{index}"] = weights
            self.last_attention[f"window.{index}"] = weights.detach()
        window_features = torch_mod.stack(summaries, dim=1)
        representations["window_features"] = window_features
        temporal = self.temporal(window_features.transpose(1, 2)).transpose(1, 2)
        representations["temporal"] = temporal
        pooled = temporal.mean(dim=1)
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
                "attention_axis_semantics": "key/token axis",
                "windows": self.windows,
                "tokens_per_window": self.tokens_per_window,
            },
        )

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            "stem": {"feature_axis": 1, "kind": "gelu", "sequence_axis_policy": "prepend"},
            "window_features": {"feature_axis": -1, "kind": "transformer"},
            "temporal": {"feature_axis": -1, "kind": "gelu"},
            "pooled": {"feature_axis": -1, "kind": "pool"},
            "embedding": {"feature_axis": -1, "kind": "linear"},
            "classifier_input": {"feature_axis": -1, "kind": "gelu"},
        }


class CNNLSTM(EEGDecoderBase):
    """Convolutional epoch encoder followed by an LSTM context model."""

    decoder_name = "cnn_lstm"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        config = config or EEGModelConfig(name=self.decoder_name, **kwargs)
        self.config = config
        width = int(config.width)
        hidden = int((config.options or {}).get("lstm_hidden", width))
        cnn_kernel = min(max(1, int(config.kernel_size)), max(1, int(config.input_length)))
        if cnn_kernel > 1 and cnn_kernel % 2 == 0:
            cnn_kernel -= 1
        self.cnn_kernel = cnn_kernel
        self.cnn = nn.Sequential(
            nn.Conv1d(config.in_channels, width, kernel_size=cnn_kernel, padding=cnn_kernel // 2, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lstm = nn.LSTM(width, hidden, num_layers=max(1, int(config.depth)), batch_first=True, dropout=config.dropout if int(config.depth) > 1 else 0.0)
        self.embedding_projection = nn.Linear(hidden, config.feature_dim)
        self.classifier_input = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)
        self.hidden_size = hidden

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        flat, sequence_shape = self._flatten_epochs(values)
        if int(flat.shape[1]) != int(self.config.in_channels):
            raise ValueError(f"CNNLSTM expected {self.config.in_channels} channels, got {int(flat.shape[1])}")
        cnn_features = self.cnn(flat).squeeze(-1)
        if sequence_shape is None:
            sequence_shape = (int(cnn_features.shape[0]), 1)
        sequence = cnn_features.reshape(sequence_shape[0], sequence_shape[1], -1)
        hidden, _state = self.lstm(sequence)
        embedding = self.embedding_projection(hidden)
        classifier_input = self.classifier_input(embedding)
        logits = self.classifier(classifier_input)
        representations: dict[str, Any] = {
            "cnn_features": sequence,
            "lstm_hidden": hidden,
            "embedding": embedding,
            "classifier_input": classifier_input,
        }
        if values.ndim == 3:
            logits = logits.squeeze(1)
            representations = {key: item.squeeze(1) for key, item in representations.items()}
        return EEGForwardBundle(
            logits=logits,
            representations=representations,
            metadata={"architecture": self.decoder_name, "recurrent": "LSTM", "hidden_size": self.hidden_size},
        )

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            "cnn_features": {"feature_axis": -1, "kind": "gelu"},
            "lstm_hidden": {"feature_axis": -1, "kind": "linear"},
            "embedding": {"feature_axis": -1, "kind": "linear"},
            "classifier_input": {"feature_axis": -1, "kind": "gelu"},
        }


class EEGGraphNet(EEGDecoderBase):
    """Learnable-adjacency graph EEG decoder.

    Electrodes are graph nodes.  The adjacency is kept in bundle metadata so
    diagnostics can version and inspect it separately from flattened node
    representations.
    """

    decoder_name = "eeg_graph"

    def __init__(self, config: EEGModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        config = config or EEGModelConfig(name=self.decoder_name, **kwargs)
        self.config = config
        width = int(config.width)
        # An even ``kernel`` with symmetric padding changes the temporal
        # length by one (e.g. length 4 -> 5), which would make the subsequent
        # per-node projection receive the wrong feature width.  Use an odd
        # same-length kernel whenever the input is long enough, and fall back
        # to a 1-tap projection for a one-sample smoke/calibration window.
        kernel = min(max(1, int(config.kernel_size)), max(1, int(config.input_length)))
        if kernel > 1 and kernel % 2 == 0:
            kernel -= 1
        self.node_temporal_kernel = kernel
        self.node_temporal = nn.Sequential(
            nn.Conv1d(config.in_channels, config.in_channels * width, kernel_size=kernel, padding=kernel // 2, groups=config.in_channels, bias=False),
            nn.BatchNorm1d(config.in_channels * width),
            nn.GELU(),
        )
        self.node_projection = nn.Linear(width, width)
        self.graph_projection = nn.Linear(width, width)
        self.adjacency_logits = nn.Parameter(torch.zeros(config.in_channels, config.in_channels))
        self.embedding_projection = nn.Linear(width, config.feature_dim)
        self.classifier_input = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.GELU(), nn.Dropout(config.dropout))
        self.classifier = nn.Linear(config.feature_dim, config.num_classes)
        self.node_count = int(config.in_channels)

    def _adjacency(self) -> Any:
        return torch.softmax(self.adjacency_logits, dim=-1)

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        torch_mod = require_torch()
        flat, sequence_shape = self._flatten_epochs(values)
        if int(flat.shape[1]) != self.node_count:
            raise ValueError(f"EEGGraphNet expected {self.node_count} channels, got {int(flat.shape[1])}")
        node_map = self.node_temporal(flat).reshape(flat.shape[0], self.node_count, -1, flat.shape[-1]).mean(dim=-1)
        node_features = self.node_projection(node_map)
        adjacency = self._adjacency()
        aggregated = torch_mod.einsum("ij,njd->nid", adjacency, node_features)
        graph_features = F.gelu(self.graph_projection(aggregated) + node_features)
        pooled = graph_features.mean(dim=1)
        embedding = self.embedding_projection(pooled)
        classifier_input = self.classifier_input(embedding)
        logits = self.classifier(classifier_input)
        representations: dict[str, Any] = {
            "node_features": node_features,
            "graph_features": graph_features,
            "pooled": pooled,
            "embedding": embedding,
            "classifier_input": classifier_input,
        }
        logits = self._restore_epochs(logits, sequence_shape)
        for key, item in list(representations.items()):
            representations[key] = self._restore_epochs(item, sequence_shape)
        return EEGForwardBundle(
            logits=logits,
            representations=representations,
            metadata={
                "architecture": self.decoder_name,
                "node_count": self.node_count,
                "adjacency": adjacency.detach(),
                "adjacency_semantics": "row-softmax learnable electrode graph",
            },
        )

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        return {
            "node_features": {"feature_axis": -1, "kind": "gelu"},
            "graph_features": {"feature_axis": -1, "kind": "gelu"},
            "pooled": {"feature_axis": -1, "kind": "pool"},
            "embedding": {"feature_axis": -1, "kind": "linear"},
            "classifier_input": {"feature_axis": -1, "kind": "gelu"},
        }


# Readable aliases used by experiment manifests.
CNN_LSTM = CNNLSTM
GraphEEGNet = EEGGraphNet
