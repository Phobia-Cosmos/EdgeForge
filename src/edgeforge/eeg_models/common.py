"""Shared PyTorch-optional building blocks for EEG decoders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # Keep importing ``edgeforge`` possible in a dependency-free install.
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]

    class _UnavailableModule:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required for EEG decoder models")

    class _UnavailableNN:
        Module = _UnavailableModule

    nn = _UnavailableNN()  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def require_torch() -> Any:
    if torch is None:  # pragma: no cover - see import guard above
        raise RuntimeError("PyTorch is required for EEG decoder models")
    return torch


@dataclass
class EEGForwardBundle:
    """Model output plus named representations used by LoP diagnostics.

    ``logits`` is ``[B, K]`` for a regular epoch and ``[B, T, K]`` for a
    sequence of epochs.  Representations are detached only if the caller
    requests that explicitly; by default they retain the autograd graph so a
    probe can differentiate through a selected layer.
    """

    logits: Any
    representations: dict[str, Any] = field(default_factory=dict)
    attention: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def features(self) -> Any:
        """Return the preferred classifier representation when available."""

        for key in ("classifier_input", "embedding", "fusion", "tokens"):
            if key in self.representations:
                return self.representations[key]
        return self.logits


class EEGDecoderBase(nn.Module):
    """Base class implementing input flattening and bundle-compatible forward."""

    decoder_name = "base"

    def __init__(self) -> None:
        require_torch()
        super().__init__()

    @staticmethod
    def _flatten_epochs(values: Any) -> tuple[Any, tuple[int, ...] | None]:
        """Flatten ``[B,T,C,S]`` to ``[B*T,C,S]`` and remember its shape."""

        require_torch()
        if not isinstance(values, torch.Tensor):
            raise TypeError("EEG decoder input must be a torch.Tensor")
        if values.ndim == 3:
            return values, None
        if values.ndim == 4:
            batch, sequence, channels, samples = (int(item) for item in values.shape)
            return values.reshape(batch * sequence, channels, samples), (batch, sequence)
        raise ValueError(
            "EEG decoder input must have shape [B,C,S] or [B,T,C,S], "
            f"got {tuple(values.shape)}"
        )

    @staticmethod
    def _restore_epochs(values: Any, shape: tuple[int, ...] | None) -> Any:
        if shape is None:
            return values
        batch, sequence = shape
        return values.reshape(batch, sequence, *values.shape[1:])

    def forward_bundle(self, values: Any, *args: Any, **kwargs: Any) -> EEGForwardBundle:
        raise NotImplementedError

    def forward(self, values: Any, *args: Any, **kwargs: Any) -> Any:
        return self.forward_bundle(values, *args, **kwargs).logits

    def representation_specs(self) -> dict[str, dict[str, Any]]:
        """Return feature-axis hints consumed by LoP instrumentation.

        The hints are intentionally metadata, not assumptions in metric code:
        Conv1d maps use channel axis 1 and token embeddings use feature axis
        -1.  Sequence restoration is handled by each decoder.
        """

        return {}


class InspectableTransformerBlock(nn.Module):
    """Pre-norm Transformer block that exposes per-head key-axis attention."""

    def __init__(self, width: int, heads: int, dropout: float = 0.0, mlp_ratio: float = 4.0) -> None:
        require_torch()
        super().__init__()
        if width % heads:
            raise ValueError("transformer width must be divisible by heads")
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(width)
        hidden = max(width, int(round(width * mlp_ratio)))
        self.ffn = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width), nn.Dropout(dropout))
        self.last_attention: Any = None

    def forward(self, values: Any) -> Any:
        normalized = self.norm1(values)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attention = weights.detach()
        values = values + attended
        return values + self.ffn(self.norm2(values))


class BrainUICLAttentionBlock(nn.Module):
    """Attention block matching BrainUICL's optional head-axis softmax.

    BrainUICL's released ``MultiHeadAttentionBlock`` normalises its score
    tensor along dimension 1 (the head axis), unlike the usual key/token axis
    ``-1``.  The default ``normalization_axis=1`` preserves that behaviour;
    callers may set ``-1`` for a conventional Transformer.
    """

    def __init__(self, width: int, heads: int, dropout: float = 0.0, mlp_ratio: float = 4.0, normalization_axis: int = 1) -> None:
        require_torch()
        super().__init__()
        if width % heads:
            raise ValueError("transformer width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.normalization_axis = int(normalization_axis)
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(width)
        hidden = max(width, int(round(width * mlp_ratio)))
        self.ffn = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width), nn.Dropout(dropout))
        self.last_attention: Any = None

    def forward(self, values: Any) -> Any:
        torch_mod = require_torch()
        normalized = self.norm1(values)
        batch, tokens, _ = normalized.shape
        qkv = self.qkv(normalized).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scores = torch_mod.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        axis = self.normalization_axis
        if axis < 0:
            axis += scores.ndim
        if axis < 0 or axis >= scores.ndim:
            raise ValueError(f"attention normalization axis {self.normalization_axis} is invalid")
        probabilities = torch_mod.softmax(scores, dim=axis)
        self.last_attention = probabilities.detach()
        attended = torch_mod.matmul(self.dropout(probabilities), value).transpose(1, 2).reshape(batch, tokens, self.width)
        values = values + self.dropout(self.projection(attended))
        return values + self.ffn(self.norm2(values))


class EpochClassifierMixin:
    """Helpers shared by models whose body returns a pooled epoch feature."""

    def _bundle_from_pooled(self, pooled: Any, shape: tuple[int, ...] | None, representations: dict[str, Any], attention: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> EEGForwardBundle:
        logits = self.classifier(self.classifier_input(pooled))
        classifier_input = self.classifier_input_output if hasattr(self, "classifier_input_output") else None
        # ``classifier_input`` is set by concrete classes before this helper
        # is called.  Keep a fallback for custom subclasses.
        if classifier_input is not None:
            representations["classifier_input"] = classifier_input
        logits = self._restore_epochs(logits, shape)
        for key, value in list(representations.items()):
            representations[key] = self._restore_epochs(value, shape)
        return EEGForwardBundle(logits=logits, representations=representations, attention=attention or {}, metadata=metadata or {})

