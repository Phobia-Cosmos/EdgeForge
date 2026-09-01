"""Registry and factory for the standalone EEG decoder components."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable, Mapping
from typing import Any

from .brainuicl import BrainUICLDecoder
from .config import EEGDecoderConfig, EEGModelConfig
from .convnets import DeepConvNet, ShallowConvNet
from .advanced import EEGConformer, FBCNet, LoPMLP
from .eegnet import EEGNet
from .sequence import ATCNet, CNNLSTM, EEGGraphNet, TSception
from .tcn import TCN
from .transformer import EEGTransformer


Builder = Callable[[EEGModelConfig], Any]


@dataclass(frozen=True)
class EEGDecoderSpec:
    """Human-readable registry entry."""

    name: str
    builder: Builder
    aliases: tuple[str, ...] = ()
    description: str = ""


_SPECS: dict[str, EEGDecoderSpec] = {}
_ALIASES: dict[str, str] = {}
_BUILTINS_REGISTERED = False


def register_eeg_decoder(
    name: str,
    builder: Builder,
    *,
    aliases: tuple[str, ...] | list[str] = (),
    description: str = "",
    overwrite: bool = False,
) -> EEGDecoderSpec:
    """Register a decoder class/factory.

    ``builder`` receives an :class:`EEGModelConfig` and must return a
    ``torch.nn.Module`` implementing ``forward_bundle``.  Registration is
    intentionally lightweight, allowing projects to add private EEG models
    without changing EdgeForge source files.
    """

    canonical = str(name).strip().lower()
    if not canonical:
        raise ValueError("decoder name must not be empty")
    if not callable(builder):
        raise TypeError("decoder builder must be callable")
    normalized_aliases = tuple(dict.fromkeys(str(alias).strip().lower() for alias in aliases if str(alias).strip()))
    if not overwrite and (canonical in _SPECS or canonical in _ALIASES):
        raise ValueError(f"EEG decoder {canonical!r} is already registered")
    for alias in normalized_aliases:
        if alias == canonical:
            continue
        if not overwrite and (alias in _SPECS or alias in _ALIASES):
            raise ValueError(f"EEG decoder alias {alias!r} is already registered")
    spec = EEGDecoderSpec(canonical, builder, normalized_aliases, description)
    _SPECS[canonical] = spec
    for alias in normalized_aliases:
        _ALIASES[alias] = canonical
    return spec


def _register_builtins() -> None:
    global _BUILTINS_REGISTERED
    # Custom decoders may be registered before the first factory call.  Using
    # ``if _SPECS`` here would accidentally suppress all built-ins in that
    # perfectly valid extension workflow, so keep an explicit initialization
    # flag instead.
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True
    try:
        register_eeg_decoder("eegnet", EEGNet, aliases=("eeg_net", "eegnet_decoder"), description="Depthwise/separable temporal EEGNet")
        register_eeg_decoder("tcn", TCN, aliases=("temporal_cnn", "tcn_decoder"), description="Dilated temporal convolutional network")
        register_eeg_decoder("transformer", EEGTransformer, aliases=("eeg_transformer", "transformer_decoder"), description="Patch-token Transformer with exposed attention")
        register_eeg_decoder("conformer", EEGConformer, aliases=("eeg_conformer",), description="Convolution-augmented EEG Transformer")
        register_eeg_decoder("shallowconvnet", ShallowConvNet, aliases=("shallow", "shallow_convnet", "shallowconv"), description="Square/log ShallowConvNet")
        register_eeg_decoder("deepconvnet", DeepConvNet, aliases=("deep", "deep_convnet", "deepconv"), description="Stacked ELU DeepConvNet")
        register_eeg_decoder("fbcnet", FBCNet, aliases=("filter_bank_cnn", "filterbank"), description="Multi-band filter-bank CNN")
        register_eeg_decoder("tsception", TSception, aliases=("tsception_net", "multi_scale_cnn"), description="Multi-scale temporal/spatial inception EEG decoder")
        register_eeg_decoder("atcnet", ATCNet, aliases=("attention_tcn", "eeg_atcnet"), description="Attention-temporal-convolution EEG decoder")
        register_eeg_decoder("cnn_lstm", CNNLSTM, aliases=("cnnlstm", "cnn-rnn", "eeg_cnn_lstm"), description="CNN plus recurrent LSTM context decoder")
        register_eeg_decoder("eeg_graph", EEGGraphNet, aliases=("graph_eeg", "graph_eegnet", "dgcnn"), description="Learnable-adjacency graph EEG decoder")
        register_eeg_decoder("lop_mlp", LoPMLP, aliases=("lop", "mlp", "brainuicl_lop_mlp", "brainuicl-lop-mlp", "brainuicl/lop_mlp"), description="Three-hidden-layer LoP MLP baseline")
        register_eeg_decoder(
            "brainuicl",
            BrainUICLDecoder,
            aliases=("brain_uicl", "brainuicl_decoder"),
            description="BrainUICL-style EEG/EOG frontend, attention encoder and MLP",
        )
    except Exception:
        # Leave the registry retryable if a future built-in registration is
        # edited incorrectly or collides with a user entry.
        _BUILTINS_REGISTERED = False
        raise


_register_builtins()


def canonical_decoder_name(name: str) -> str:
    """Resolve a canonical name or alias, raising a useful error otherwise."""

    _register_builtins()
    normalized = str(name).strip().lower()
    canonical = _ALIASES.get(normalized, normalized)
    if canonical not in _SPECS:
        available = ", ".join(available_eeg_decoders())
        raise ValueError(f"unknown EEG decoder {name!r}; available: {available}")
    return canonical


def available_eeg_decoders(*, include_aliases: bool = False) -> tuple[str, ...]:
    """Return stable registry names (or names plus aliases)."""

    _register_builtins()
    names = sorted(_SPECS)
    if include_aliases:
        names.extend(sorted(_ALIASES))
    return tuple(names)


def eeg_decoder_specs(*, include_aliases: bool = False) -> dict[str, EEGDecoderSpec]:
    """Return a copy of registry metadata suitable for manifests."""

    _register_builtins()
    result = dict(_SPECS)
    if include_aliases:
        result.update({alias: _SPECS[target] for alias, target in _ALIASES.items()})
    return result


def build_eeg_decoder(
    name: str | EEGModelConfig | EEGDecoderConfig | Mapping[str, Any] = "eegnet",
    config: EEGModelConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Build a registered EEG decoder from a name and/or config.

    Examples::

        build_eeg_decoder("eegnet", in_channels=8, num_classes=4)
        build_eeg_decoder(EEGModelConfig(name="tcn", width=32))
        build_eeg_decoder({"name": "brainuicl", "eog_channels": 2})
    """

    _register_builtins()
    if isinstance(name, EEGModelConfig):
        if config is not None:
            raise TypeError("config must be omitted when name is an EEGModelConfig")
        cfg = name
        requested = cfg.name
        if overrides:
            cfg = EEGModelConfig.from_mapping(cfg.to_dict(), **overrides)
    elif isinstance(name, Mapping):
        if config is not None:
            raise TypeError("config cannot be combined with a mapping name")
        requested = str(name.get("name", "eegnet"))
        cfg = EEGModelConfig.from_mapping(name, **overrides)
    else:
        if isinstance(config, EEGModelConfig):
            requested = config.name if not name or name == "eegnet" else str(name)
        elif isinstance(config, Mapping):
            requested = str(name or config.get("name", "eegnet"))
        else:
            requested = str(name or "eegnet")
        if isinstance(config, EEGModelConfig):
            cfg = config
            if overrides:
                cfg = EEGModelConfig.from_mapping(cfg.to_dict(), **overrides)
        else:
            values = dict(config or {}) if isinstance(config, Mapping) else {}
            values.setdefault("name", requested)
            cfg = EEGModelConfig.from_mapping(values, **overrides)
    canonical = canonical_decoder_name(requested or cfg.name)
    if cfg.name != canonical:
        cfg = replace(cfg, name=canonical)
    return _SPECS[canonical].builder(cfg)


# Friendly alias for callers that use the shorter factory name.
build_decoder = build_eeg_decoder
