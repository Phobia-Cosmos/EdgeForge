"""Standalone, configurable EEG decoder architectures.

All built-in models implement ``forward_bundle`` and retain ordinary
``nn.Module`` behaviour (``model(x)`` returns logits).  The bundle exposes
named intermediate representations and attention maps for LoP diagnostics.
"""

from .brainuicl import BrainUICL, BrainUICLDecoder, BrainUICLFrontend
from .common import BrainUICLAttentionBlock, EEGDecoderBase, EEGForwardBundle, InspectableTransformerBlock
from .config import EEGDecoderConfig, EEGModelConfig
from .convnets import DeepConvNet, DeepConvNetDecoder, ShallowConvNet, ShallowConvNetDecoder
from .eegnet import EEGNet, EEGNetDecoder
from .advanced import EEGConformer, FBCNet, LoPMLP
from .sequence import ATCNet, CNNLSTM, CNN_LSTM, EEGGraphNet, GraphEEGNet, TSception
from .registry import (
    EEGDecoderSpec,
    available_eeg_decoders,
    build_decoder,
    build_eeg_decoder,
    canonical_decoder_name,
    eeg_decoder_specs,
    register_eeg_decoder,
)
from .tcn import TCN, TCNDecoder, TemporalResidualBlock
from .transformer import EEGTransformer, TransformerDecoder

__all__ = [
    "EEGModelConfig",
    "EEGDecoderConfig",
    "EEGForwardBundle",
    "EEGDecoderBase",
    "EEGNet",
    "EEGNetDecoder",
    "TCN",
    "TCNDecoder",
    "TemporalResidualBlock",
    "EEGTransformer",
    "TransformerDecoder",
    "EEGConformer",
    "ShallowConvNet",
    "ShallowConvNetDecoder",
    "DeepConvNet",
    "DeepConvNetDecoder",
    "FBCNet",
    "LoPMLP",
    "TSception",
    "ATCNet",
    "CNNLSTM",
    "CNN_LSTM",
    "EEGGraphNet",
    "GraphEEGNet",
    "BrainUICL",
    "BrainUICLDecoder",
    "BrainUICLFrontend",
    "InspectableTransformerBlock",
    "BrainUICLAttentionBlock",
    "EEGDecoderSpec",
    "register_eeg_decoder",
    "build_eeg_decoder",
    "build_decoder",
    "available_eeg_decoders",
    "eeg_decoder_specs",
    "canonical_decoder_name",
]
