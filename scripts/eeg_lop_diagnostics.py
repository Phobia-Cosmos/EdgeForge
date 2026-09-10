#!/usr/bin/env python3
"""Run a small, reproducible LoP diagnostic on EEG or image streams.

The script is intentionally self-contained and uses synthetic data by
default, so it can validate metric plumbing even when ISRUC/FACED payloads
are not mounted.  Replace ``make_stream`` and the model factory with an
adapter for real data; the output schema stays the same.

The experiment is a task-free domain stream: each task keeps the label space
fixed and changes the input domain.  After every task, the script records
layer-wise spectrum/activation/Jacobian/gradient/attention diagnostics and
probes the *next* task from both the current (warm) checkpoint and a fresh
initialization under the same finite budget.  The fresh-vs-warm gap is the
primary LoP outcome; all other metrics are predictors or mechanism
diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from edgeforge.lop_metrics import (  # noqa: E402
    activation_summary,
    attention_summary,
    capture_representations,
    fixed_budget_probe,
    gradient_summary,
    jacobian_summary,
    linear_cka,
    local_linearity_summary,
    parameter_norm_summary,
    parameter_spectral_summary,
    procrustes_residual,
    sampled_parameter_jacobian,
    spectral_summary,
)
from edgeforge.eeg_models import (  # noqa: E402
    EEGModelConfig,
    available_eeg_decoders,
    build_eeg_decoder,
    canonical_decoder_name,
)
from edgeforge.lop_diagnostics import (  # noqa: E402
    calibration_manifest_provenance,
    empirical_fisher_summary,
    exact_hessian_matrix,
    hessian_top_eigenvalue_summary,
    hutchinson_trace_summary,
    objective_loss,
)
from edgeforge.lop_envelope import MAX_METRICS, diagnostic_to_metrics  # noqa: E402


class MeanPool(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.mean(dim=-1)


class TokenMean(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.mean(dim=1)


class InspectableTransformerBlock(nn.Module):
    """Small standard key-axis attention block with exposed probabilities."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.last_attention: torch.Tensor | None = None

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(values)
        attended, weights = self.attention(normalized, normalized, normalized, need_weights=True, average_attn_weights=False)
        self.last_attention = weights.detach()
        values = values + attended
        return values + self.ffn(self.norm2(values))


class EEGTransformer(nn.Module):
    def __init__(self, channels: int, length: int, classes: int, width: int = 32, patch: int = 8, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("transformer width must be divisible by heads")
        self.patch_embed = nn.Conv1d(channels, width, kernel_size=patch, stride=patch)
        token_count = max(1, length // patch)
        self.position = nn.Parameter(torch.zeros(1, token_count, width))
        self.token_norm = nn.LayerNorm(width)
        self.encoder = nn.ModuleList([InspectableTransformerBlock(width, heads) for _ in range(layers)])
        self.pool = TokenMean()
        self.classifier_input = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.classifier = nn.Linear(width, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(values).transpose(1, 2)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        tokens = self.token_norm(tokens)
        for block in self.encoder:
            tokens = block(tokens)
        pooled = self.pool(tokens)
        features = self.classifier_input(pooled)
        return self.classifier(features)


class EEGNet(nn.Module):
    def __init__(self, channels: int, length: int, classes: int, width: int = 24) -> None:
        super().__init__()
        self.temporal_conv = nn.Conv1d(channels, width, kernel_size=15, padding=7, bias=False)
        self.temporal_norm = nn.BatchNorm1d(width)
        self.temporal_activation = nn.ReLU()
        self.depthwise_conv = nn.Conv1d(width, width, kernel_size=9, padding=4, groups=width, bias=False)
        self.pointwise_conv = nn.Conv1d(width, 2 * width, kernel_size=1, bias=False)
        self.separable_norm = nn.BatchNorm1d(2 * width)
        self.separable_activation = nn.ELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier_input = nn.Sequential(nn.Flatten(), nn.Linear(2 * width, width), nn.GELU())
        self.classifier = nn.Linear(width, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.temporal_activation(self.temporal_norm(self.temporal_conv(values)))
        values = self.depthwise_conv(values)
        values = self.separable_activation(self.separable_norm(self.pointwise_conv(values)))
        values = self.pool(values)
        return self.classifier(self.classifier_input(values))


class TCN(nn.Module):
    def __init__(self, channels: int, length: int, classes: int, width: int = 24) -> None:
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv1d(channels, width, 3, padding=1), nn.BatchNorm1d(width), nn.GELU())
        self.block2 = nn.Sequential(nn.Conv1d(width, width, 3, padding=2, dilation=2), nn.BatchNorm1d(width), nn.GELU())
        self.block3 = nn.Sequential(nn.Conv1d(width, width, 3, padding=4, dilation=4), nn.BatchNorm1d(width), nn.GELU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier_input = nn.Sequential(nn.Flatten(), nn.Linear(width, width), nn.GELU())
        self.classifier = nn.Linear(width, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.block1(values)
        values = self.block2(values)
        values = self.block3(values)
        return self.classifier(self.classifier_input(self.pool(values)))


class ImageTransformer(nn.Module):
    def __init__(self, classes: int, image_size: int = 16, width: int = 32, patch: int = 4, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("transformer width must be divisible by heads")
        self.patch_embed = nn.Conv2d(1, width, kernel_size=patch, stride=patch)
        tokens = max(1, (image_size // patch) ** 2)
        self.position = nn.Parameter(torch.zeros(1, tokens, width))
        self.encoder = nn.ModuleList([InspectableTransformerBlock(width, heads) for _ in range(layers)])
        self.pool = TokenMean()
        self.classifier_input = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.classifier = nn.Linear(width, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(values).flatten(2).transpose(1, 2)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        for block in self.encoder:
            tokens = block(tokens)
        return self.classifier(self.classifier_input(self.pool(tokens)))


class TinyCNN(nn.Module):
    def __init__(self, classes: int, width: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_input = nn.Sequential(nn.Flatten(), nn.Linear(2 * width, width), nn.GELU())
        self.classifier = nn.Linear(width, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.conv1(values)
        values = self.conv2(values)
        return self.classifier(self.classifier_input(self.pool(values)))


def _model_and_specs(
    name: str,
    *,
    data: str,
    channels: int,
    length: int,
    classes: int,
    image_size: int,
    model_payload: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, Callable[[], nn.Module], dict[str, Any]]:
    if data == "synthetic-eeg":
        # Synthetic EEG now uses the same registry as training/deployment.
        # Keep the image-only classes below local to this script so historical
        # image smoke tests remain bit-for-bit compatible.  The small widths
        # are deliberate: this command validates plumbing, not benchmark
        # accuracy or parameter-count parity.
        canonical = canonical_decoder_name(name)
        options: dict[str, Any] = {}
        if canonical == "brainuicl":
            # The real BrainUICL uses d_model=512 and ISRUC has an EEG+EOG
            # dual branch.  Synthetic EEG has one branch; use a compact model
            # while recording the architecture/attention semantics in its
            # forward bundle metadata.
            options = {"d_model": 32, "classifier_hidden": 32}
        elif canonical == "lop_mlp":
            options = {"hidden_dims": (64, 64, 64)}
        config_values: dict[str, Any] = {
            "name": canonical,
            "in_channels": int(channels),
            "eeg_channels": int(channels),
            "eog_channels": 0,
            "input_length": int(length),
            "num_classes": int(classes),
            "feature_dim": 32,
            "width": 24,
            "depth": 3,
            "kernel_size": max(3, min(15, int(length))),
            "patch_size": max(1, min(8, int(length))),
            "layers": 2,
            "heads": 4,
            "options": options,
        }
        # A JSON model manifest may override any registry field (including
        # architecture-specific ``options``).  CLI-derived dimensions above
        # remain the defaults and explicit CLI values are already reflected in
        # ``channels/length/classes`` before this merge.
        if isinstance(model_payload, Mapping):
            config_values.update({str(key): value for key, value in model_payload.items()})
        config_values["name"] = canonical
        # ``channels/length/classes`` are the resolved CLI values (possibly
        # supplied by the manifest).  Keep them authoritative so an explicit
        # short-window smoke cannot accidentally inherit ISRUC's 8-channel,
        # 3000-sample dimensions from the example manifest.  A dual-branch
        # declaration is retained only when its channel total still matches.
        config_values["in_channels"] = int(channels)
        config_values["input_length"] = int(length)
        config_values["num_classes"] = int(classes)
        eeg_declared = int(config_values.get("eeg_channels", channels))
        eog_declared = int(config_values.get("eog_channels", 0))
        if eeg_declared + eog_declared != int(channels):
            config_values["eeg_channels"] = int(channels)
            config_values["eog_channels"] = 0
        config = EEGModelConfig.from_mapping(config_values)

        def factory() -> nn.Module:
            # Rebuild from a serialized copy so a builder cannot accidentally
            # mutate the configuration shared by later fresh probes.
            return build_eeg_decoder(EEGModelConfig.from_mapping(config.to_dict()))

        model = factory()
        specs = dict(model.representation_specs())
        return model, factory, specs
    if data in {"synthetic-image", "digits"}:
        if name == "transformer":
            factory = lambda: ImageTransformer(classes, image_size=image_size)
            model = factory()
            specs = {
                "patch_embed": (1, "conv"),
                **{f"encoder.{index}": (-1, "transformer") for index in range(len(model.encoder))},
                "pool": (-1, "pool"),
                "classifier_input": (-1, "gelu"),
            }
            return model, factory, specs
        if name == "cnn":
            factory = lambda: TinyCNN(classes)
            model = factory()
            specs = {"conv1": (1, "relu"), "conv2": (1, "relu"), "pool": (1, "pool"), "classifier_input": (-1, "gelu")}
            return model, factory, specs
    raise ValueError(f"unsupported architecture {name!r} for {data}")


def _layer_map(model: nn.Module, specs: dict[str, Any]) -> dict[str, nn.Module]:
    modules = dict(model.named_modules())
    result: dict[str, nn.Module] = {}
    for name in specs:
        if name not in modules:
            raise KeyError(f"metric layer {name!r} is not present in model")
        result[name] = modules[name]
    return result


def _spec_axis_kind(spec: Any) -> tuple[int, str]:
    """Normalize legacy ``(axis, kind)`` and registry metadata specs."""

    if isinstance(spec, Mapping):
        return int(spec.get("feature_axis", -1)), str(spec.get("kind", "unknown"))
    if isinstance(spec, (tuple, list)) and len(spec) >= 2:
        return int(spec[0]), str(spec[1])
    raise TypeError(f"invalid representation spec: {spec!r}")


def _move_input(value: Any, device: torch.device) -> Any:
    """Move tensor, tuple/list, or mapping model inputs to a device."""

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_input(item, device) for item in value)
    if isinstance(value, list):
        return [_move_input(item, device) for item in value]
    if isinstance(value, Mapping):
        return {key: _move_input(item, device) for key, item in value.items()}
    return value


def _logits_from_output(output: Any) -> torch.Tensor:
    """Extract logits from a tensor or an EEGForwardBundle-like object."""

    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
        return output.logits
    if isinstance(output, Mapping):
        for key in ("logits", "output", "outputs", "prediction", "pred"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
    raise TypeError("model output does not contain a logits tensor")


def _classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute CE with the class dimension kept last for epoch sequences."""

    if logits.ndim > 2:
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
    if labels.ndim > 1 and labels.shape[-1] == logits.shape[-1]:
        # Permit one-hot labels in small custom streams while retaining the
        # integer-label path used by the synthetic generators.
        labels = labels.argmax(dim=-1)
    return nn.functional.cross_entropy(logits, labels.long())


def _diagnostic_objective_loss(
    model: nn.Module,
    values: Any,
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    *,
    objective: str,
    label_source: str,
) -> torch.Tensor:
    """Use the shared LoP objective while keeping pseudo labels explicit."""

    source = str(label_source).lower().strip()
    # The generic stream keeps oracle labels for evaluation.  They must not be
    # consumed when the diagnostic explicitly requests model-derived labels.
    objective_labels = None if source == "pseudo" else labels
    loss = objective_loss(
        model,
        (values, objective_labels),
        logits,
        objective_labels,
        objective=objective,
        label_source=source,
    )
    if isinstance(loss, torch.Tensor):
        return loss
    return torch.as_tensor(loss, dtype=logits.dtype, device=logits.device)


def _capture_bundle(
    model: nn.Module,
    calibration_batches: list[tuple[torch.Tensor, torch.Tensor]],
    names: list[str],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor | None], dict[str, Any]]:
    """Capture semantic registry taps and attention maps in one forward pass."""

    collected: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    attention_values: dict[str, list[torch.Tensor]] = {}
    attention_axis: int | None = None
    model_was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for values, _labels in calibration_batches:
                bundle = model.forward_bundle(_move_input(values, device))
                reps = getattr(bundle, "representations", {})
                if isinstance(reps, dict):
                    for name in names:
                        value = reps.get(name)
                        if isinstance(value, torch.Tensor):
                            collected[name].append(value.detach().cpu())
                attention = getattr(bundle, "attention", {})
                if isinstance(attention, dict):
                    for name, value in attention.items():
                        if isinstance(value, torch.Tensor):
                            attention_values.setdefault(str(name), []).append(value.detach().cpu())
                metadata = getattr(bundle, "metadata", {})
                if isinstance(metadata, dict) and "attention_normalization_axis" in metadata:
                    attention_axis = int(metadata["attention_normalization_axis"])
    finally:
        model.train(model_was_training)
    representations = {
        name: (torch.cat(values, dim=0) if values else None)
        for name, values in collected.items()
    }
    attention_payload = {
        name: torch.cat(values, dim=0)
        for name, values in attention_values.items()
        if values
    }
    return representations, {"attention": attention_payload, "attention_axis": attention_axis}


def _make_batches(values: torch.Tensor, labels: torch.Tensor, batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(x, y) for x, y in zip(values.split(batch_size), labels.split(batch_size))]


def _make_stream(args: argparse.Namespace) -> list[tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]]:
    generator = torch.Generator().manual_seed(args.seed)
    stream = []
    if args.data == "synthetic-eeg":
        time = torch.linspace(0.0, 1.0, args.length)
        prototypes = []
        for label in range(args.classes):
            signal = []
            for channel in range(args.channels):
                frequency = 2.0 + label * 0.8 + channel * 0.11
                phase = 0.3 * channel
                signal.append(torch.sin(2 * math.pi * frequency * time + phase) + 0.35 * torch.cos(2 * math.pi * (frequency + 0.7) * time))
            prototypes.append(torch.stack(signal))
        prototypes = torch.stack(prototypes)
        for task in range(args.tasks):
            def make(count: int) -> tuple[torch.Tensor, torch.Tensor]:
                labels = torch.randint(args.classes, (count,), generator=generator)
                values = prototypes[labels].clone()
                scale = 1.0 + 0.10 * task
                drift = 0.20 * task * torch.sin(2 * math.pi * (0.5 + 0.03 * task) * time)
                values = scale * values + drift
                values = values + 0.18 * torch.randn(values.shape, generator=generator)
                return values.float(), labels.long()
            train = make(args.train_samples)
            evaluation = make(args.eval_samples)
            stream.append((_make_batches(*train, args.batch_size), _make_batches(*evaluation, args.batch_size)))
        return stream

    if args.data == "digits":
        try:
            from sklearn.datasets import load_digits
        except ImportError as error:
            raise RuntimeError("--data digits requires scikit-learn in the selected environment") from error
        digits = load_digits()
        values = torch.from_numpy(digits.images).float().unsqueeze(1) / 16.0
        labels = torch.from_numpy(digits.target).long()
        keep = labels < args.classes
        values, labels = values[keep], labels[keep]
        if args.image_size != 8:
            values = torch.nn.functional.interpolate(values, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
        stream = []
        for task in range(args.tasks):
            def make(count: int) -> tuple[torch.Tensor, torch.Tensor]:
                indices = torch.randint(len(values), (count,), generator=generator)
                selected_values = values[indices].clone()
                selected_labels = labels[indices].clone()
                selected_values = torch.rot90(selected_values, task % 4, dims=(-2, -1))
                selected_values = (1.0 + 0.08 * task) * selected_values + 0.02 * task
                selected_values = selected_values + 0.08 * torch.randn(selected_values.shape, generator=generator)
                return selected_values.clamp(0.0, 1.0), selected_labels
            train = make(args.train_samples)
            evaluation = make(args.eval_samples)
            stream.append((_make_batches(*train, args.batch_size), _make_batches(*evaluation, args.batch_size)))
        return stream

    grid = torch.linspace(-1.0, 1.0, args.image_size)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    prototypes = []
    for label in range(args.classes):
        if label % 4 == 0:
            pattern = (xx > 0).float() - (xx <= 0).float()
        elif label % 4 == 1:
            pattern = (yy > 0).float() - (yy <= 0).float()
        elif label % 4 == 2:
            pattern = torch.sign(xx * yy)
        else:
            pattern = torch.sign(xx + yy)
        prototypes.append(pattern)
    prototypes = torch.stack(prototypes).unsqueeze(1)
    for task in range(args.tasks):
        def make(count: int) -> tuple[torch.Tensor, torch.Tensor]:
            labels = torch.randint(args.classes, (count,), generator=generator)
            values = prototypes[labels].clone()
            values = (1.0 + 0.10 * task) * values + 0.10 * task * xx[None, None]
            values = values + 0.20 * torch.randn(values.shape, generator=generator)
            return values.float(), labels.long()
        train = make(args.train_samples)
        evaluation = make(args.eval_samples)
        stream.append((_make_batches(*train, args.batch_size), _make_batches(*evaluation, args.batch_size)))
    return stream


def _train_task(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]], *, epochs: int, lr: float, device: torch.device) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for _ in range(epochs):
        model.train()
        for values, labels in batches:
            values, labels = values.to(device), labels.to(device)
            loss = _classification_loss(_logits_from_output(model(values)), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def _collect_gradients(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    max_batches: int = 4,
    objective: str = "cross_entropy",
    label_source: str = "true",
) -> torch.Tensor:
    rows = []
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    was_training = bool(model.training)
    model.eval()
    try:
        for values, labels in batches[:max_batches]:
            values, labels = values.to(device), labels.to(device)
            model.zero_grad(set_to_none=True)
            logits = _logits_from_output(model(values))
            loss = _diagnostic_objective_loss(
                model,
                values,
                logits,
                labels,
                objective=objective,
                label_source=label_source,
            )
            loss.backward()
            rows.append(torch.cat([(parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).detach().float().reshape(-1) for parameter in parameters]).cpu())
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    return torch.stack(rows) if rows else torch.empty((0, sum(int(parameter.numel()) for parameter in parameters)))


def _attention_metrics(
    model: nn.Module,
    *,
    captured: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize exposed attention and preserve the implementation axis."""

    if captured is not None:
        values = captured.get("attention", {})
        axis = captured.get("attention_axis")
        if axis is None:
            axis = -1
        if values:
            records = []
            for name, attention in values.items():
                try:
                    records.append({"layer": name, **attention_summary(attention, normalization_axis=int(axis))})
                except Exception as error:
                    records.append({"layer": name, "status": "error", "error": f"{type(error).__name__}: {error}"})
            return {
                "status": "computed",
                "layers": records,
                "normalization_axis": int(axis),
                "source": "forward_bundle",
            }
        return {
            "status": "unavailable",
            "reason": "model did not expose attention weights",
            "normalization_axis": int(axis),
            "source": "forward_bundle",
        }
    records = []
    for index, block in enumerate(getattr(model, "encoder", [])):
        attention = getattr(block, "last_attention", None)
        if attention is not None:
            records.append({"layer": index, **attention_summary(attention, normalization_axis=-1)})
    return {"status": "computed", "layers": records, "normalization_axis": -1} if records else {"status": "unavailable", "reason": "model has no exposed attention weights"}


def _stage_metrics(
    model: nn.Module,
    calibration_batches: list[tuple[torch.Tensor, torch.Tensor]],
    specs: dict[str, Any],
    *,
    device: torch.device,
    max_observations: int,
    previous_representations: dict[str, torch.Tensor] | None,
    seed: int,
    hessian_mode: str = "none",
    hessian_probes: int = 2,
    hessian_iterations: int = 8,
    hessian_tolerance: float = 1e-5,
    hessian_max_params: int = 256,
    fisher_samples: int = 0,
    objective: str = "cross_entropy",
    label_source: str = "true",
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.to(device)
    cpu_batches = [(values, labels) for values, labels in calibration_batches]
    # Registry decoders expose semantic taps through ``forward_bundle``;
    # legacy image models still use module hooks.  This boundary lets the
    # diagnostic code remain independent of concrete model internals.
    semantic_specs = bool(specs) and all(isinstance(value, Mapping) for value in specs.values()) and hasattr(model, "forward_bundle")
    bundle_capture: dict[str, Any] | None = None
    if semantic_specs:
        representations, bundle_capture = _capture_bundle(
            model,
            cpu_batches,
            [str(name) for name in specs],
            device=device,
        )
    else:
        layer_map = _layer_map(model, specs)
        representations = capture_representations(
            model,
            [values.to(device) for values, _labels in cpu_batches],
            layer_map,
            forward_fn=lambda module, values: module(values),
            max_batches=len(cpu_batches),
        )
    # Registry EEG decoders restore bundle taps to ``[B,T,...]`` after
    # flattening epochs internally.  Keep the channel feature axis explicit
    # for sequence inputs; token/embedding specs already use ``-1`` and are
    # unaffected.  A caller can opt out with ``layout=flattened_epoch_map``.
    sequence_input = False
    if cpu_batches:
        first_input = cpu_batches[0][0]
        first_tensor = first_input
        if isinstance(first_input, (tuple, list)):
            first_tensor = next((item for item in first_input if isinstance(item, torch.Tensor)), None)
        if isinstance(first_tensor, torch.Tensor) and first_tensor.ndim == 4:
            sequence_input = True
    metrics: dict[str, Any] = {
        "layers": {},
        "attention": _attention_metrics(model, captured=bundle_capture),
        "parameters": parameter_norm_summary(model),
        "parameter_spectra": parameter_spectral_summary(model),
        "objective": str(objective),
        "label_source": str(label_source),
    }
    for name, spec in specs.items():
        axis, kind = _spec_axis_kind(spec)
        axis_adjusted = False
        if isinstance(spec, Mapping) and sequence_input and axis >= 0 and spec.get("layout") != "flattened_epoch_map" and spec.get("sequence_axis_policy") in {"prepend", "restored"}:
            axis += 1
            axis_adjusted = True
        values = representations.get(name)
        if values is None:
            metrics["layers"][name] = {"status": "unavailable"}
            continue
        item = {
            "spectrum": spectral_summary(values, feature_axis=axis, max_observations=max_observations),
            "activation": activation_summary(values, kind=kind),
            "feature_axis": axis,
            "sequence_axis_adjusted": axis_adjusted,
        }
        if previous_representations and previous_representations.get(name) is not None:
            previous = previous_representations[name]
            item["drift"] = {
                "cka": linear_cka(values, previous, feature_axis=axis, max_observations=max_observations),
                "procrustes_residual": procrustes_residual(values, previous, feature_axis=axis, max_observations=max_observations),
            }
        metrics["layers"][name] = item
    first_values = _move_input(cpu_batches[0][0], device)
    first_count = int(first_values[0].shape[0]) if isinstance(first_values, (tuple, list)) else int(first_values.shape[0])
    forward_tensor = lambda module, item: _logits_from_output(module(item))
    jacobian = sampled_parameter_jacobian(
        model,
        first_values,
        forward_fn=forward_tensor,
        max_samples=min(4, first_count),
    )
    metrics["jacobian"] = jacobian_summary(jacobian)
    metrics["local_linearity"] = local_linearity_summary(
        model,
        first_values,
        forward_fn=forward_tensor,
        epsilons=(1e-3, 1e-2),
        directions=2,
        seed=seed,
    )
    gradients = _collect_gradients(
        model,
        cpu_batches,
        device=device,
        objective=objective,
        label_source=label_source,
    )
    metrics["gradient"] = gradient_summary(gradients)
    metrics["gradient"]["objective"] = str(objective)
    metrics["gradient"]["label_source"] = str(label_source)
    if int(fisher_samples) > 0:
        try:
            metrics["fisher"] = empirical_fisher_summary(
                model,
                cpu_batches,
                objective=objective,
                label_source=label_source,
                max_samples=int(fisher_samples),
            )
        except Exception as error:
            metrics["fisher"] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    else:
        metrics["fisher"] = {"status": "disabled"}
    if hessian_mode != "none":
        # Curvature is intentionally opt-in: even matrix-free HVPs can be
        # expensive for a full BrainUICL checkpoint.  The objective and
        # labels are explicit so the report cannot be mistaken for a
        # structure-only "network Hessian".
        try:
            curvature: dict[str, Any] = {
                "status": "computed",
                "mode": str(hessian_mode),
                "objective": str(objective),
                "label_source": str(label_source),
                "data_dependent": True,
                "batch_count": len(cpu_batches),
                "batch_norm_policy": "eval/frozen during probe",
            }
            if hessian_mode in {"hvp", "power", "all"}:
                top = hessian_top_eigenvalue_summary(
                    model,
                    cpu_batches,
                    objective=objective,
                    label_source=label_source,
                    iterations=hessian_iterations,
                    tolerance=hessian_tolerance,
                    seed=seed,
                )
                curvature["top_eigenvalue"] = top["eigenvalue"]
                curvature["top_eigenvalue_summary"] = top
            if hessian_mode in {"hvp", "hutchinson", "all"}:
                trace = hutchinson_trace_summary(
                    model,
                    cpu_batches,
                    probes=hessian_probes,
                    objective=objective,
                    label_source=label_source,
                    seed=seed,
                )
                curvature["trace"] = trace["trace"]
                curvature["trace_summary"] = trace
            if hessian_mode in {"exact", "all"}:
                parameter_count = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
                if parameter_count <= int(hessian_max_params):
                    matrix = exact_hessian_matrix(
                        model,
                        cpu_batches,
                        objective=objective,
                        label_source=label_source,
                        max_parameters=int(hessian_max_params),
                    )
                    eigenvalues = torch.linalg.eigvalsh((matrix + matrix.T) / 2.0)
                    curvature["exact"] = {
                        "status": "computed",
                        "shape": [int(item) for item in matrix.shape],
                        "trace": float(torch.trace(matrix).item()),
                        "top_eigenvalue": float(eigenvalues.max().item()) if eigenvalues.numel() else 0.0,
                        "min_eigenvalue": float(eigenvalues.min().item()) if eigenvalues.numel() else 0.0,
                    }
                else:
                    curvature["exact"] = {
                        "status": "skipped",
                        "reason": "parameter_count_exceeds_limit",
                        "parameter_count": parameter_count,
                        "max_parameters": int(hessian_max_params),
                    }
            metrics["hessian"] = curvature
        except Exception as error:
            metrics["hessian"] = {
                "status": "error",
                "mode": str(hessian_mode),
                "error": f"{type(error).__name__}: {error}",
            }
    else:
        metrics["hessian"] = {"status": "disabled", "mode": "none"}
    return metrics, {name: values for name, values in representations.items() if values is not None}


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# EEG/Image LoP diagnostics",
        "",
        f"- Data: `{result['config']['data']}`; architectures: `{', '.join(result['config']['architectures'])}`",
        f"- Tasks: {result['config']['tasks']}; seeds: `{result['config']['seed']}`; scientific conclusion allowed: `false`",
        f"- Calibration manifest: `{result['config'].get('calibration_manifest', {}).get('status', 'not-provided')}`; SHA-256: `{result['config'].get('calibration_manifest', {}).get('sha256') or '—'}`",
        f"- Diagnostic objective: `{result['config'].get('objective', 'cross_entropy')}`; label source: `{result['config'].get('label_source', 'true')}`",
        "- Primary outcome: `fresh_gap_final = accuracy_fresh - accuracy_warm`; positive values mean the fresh model adapted better.",
        "- Spectrum, activation, Jacobian, gradient and attention values are diagnostics, not causal evidence.",
        "",
        "## Fixed-budget outcomes",
        "",
        "| architecture | stage | next task | fresh final gap | fresh AULC gap | warm acc gain | fresh acc gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in result["runs"]:
        for probe in run.get("probes", []):
            outcome = probe["outcome"]
            lines.append(
                f"| {run['architecture']} | {probe['after_stage']} | {probe['next_task']} | {outcome['fresh_gap_final']:+.6f} | {outcome['fresh_auc_gap']:+.6f} | {outcome['warm_acc_gain']:+.6f} | {outcome['fresh_acc_gain']:+.6f} |"
            )
    lines.extend([
        "",
        "## Layer diagnostics at each stage",
        "",
        "| architecture | stage | layer | shape | effective rank | normalized ER | stable rank | near-zero | CKA to previous |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for run in result["runs"]:
        for stage in run["stages"]:
            for layer, item in stage["metrics"]["layers"].items():
                spectrum = item.get("spectrum", {})
                activation = item.get("activation", {})
                drift = item.get("drift", {})
                cka = drift.get("cka")
                fmt = lambda value: "—" if value is None else f"{float(value):.6g}"
                lines.append(
                    f"| {run['architecture']} | {stage['stage']} | {layer} | `{spectrum.get('shape', [])}` | {fmt(spectrum.get('effective_rank'))} | {fmt(spectrum.get('effective_rank_normalized'))} | {fmt(spectrum.get('stable_rank'))} | {fmt(activation.get('near_zero_fraction'))} | {fmt(cka)} |"
                )
    lines.extend(["", "The JSON file contains the complete curves and provenance.", ""])
    return "\n".join(lines)


def _manifest_cli_defaults(path: Path | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load optional JSON experiment config and translate stable CLI fields.

    The model/diagnostics sections intentionally remain separate.  Unknown
    fields are preserved in the returned model payload and consumed by the
    registry config, while only documented scalar controls become argparse
    defaults.  This lets explicit command-line flags override a manifest
    without making the script depend on a YAML package.
    """

    if path is None:
        return {}, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"unable to read --config {path}: {type(error).__name__}: {error}") from error
    if not isinstance(payload, Mapping):
        raise SystemExit("--config must contain a JSON object")
    model = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), Mapping) else {}
    defaults: dict[str, Any] = {}
    if model.get("name") is not None:
        defaults["architectures"] = str(model["name"])
    for source, target in {
        "in_channels": "channels",
        "input_length": "length",
        "num_classes": "classes",
    }.items():
        if model.get(source) is not None:
            defaults[target] = int(model[source])
    for source, target in {
        "max_observations": "max_observations",
        "max_metrics": "max_metrics",
        "jacobian_samples": "jacobian_samples",
        "hessian_probes": "hessian_probes",
        "hessian_iterations": "hessian_iterations",
        "hessian_tolerance": "hessian_tolerance",
        "hessian_max_params": "hessian_max_params",
        "fisher_samples": "fisher_samples",
        "seed": "seed",
    }.items():
        if diagnostics.get(source) is not None:
            defaults[target] = diagnostics[source]
    objective_aliases = {
        "ce": "cross_entropy",
        "cross-entropy": "cross_entropy",
        "negative_log_likelihood": "cross_entropy",
        "mse_to_zero": "mse_to_zero",
        "mean_squared_error": "mse",
        "squared_output": "mse_to_zero",
        "mean_output": "output_mean",
        "norm": "output_norm",
    }
    label_aliases = {
        "ground_truth": "true",
        "ground-truth": "true",
        "pseudo_label": "pseudo",
        "pseudo-label": "pseudo",
        "unlabeled": "none",
    }
    objective = diagnostics.get("objective")
    if isinstance(objective, Mapping):
        if objective.get("name", objective.get("loss")) is not None:
            raw_objective = str(objective.get("name", objective.get("loss"))).lower().strip()
            defaults["objective"] = objective_aliases.get(raw_objective, raw_objective)
        if objective.get("label_source") is not None:
            raw_source = str(objective["label_source"]).lower().strip()
            defaults["label_source"] = label_aliases.get(raw_source, raw_source)
    elif objective is not None:
        raw_objective = str(objective).lower().strip()
        defaults["objective"] = objective_aliases.get(raw_objective, raw_objective)
    if diagnostics.get("label_source") is not None:
        raw_source = str(diagnostics["label_source"]).lower().strip()
        defaults["label_source"] = label_aliases.get(raw_source, raw_source)
    if diagnostics.get("calibration_manifest") is not None:
        defaults["calibration_manifest"] = Path(str(diagnostics["calibration_manifest"]))
    if diagnostics.get("calibration_manifest_digest") is not None:
        defaults["calibration_manifest_digest"] = str(diagnostics["calibration_manifest_digest"])
    hessian = diagnostics.get("hessian")
    if isinstance(hessian, Mapping):
        for source, target in {
            "mode": "hessian_mode",
            "probes": "hessian_probes",
            "iterations": "hessian_iterations",
            "tolerance": "hessian_tolerance",
        }.items():
            if hessian.get(source) is not None:
                defaults[target] = hessian[source]
    fresh_warm = diagnostics.get("fresh_warm")
    if isinstance(fresh_warm, Mapping) and fresh_warm.get("budgets") is not None:
        budgets = fresh_warm["budgets"]
        if isinstance(budgets, (list, tuple)) and budgets:
            defaults["probe_steps"] = ",".join(str(int(value)) for value in budgets)
    return defaults, dict(model) if model else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Pre-parse only --config so its values can become parser defaults; any
    # explicit flag in ``argv`` still wins during the real parse.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _unknown = pre_parser.parse_known_args(argv)
    config_defaults, model_payload = _manifest_cli_defaults(pre_args.config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=pre_args.config, help="optional JSON model/diagnostics manifest")
    parser.add_argument("--data", choices=("synthetic-eeg", "synthetic-image", "digits"), default="synthetic-eeg")
    parser.add_argument(
        "--architectures",
        default="all",
        help=(
            "comma-separated EEG registry names (eegnet,tcn,transformer,conformer,"
            "shallowconvnet,deepconvnet,fbcnet,tsception,atcnet,cnn_lstm,eeg_graph,"
            "lop_mlp,brainuicl); image uses transformer,cnn"
        ),
    )
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=96)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--probe-lr", type=float, default=2e-3)
    parser.add_argument("--probe-steps", default="0,5,10,20")
    parser.add_argument(
        "--objective",
        choices=("cross_entropy", "mse", "mse_to_zero", "output_mean", "output_norm"),
        default="cross_entropy",
        help="objective used by gradient/Hessian/Fisher diagnostics",
    )
    parser.add_argument(
        "--label-source",
        choices=("true", "pseudo", "none"),
        default="true",
        help="label source for the diagnostic objective; pseudo derives argmax labels from logits",
    )
    parser.add_argument("--max-observations", type=int, default=256)
    parser.add_argument(
        "--max-metrics",
        type=int,
        default=MAX_METRICS,
        help="maximum flat envelope rows; primary predictor/outcome/retention rows are retained first",
    )
    parser.add_argument("--hessian-mode", choices=("none", "hvp", "power", "hutchinson", "exact", "all"), default="none", help="optional curvature probe; none keeps the smoke run fast")
    parser.add_argument("--hessian-probes", type=int, default=2)
    parser.add_argument("--hessian-iterations", type=int, default=8)
    parser.add_argument("--hessian-tolerance", type=float, default=1e-5)
    parser.add_argument("--hessian-max-params", type=int, default=256)
    parser.add_argument(
        "--fisher-samples",
        type=int,
        default=0,
        help="optional per-sample empirical-Fisher probe; 0 keeps the smoke run fast",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=None,
        help="optional manifest path; only its bytes are hashed for provenance",
    )
    parser.add_argument(
        "--calibration-manifest-digest",
        default=None,
        help="optional externally declared SHA-256 (with or without sha256: prefix)",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs" / "lop-diagnostics")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)
    args.model_payload = model_payload
    args.config_defaults = config_defaults
    return args


def main() -> None:
    args = parse_args()
    if args.tasks < 2 or args.epochs < 1 or args.batch_size < 1 or args.max_metrics < 1:
        raise SystemExit("tasks must be >=2, epochs and batch-size must be positive")
    requested = [int(item) for item in args.probe_steps.split(",") if item.strip()]
    if not requested or requested[0] != 0 or requested[-1] <= 0:
        raise SystemExit("probe-steps must include 0 and a positive step")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    available = available_eeg_decoders() if args.data == "synthetic-eeg" else ("transformer", "cnn")
    if args.architectures == "all":
        architectures = available
    elif args.data == "synthetic-eeg":
        try:
            architectures = tuple(canonical_decoder_name(item.strip()) for item in args.architectures.split(",") if item.strip())
        except ValueError as error:
            raise SystemExit(str(error)) from error
    else:
        architectures = tuple(item.strip() for item in args.architectures.split(",") if item.strip())
    if not architectures or any(item not in available for item in architectures):
        raise SystemExit(f"architectures must be drawn from {', '.join(available)}")
    stream = _make_stream(args)
    runs = []
    for architecture_index, architecture in enumerate(architectures):
        torch.manual_seed(args.seed + architecture_index * 1000)
        model, factory, specs = _model_and_specs(
            architecture,
            data=args.data,
            channels=args.channels,
            length=args.length,
            classes=args.classes,
            image_size=args.image_size,
            model_payload=(
                getattr(args, "model_payload", None)
                if not isinstance(getattr(args, "model_payload", None), Mapping)
                or str(getattr(args, "model_payload", {}).get("name", architecture)).strip().lower() == architecture
                else None
            ),
        )
        model.to(device)
        stages = []
        probes = []
        previous_representations = None
        fixed_anchor = stream[0][1]
        for task_index, (train_batches, eval_batches) in enumerate(stream):
            _train_task(model, train_batches, epochs=args.epochs, lr=args.lr, device=device)
            metrics, current_representations = _stage_metrics(
                model,
                eval_batches,
                specs,
                device=device,
                max_observations=args.max_observations,
                previous_representations=previous_representations,
                seed=args.seed + task_index,
                hessian_mode=args.hessian_mode,
                hessian_probes=args.hessian_probes,
                hessian_iterations=args.hessian_iterations,
                hessian_tolerance=args.hessian_tolerance,
                hessian_max_params=args.hessian_max_params,
                fisher_samples=args.fisher_samples,
                objective=args.objective,
                label_source=args.label_source,
            )
            # The anchor is fixed so CKA/drift describe the same data rather
            # than conflating task shift with representation movement.
            anchor_metrics, anchor_representations = _stage_metrics(
                model,
                fixed_anchor,
                specs,
                device=device,
                max_observations=args.max_observations,
                previous_representations=previous_representations,
                seed=args.seed + 10000 + task_index,
                hessian_mode=args.hessian_mode,
                hessian_probes=args.hessian_probes,
                hessian_iterations=args.hessian_iterations,
                hessian_tolerance=args.hessian_tolerance,
                hessian_max_params=args.hessian_max_params,
                fisher_samples=args.fisher_samples,
                objective=args.objective,
                label_source=args.label_source,
            )
            # The drift comparison must use the fixed anchor data at both
            # checkpoints.  Copy only that same-data drift into the main row;
            # task-specific evaluation tensors remain available separately.
            if previous_representations is not None:
                for name, item in metrics["layers"].items():
                    anchor_item = anchor_metrics["layers"].get(name, {})
                    if isinstance(anchor_item, dict) and "drift" in anchor_item:
                        item["drift"] = anchor_item["drift"]
            stages.append({"stage": task_index, "task": task_index, "metrics": metrics, "anchor_metrics": anchor_metrics})
            if task_index + 1 < len(stream):
                next_train, next_eval = stream[task_index + 1]
                torch.manual_seed(args.seed + 50000 + architecture_index * 1000 + task_index)
                fresh = factory()
                probe = fixed_budget_probe(
                    model,
                    fresh,
                    next_train,
                    next_eval,
                    steps=requested,
                    lr=args.probe_lr,
                    seed=args.seed + 60000 + architecture_index * 1000 + task_index,
                )
                probes.append({"after_stage": task_index, "next_task": task_index + 1, **probe})
            # Compare later checkpoints on the same fixed anchor batch.  Using
            # each task's own evaluation stream here would mix domain shift
            # with representation drift and make CKA uninterpretable.
            previous_representations = anchor_representations
        model_config = getattr(model, "config", None)
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()
        runs.append({
            "architecture": architecture,
            "model_config": model_config,
            "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
            "stages": stages,
            "probes": probes,
        })

    result = {
        "schema_version": 1,
        "protocol": "generic-lop-eeg-image-diagnostics-v1",
        # ``metrics`` is an edgeforge-bundle-v1 compatible envelope.  The
        # richer ``runs`` tree remains alongside it for human inspection and
        # does not need to be re-normalized by the Worker importer.
        "scientific_conclusion_allowed": False,
        "config": {
            "data": args.data,
            "architectures": list(architectures),
            "tasks": args.tasks,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "channels": args.channels,
            "length": args.length,
            "image_size": args.image_size,
            "classes": args.classes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "probe_steps": requested,
            "objective": args.objective,
            "label_source": args.label_source,
            "max_metrics": args.max_metrics,
            "hessian_mode": args.hessian_mode,
            "hessian_probes": args.hessian_probes,
            "hessian_iterations": args.hessian_iterations,
            "hessian_tolerance": args.hessian_tolerance,
            "fisher_samples": args.fisher_samples,
            "calibration_manifest": calibration_manifest_provenance(
                args.calibration_manifest,
                declared_digest=args.calibration_manifest_digest,
            ),
            "config_path": str(args.config) if args.config is not None else None,
            "model_manifest": getattr(args, "model_payload", None),
            "seed": args.seed,
            "device": str(device),
        },
        "runs": runs,
    }
    result["metrics"] = diagnostic_to_metrics(result, max_metrics=args.max_metrics)
    result["envelope"] = {
        "schema": "edgeforge-bundle-v1",
        "metric_count": len(result["metrics"]),
        "max_metrics": args.max_metrics,
        "possibly_truncated": len(result["metrics"]) >= args.max_metrics,
        "scientific_conclusion_allowed": False,
        "roles": {
            "predictor": "rank-like spectrum rows (effective/stable rank)",
            "outcome": "fixed-budget fresh-vs-warm plasticity rows",
            "retention": "not collected by this generic stream unless supplied by an adapter",
            "diagnostic": "Jacobian/NTK, gradient, Hessian, Fisher, activation, attention and parameter rows",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.data}-lop-diagnostics.json"
    markdown_path = args.output_dir / f"{args.data}-lop-diagnostics.md"
    json_path.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "architectures": list(architectures), "scientific_conclusion_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
