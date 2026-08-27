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
    procrustes_residual,
    sampled_parameter_jacobian,
    spectral_summary,
)


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


def _model_and_specs(name: str, *, data: str, channels: int, length: int, classes: int, image_size: int) -> tuple[nn.Module, Callable[[], nn.Module], dict[str, tuple[int, str]]]:
    if data == "synthetic-eeg":
        if name == "transformer":
            factory = lambda: EEGTransformer(channels, length, classes)
            model = factory()
            specs = {
                "patch_embed": (1, "conv"),
                "token_norm": (-1, "layernorm"),
                **{f"encoder.{index}": (-1, "transformer") for index in range(len(model.encoder))},
                "pool": (-1, "pool"),
                "classifier_input": (-1, "gelu"),
            }
            return model, factory, specs
        if name == "eegnet":
            factory = lambda: EEGNet(channels, length, classes)
            model = factory()
            specs = {
                "temporal_conv": (1, "conv"),
                "temporal_activation": (1, "relu"),
                "depthwise_conv": (1, "conv"),
                "separable_activation": (1, "elu"),
                "pool": (1, "pool"),
                "classifier_input": (-1, "gelu"),
            }
            return model, factory, specs
        if name == "tcn":
            factory = lambda: TCN(channels, length, classes)
            model = factory()
            specs = {
                "block1": (1, "gelu"),
                "block2": (1, "gelu"),
                "block3": (1, "gelu"),
                "pool": (1, "pool"),
                "classifier_input": (-1, "gelu"),
            }
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


def _layer_map(model: nn.Module, specs: dict[str, tuple[int, str]]) -> dict[str, nn.Module]:
    modules = dict(model.named_modules())
    result: dict[str, nn.Module] = {}
    for name in specs:
        if name not in modules:
            raise KeyError(f"metric layer {name!r} is not present in model")
        result[name] = modules[name]
    return result


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
            loss = nn.functional.cross_entropy(model(values), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def _collect_gradients(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]], *, device: torch.device, max_batches: int = 4) -> torch.Tensor:
    rows = []
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.eval()
    for values, labels in batches[:max_batches]:
        values, labels = values.to(device), labels.to(device)
        model.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(values), labels)
        loss.backward()
        rows.append(torch.cat([(parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).detach().float().reshape(-1) for parameter in parameters]).cpu())
    model.zero_grad(set_to_none=True)
    return torch.stack(rows) if rows else torch.empty((0, sum(int(parameter.numel()) for parameter in parameters)))


def _attention_metrics(model: nn.Module) -> dict[str, Any]:
    records = []
    for index, block in enumerate(getattr(model, "encoder", [])):
        attention = getattr(block, "last_attention", None)
        if attention is not None:
            records.append({"layer": index, **attention_summary(attention, normalization_axis=-1)})
    return {"status": "computed", "layers": records, "normalization_axis": -1} if records else {"status": "unavailable", "reason": "model has no exposed attention weights"}


def _stage_metrics(
    model: nn.Module,
    calibration_batches: list[tuple[torch.Tensor, torch.Tensor]],
    specs: dict[str, tuple[int, str]],
    *,
    device: torch.device,
    max_observations: int,
    previous_representations: dict[str, torch.Tensor] | None,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    layer_map = _layer_map(model, specs)
    model.to(device)
    cpu_batches = [(values, labels) for values, labels in calibration_batches]
    representations = capture_representations(
        model,
        [values.to(device) for values, _labels in cpu_batches],
        layer_map,
        forward_fn=lambda module, values: module(values),
        max_batches=len(cpu_batches),
    )
    metrics: dict[str, Any] = {"layers": {}, "attention": _attention_metrics(model), "parameters": parameter_norm_summary(model)}
    for name, (axis, kind) in specs.items():
        values = representations.get(name)
        if values is None:
            metrics["layers"][name] = {"status": "unavailable"}
            continue
        item = {
            "spectrum": spectral_summary(values, feature_axis=axis, max_observations=max_observations),
            "activation": activation_summary(values, kind=kind),
        }
        if previous_representations and previous_representations.get(name) is not None:
            previous = previous_representations[name]
            item["drift"] = {
                "cka": linear_cka(values, previous, feature_axis=axis, max_observations=max_observations),
                "procrustes_residual": procrustes_residual(values, previous, feature_axis=axis, max_observations=max_observations),
            }
        metrics["layers"][name] = item
    first_values = cpu_batches[0][0].to(device)
    jacobian = sampled_parameter_jacobian(model, first_values, max_samples=min(4, int(first_values.shape[0])))
    metrics["jacobian"] = jacobian_summary(jacobian)
    metrics["local_linearity"] = local_linearity_summary(model, first_values, epsilons=(1e-3, 1e-2), directions=2, seed=seed)
    gradients = _collect_gradients(model, cpu_batches, device=device)
    metrics["gradient"] = gradient_summary(gradients)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=("synthetic-eeg", "synthetic-image", "digits"), default="synthetic-eeg")
    parser.add_argument("--architectures", default="all", help="comma-separated: transformer,eegnet,tcn for EEG; transformer,cnn for image")
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
    parser.add_argument("--max-observations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs" / "lop-diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tasks < 2 or args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("tasks must be >=2, epochs and batch-size must be positive")
    requested = [int(item) for item in args.probe_steps.split(",") if item.strip()]
    if not requested or requested[0] != 0 or requested[-1] <= 0:
        raise SystemExit("probe-steps must include 0 and a positive step")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    available = ("transformer", "eegnet", "tcn") if args.data == "synthetic-eeg" else ("transformer", "cnn")
    architectures = available if args.architectures == "all" else tuple(item.strip() for item in args.architectures.split(",") if item.strip())
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
        runs.append({"architecture": architecture, "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()), "stages": stages, "probes": probes})

    result = {
        "schema_version": 1,
        "protocol": "generic-lop-eeg-image-diagnostics-v1",
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
            "seed": args.seed,
            "device": str(device),
        },
        "runs": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.data}-lop-diagnostics.json"
    markdown_path = args.output_dir / f"{args.data}-lop-diagnostics.md"
    json_path.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "architectures": list(architectures), "scientific_conclusion_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
