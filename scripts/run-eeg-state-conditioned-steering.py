#!/usr/bin/env python3
"""Run a small, label-preserving EEG activation-steering pilot.

The pilot trains an EEGNet warm checkpoint on source subjects, selects temporal
ReLU filters with low active coverage, and constructs four target conditions:
clean, equal-RMS band-limited random noise, spectrum-matched noise, and
state-conditioned noise optimized against the selected filters.  The same
already-generated target batch is used by warm and fresh probes.  This is a
mechanism control, not a natural-data LoP claim.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _load_continuous_runner() -> Any:
    path = Path(__file__).with_name("run-eeg-architecture-continuous-lop.py")
    spec = importlib.util.spec_from_file_location("edgeforge_continuous_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_continuous_runner()


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def rms(values: torch.Tensor, dims: tuple[int, ...] = (-1, -2), keepdim: bool = True) -> torch.Tensor:
    return torch.sqrt(values.square().mean(dim=dims, keepdim=keepdim).clamp_min(1e-20))


def band_limit(values: torch.Tensor, sample_rate: float, low_hz: float, high_hz: float) -> torch.Tensor:
    """Keep only the requested real FFT bins; this is differentiable."""
    if values.ndim != 3:
        raise ValueError(f"expected [batch, channels, samples], got {tuple(values.shape)}")
    samples = int(values.shape[-1])
    frequencies = torch.fft.rfftfreq(samples, d=1.0 / float(sample_rate), device=values.device)
    mask = (frequencies >= float(low_hz)) & (frequencies <= float(high_hz))
    spectrum = torch.fft.rfft(values, dim=-1)
    return torch.fft.irfft(spectrum * mask.to(spectrum.dtype), n=samples, dim=-1)


def scale_to_fraction(delta: torch.Tensor, base: torch.Tensor, fraction: float) -> torch.Tensor:
    delta_rms = rms(delta, dims=(-1,), keepdim=True).clamp_min(1e-20)
    base_rms = rms(base, dims=(-1,), keepdim=True)
    return delta * (float(fraction) * base_rms / delta_rms)


def random_delta(base: torch.Tensor, seed: int, fraction: float, sample_rate: float) -> torch.Tensor:
    generator = torch.Generator(device=base.device).manual_seed(int(seed))
    noise = torch.randn(base.shape, generator=generator, device=base.device, dtype=base.dtype)
    return scale_to_fraction(band_limit(noise, sample_rate, 0.5, 40.0), base, fraction)


def spectral_match_delta(base: torch.Tensor, seed: int, fraction: float, sample_rate: float = 100.0) -> torch.Tensor:
    """Make random-phase perturbations with the input's per-channel spectrum."""
    generator = torch.Generator(device=base.device).manual_seed(int(seed))
    spectrum = torch.fft.rfft(base, dim=-1)
    phase = torch.rand(spectrum.shape, generator=generator, device=base.device, dtype=base.dtype) * (2.0 * math.pi)
    phase = torch.complex(torch.cos(phase), torch.sin(phase))
    frequencies = torch.fft.rfftfreq(base.shape[-1], d=1.0 / float(sample_rate), device=base.device)
    band = ((frequencies >= 0.5) & (frequencies <= 40.0)).to(spectrum.dtype)
    delta = torch.fft.irfft(spectrum.abs() * phase * band, n=base.shape[-1], dim=-1)
    delta = delta - delta.mean(dim=-1, keepdim=True)
    return scale_to_fraction(delta, base, fraction)


def temporal_preactivation(model: torch.nn.Module, values: torch.Tensor) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        captured.append(output)

    handle = model.temporal_norm.register_forward_hook(hook)
    try:
        model.eval()
        model(values)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("EEGNet temporal_norm hook did not capture a tensor")
    return captured[-1]


@torch.no_grad()
def unit_active_fraction(model: torch.nn.Module, values: torch.Tensor) -> torch.Tensor:
    preactivation = temporal_preactivation(model, values)
    # Canonical EEGNet is [B, filters, 1, samples]; tolerate the legacy 1-D map.
    reduce_dims = tuple(index for index in range(preactivation.ndim) if index not in (0, 1))
    return (preactivation > 0.0).float().mean(dim=reduce_dims).mean(dim=0)


def state_conditioned_delta(
    model: torch.nn.Module,
    base: torch.Tensor,
    low_units: torch.Tensor,
    *,
    seed: int,
    fraction: float,
    sample_rate: float,
    steps: int,
    lr: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Optimize a small band-limited perturbation to suppress selected filters."""
    delta = random_delta(base, seed, fraction, sample_rate).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=float(lr))
    history: list[float] = []
    model.eval()
    for _step in range(max(1, int(steps))):
        preactivation = temporal_preactivation(model, base + delta)
        selected = preactivation.index_select(1, low_units.to(preactivation.device))
        objective = F.relu(selected).mean() + 1e-3 * (delta.square().mean() / base.square().mean().clamp_min(1e-20))
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            projected = band_limit(delta, sample_rate, 0.5, 40.0)
            delta.copy_(scale_to_fraction(projected, base, fraction))
        history.append(float(objective.detach().item()))
    result = delta.detach()
    pre_clean = temporal_preactivation(model, base).detach()
    pre_steered = temporal_preactivation(model, base + result).detach()
    selected_clean = pre_clean.index_select(1, low_units).float()
    selected_steered = pre_steered.index_select(1, low_units).float()
    return result, {
        "objective_history": history,
        "selected_active_fraction_clean": float((selected_clean > 0).float().mean().item()),
        "selected_active_fraction_steered": float((selected_steered > 0).float().mean().item()),
        "perturbation_rms_fraction": float((rms(result, dims=(-1,)) / rms(base, dims=(-1,))).mean().item()),
        "waveform_correlation": float(torch.corrcoef(torch.stack((base.flatten(), (base + result).flatten())))[0, 1].item()),
    }


def reset_dormant_units(model: torch.nn.Module, units: torch.Tensor, seed: int) -> None:
    """Reinitialize selected EEGNet temporal filters as a ReDo-style control."""
    generator = torch.Generator(device=model.temporal_conv.weight.device).manual_seed(int(seed))
    with torch.no_grad():
        for unit in units.tolist():
            weight = model.temporal_conv.weight[int(unit)]
            bound = 1.0 / math.sqrt(max(1, weight[0].numel()))
            weight.copy_(torch.empty_like(weight).uniform_(-bound, bound, generator=generator))
            if getattr(model.temporal_norm, "weight", None) is not None:
                model.temporal_norm.weight[int(unit)].fill_(1.0)
            if getattr(model.temporal_norm, "bias", None) is not None:
                model.temporal_norm.bias[int(unit)].zero_()


def train_source(model: torch.nn.Module, values: torch.Tensor, labels: torch.Tensor, *, epochs: int, batch_size: int, lr: float, seed: int) -> torch.nn.Module:
    batches = runner.make_batches(values, labels, batch_size, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    for _epoch in range(max(1, int(epochs))):
        model.train()
        for batch_values, target in batches:
            loss = F.cross_entropy(model(batch_values), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def embedding_rank(model: torch.nn.Module, values: torch.Tensor) -> float:
    with torch.no_grad():
        representation = model.forward_bundle(values).representations["embedding"]
        return float(runner.lop_metrics.spectral_summary(representation, feature_axis=-1, max_observations=128)["effective_rank"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_all(args.seed)
    root = args.data_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_x, source_y, _ = runner.read_group(root, "source", args.source_subjects)
    retention_x, retention_y, _ = runner.read_group(root, "retention", args.retention_subjects)
    target_train, target_y, target_eval, target_eval_y, _ = runner.read_target_stage(root, args.target_subjects[0])
    source_x, source_y = source_x[: args.max_source_epochs], source_y[: args.max_source_epochs]
    retention_x, retention_y = retention_x[: args.max_retention_epochs], retention_y[: args.max_retention_epochs]
    target_train, target_y = target_train[: args.max_target_epochs], target_y[: args.max_target_epochs]
    target_eval, target_eval_y = target_eval[: args.max_target_eval_epochs], target_eval_y[: args.max_target_eval_epochs]
    device = torch.device(args.device)
    warm = runner.build_model("eegnet").to(device)
    train_source(warm, source_x.to(device), source_y.to(device), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    calibration = source_x[: min(args.calibration_epochs, len(source_x))].to(device)
    coverage = unit_active_fraction(warm, calibration)
    count = max(1, int(round(len(coverage) * float(args.low_unit_fraction))))
    low_units = torch.argsort(coverage)[:count].to(device)
    conditions: dict[str, torch.Tensor] = {
        "clean": target_train.to(device),
        "matched_random": target_train.to(device) + random_delta(target_train.to(device), args.seed + 11, args.perturbation_fraction, args.sample_rate),
        "spectral_match": target_train.to(device) + spectral_match_delta(target_train.to(device), args.seed + 12, args.perturbation_fraction, args.sample_rate),
    }
    steered_delta, steering = state_conditioned_delta(
        warm,
        target_train.to(device),
        low_units,
        seed=args.seed + 13,
        fraction=args.perturbation_fraction,
        sample_rate=args.sample_rate,
        steps=args.steering_steps,
        lr=args.steering_lr,
    )
    conditions["state_conditioned"] = target_train.to(device) + steered_delta
    retention_batches = runner.make_batches(retention_x, retention_y, args.batch_size, args.seed + 7000)
    rows: dict[str, Any] = {}
    for name, condition_train in conditions.items():
        train_batches = runner.make_batches(condition_train.cpu(), target_y, args.batch_size, args.seed + 100)
        eval_batches = runner.make_batches(target_eval.cpu(), target_eval_y, args.batch_size, args.seed + 101)
        warm_probe_model = copy.deepcopy(warm).cpu()
        fresh_seed = args.seed + 500000
        seed_all(fresh_seed)
        fresh_model = runner.build_model("eegnet")
        warm_after, warm_probe = runner.adapt_probe(warm_probe_model, train_batches, eval_batches, retention_batches, tuple(args.budgets), args.adapt_lr, 5, torch.device("cpu"))
        fresh_after, fresh_probe = runner.adapt_probe(fresh_model, train_batches, eval_batches, retention_batches, tuple(args.budgets), args.adapt_lr, 5, torch.device("cpu"))
        gaps = [{"step": int(w["step"]), "fresh_gap": float(f["accuracy"] - w["accuracy"])} for w, f in zip(warm_probe["curve"], fresh_probe["curve"])]
        rows[name] = {
            "gaps": gaps,
            "warm": warm_probe,
            "fresh": fresh_probe,
            "warm_initial_embedding_rank": embedding_rank(warm, target_train[: min(16, len(target_train))].to(device)),
        }
        del warm_after, fresh_after
    metadata = {
        "schema": "edgeforge.eeg-state-conditioned-steering.v1",
        "data_root": str(root),
        "architecture": "eegnet",
        "source_subjects": [int(x) for x in args.source_subjects],
        "target_subject": int(args.target_subjects[0]),
        "retention_subjects": [int(x) for x in args.retention_subjects],
        "conditions": list(conditions),
        "budgets": [int(x) for x in args.budgets],
        "selected_low_coverage_units": [int(x) for x in low_units.cpu().tolist()],
        "unit_active_fraction": [float(x) for x in coverage.cpu().tolist()],
        "steering": steering,
        "reset_control": "implemented as reset_dormant_units; not applied in this pilot unless explicitly extended",
        "scientific_conclusion_allowed": False,
    }
    result = {"metadata": metadata, "results": rows, "scientific_conclusion_allowed": False}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# State-conditioned EEG steering pilot", "", "This is a mechanism control, not a natural LoP conclusion.", "", "| condition | fresh-gap B=5 | fresh-gap B=10 |", "| --- | ---: | ---: |"]
    for name, row in rows.items():
        by_step = {x["step"]: x["fresh_gap"] for x in row["gaps"]}
        lines.append(f"| {name} | {by_step.get(5, 'n/a')} | {by_step.get(10, 'n/a')} |")
    lines.extend(["", f"Selected low-coverage units: `{metadata['selected_low_coverage_units']}`.", f"Steered selected active fraction: `{steering['selected_active_fraction_clean']:.6g}` -> `{steering['selected_active_fraction_steered']:.6g}`.", "", "The strict multi-seed/multi-transition LoP gate was not applied to this one-subject pilot."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-subjects", nargs="+", type=int, default=[1, 3, 4])
    parser.add_argument("--target-subjects", nargs="+", type=int, default=[2])
    parser.add_argument("--retention-subjects", nargs="+", type=int, default=[5])
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--budgets", nargs="+", type=int, default=[0, 5, 10])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-source-epochs", type=int, default=120)
    parser.add_argument("--max-target-epochs", type=int, default=20)
    parser.add_argument("--max-target-eval-epochs", type=int, default=20)
    parser.add_argument("--max-retention-epochs", type=int, default=20)
    parser.add_argument("--calibration-epochs", type=int, default=32)
    parser.add_argument("--low-unit-fraction", type=float, default=0.25)
    parser.add_argument("--perturbation-fraction", type=float, default=0.05)
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument("--steering-steps", type=int, default=8)
    parser.add_argument("--steering-lr", type=float, default=2e-7)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
