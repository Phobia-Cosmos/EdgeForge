#!/usr/bin/env python3
"""Read-only, label-free BrainUICL prediction/consistency diagnostics.

Unlike ``brainuicl-fixed-budget-probe.py``, this adapter never loads label
arrays and never performs an optimizer update.  It measures quantities that
are available to an online unlabeled adapter: predictive entropy, maximum
probability, pseudo-label class coverage, and consistency under a controlled
signal perturbation.  Labels may still be evaluated by a separate offline
script, but they are not part of this protocol or its output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
LOADER = importlib.util.spec_from_file_location("edgeforge_brainuicl_instrumentation", HERE / "brainuicl-instrumentation.py")
if LOADER is None or LOADER.loader is None:
    raise RuntimeError("cannot load instrumentation adapter")
instrumentation = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(instrumentation)

EPS = 1e-12


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_data_batches(pairs: list[tuple[Path, Path]], batch_size: int, dataset: str) -> list[Any]:
    """Load data only; the second path is retained solely for provenance."""

    import torch

    expected = (20, 32, 2500) if dataset == "FACED" else (20, 8, 3000)
    values: list[Any] = []
    for data_path, _label_path in pairs:
        array = np.load(data_path).astype("float32", copy=False)
        if tuple(array.shape) != expected:
            raise ValueError(f"expected {data_path} to have shape {expected}, got {array.shape}")
        values.append(torch.from_numpy(array))
    return [torch.stack(values[start : start + max(1, batch_size)]) for start in range(0, len(values), max(1, batch_size))]


def sample_data_files(data_root: Path, subject: int, max_files: int) -> list[Path]:
    """Discover signal files without requiring a label file to exist."""

    subject_root = data_root / str(subject)
    if not subject_root.is_dir():
        subject_root = data_root / f"sub-{subject:03d}"
    data_dir = subject_root / "data"
    files = sorted(data_dir.glob("*.npy"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no .npy signal files under {data_dir}")
    return files


def _entropy(probabilities) -> Any:
    import torch

    classes = max(2, int(probabilities.shape[1]))
    entropy = -(probabilities * probabilities.clamp_min(EPS).log()).sum(dim=1)
    return entropy, entropy / math.log(classes)


def _kl(left, right) -> Any:
    import torch

    return (left.clamp_min(EPS) * (left.clamp_min(EPS).log() - right.clamp_min(EPS).log())).sum(dim=1)


def collect(blocks, batches, *, device, dataset: str, noise_severity: float, seed: int, max_batches: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if noise_severity < 0:
        raise ValueError("noise severity must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    entropy_values: list[Any] = []
    normalized_entropy_values: list[Any] = []
    confidence_values: list[Any] = []
    agreement_values: list[Any] = []
    kl_values: list[Any] = []
    feature_cosine_values: list[Any] = []
    class_counts: list[int] | None = None
    used = 0
    with torch.no_grad():
        for values in batches:
            if max_batches > 0 and used >= max_batches:
                break
            values = values.to(device)
            clean_output = instrumentation._forward(blocks, values, dataset)
            clean_logits = clean_output["logits"].float()
            clean_probabilities = clean_logits.softmax(dim=1)
            entropy, normalized_entropy = _entropy(clean_probabilities)
            confidence, pseudo = clean_probabilities.max(dim=1)
            if class_counts is None:
                class_counts = [0] * int(clean_probabilities.shape[1])
            counts = torch.bincount(pseudo.reshape(-1).cpu(), minlength=len(class_counts))
            class_counts = [left + int(right) for left, right in zip(class_counts, counts.tolist())]

            channel_std = values.float().std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            noise = torch.randn(values.shape, generator=generator, dtype=values.dtype).to(device) * channel_std * float(noise_severity)
            perturbed = values + noise
            perturbed_output = instrumentation._forward(blocks, perturbed, dataset)
            perturbed_probabilities = perturbed_output["logits"].float().softmax(dim=1)
            perturbed_pseudo = perturbed_probabilities.argmax(dim=1)
            kl = _kl(clean_probabilities, perturbed_probabilities)
            agreement = (pseudo == perturbed_pseudo).float()
            clean_features = clean_output["transformer_1"].float().reshape(-1, clean_output["transformer_1"].shape[-1])
            perturbed_features = perturbed_output["transformer_1"].float().reshape(-1, perturbed_output["transformer_1"].shape[-1])
            feature_cosine = F.cosine_similarity(clean_features, perturbed_features, dim=-1, eps=EPS)
            entropy_values.append(entropy.reshape(-1).cpu())
            normalized_entropy_values.append(normalized_entropy.reshape(-1).cpu())
            confidence_values.append(confidence.reshape(-1).cpu())
            agreement_values.append(agreement.reshape(-1).cpu())
            kl_values.append(kl.reshape(-1).cpu())
            feature_cosine_values.append(feature_cosine.reshape(-1).cpu())
            used += 1
    if used == 0:
        raise ValueError("no data batches available")

    def mean(items: list[Any]) -> float:
        return float(torch.cat(items).mean().item())

    def std(items: list[Any]) -> float:
        return float(torch.cat(items).std(unbiased=False).item())

    return {
        "status": "computed",
        "protocol": "unlabeled-pseudo-consistency-v1",
        "batches": used,
        "observations": int(sum(item.numel() for item in entropy_values)),
        "predictive_entropy_mean": mean(entropy_values),
        "predictive_entropy_std": std(entropy_values),
        "predictive_entropy_normalized_mean": mean(normalized_entropy_values),
        "max_probability_mean": mean(confidence_values),
        "max_probability_std": std(confidence_values),
        "pseudo_label_coverage": float(sum(1 for item in class_counts or [] if item > 0) / max(1, len(class_counts or []))),
        "pseudo_label_class_counts": class_counts or [],
        "perturbation_noise_severity": float(noise_severity),
        "perturbation_prediction_agreement": mean(agreement_values),
        "perturbation_kl_mean": mean(kl_values),
        "representation_cosine_mean": mean(feature_cosine_values),
        "representation_cosine_std": std(feature_cosine_values),
        "labels_loaded": False,
        "optimizer_update": False,
    }


def _flatten(value: Any, prefix: str, *, stage: int, context: dict[str, Any], output: list[dict[str, Any]]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        output.append({"namespace": "raeeg.research", "name": prefix, "value": float(value), "step": stage, "unit": "ratio" if any(token in prefix for token in ("coverage", "agreement", "probability", "cosine")) else "scalar", "context": context})
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                _flatten(item, f"{prefix}.{key}" if prefix else key, stage=stage, context=context, output=output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brainuicl-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ISRUC", "FACED"), required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--checkpoint-stage", type=int, default=0)
    parser.add_argument("--method", default="unlabeled-diagnostics")
    parser.add_argument("--split", default="subject-eval")
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--noise-severity", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint_stage < 0:
        raise SystemExit("checkpoint-stage must be non-negative")
    instrumentation._setup_brainuicl_import(args.brainuicl_root)
    import torch

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    data_files = sample_data_files(args.data_root, args.subject, args.max_files)
    pairs = [(path, path.parent.parent / "label" / path.name) for path in data_files]
    batches = load_data_batches(pairs, args.batch_size, args.dataset)
    blocks, checkpoint_paths = instrumentation._load_blocks(args.brainuicl_root, args.checkpoint_root, args.seed, device, args.dataset)
    diagnostics = collect(blocks, batches, device=device, dataset=args.dataset, noise_severity=args.noise_severity, seed=args.seed, max_batches=args.max_batches)
    context = {"dataset": args.dataset, "subject": str(args.subject), "method": args.method, "split": args.split, "probe_budget": diagnostics["observations"], "measurement_protocol": "unlabeled-pseudo-consistency-v1"}
    task = {"task": args.checkpoint_stage, "stage": args.checkpoint_stage, "dataset": args.dataset, "subject": args.subject, "method": args.method, "split": args.split, "measurement_protocol": "unlabeled-pseudo-consistency-v1", "unlabeled": diagnostics}
    metrics: list[dict[str, Any]] = []
    _flatten(diagnostics, "task.unlabeled", stage=args.checkpoint_stage, context=context, output=metrics)
    result = {
        "schema_version": 1,
        "instrumentation": "brainuicl-unlabeled-diagnostics-v1",
        "status": "succeeded",
        "read_only": True,
        "config": {"dataset": args.dataset, "subject": args.subject, "seed": args.seed, "checkpoint_stage": args.checkpoint_stage, "method": args.method, "split": args.split, "noise_severity": args.noise_severity, "device": str(device), "max_files": args.max_files, "max_batches": args.max_batches},
        "source": {"checkpoint_digests": [{"path": str(path), "sha256": digest(path)} for path in checkpoint_paths], "input_files": [{"data": str(data), "data_sha256": digest(data), "label_path_recorded_only": str(label)} for data, label in pairs]},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)},
        "tasks": [task],
        "metrics": metrics,
        "summary": {"dataset": args.dataset, "subject": args.subject, "stage": args.checkpoint_stage, "predictive_entropy_normalized_mean": diagnostics["predictive_entropy_normalized_mean"], "max_probability_mean": diagnostics["max_probability_mean"], "perturbation_prediction_agreement": diagnostics["perturbation_prediction_agreement"], "labels_loaded": False, "optimizer_update": False, "scientific_conclusion_allowed": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
