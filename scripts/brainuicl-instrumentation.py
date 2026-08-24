#!/usr/bin/env python3
"""Read-only instrumentation for BrainUICL checkpoints.

This adapter deliberately lives in EdgeForge rather than in the external
BrainUICL checkout.  It imports the model definitions, checkpoint files and
processed EEG arrays, evaluates them in ``eval`` mode, and writes a JSON
metric envelope.  No source, checkpoint or dataset file in ``--brainuicl-root``
is modified.

The output contains the metrics needed by the EdgeForge LoP analysis contract:

``task.spectra.transformer_1.effective_rank``
    Effective rank of the Transformer representation at a checkpoint stage.
``task.representation.*``
    Drift against an explicitly supplied reference checkpoint (when present).
``task.attention.*``
    Entropy of BrainUICL's actual ``softmax(dim=1)`` attention axis.
``task.importance.*``
    Empirical-Fisher (squared cross-entropy gradients) summaries.
``task.norms.*`` / ``task.weight_norms.*``
    Parameter norm and count summaries.

The effective-rank and attention calculations are intentionally explicit and
are not delegated to a compiler pass.  This keeps the values stable across
eager/Inductor runs and makes the provenance of a LoP predictor auditable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


CHECKPOINT_NAMES = ("feature_extractor", "feature_encoder", "sleep_classifier")
COMPONENTS = ("fusion", "transformer_1", "classifier_input")
EPS = 1e-12


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _auxiliary_checkpoint_files(root: Path) -> list[dict[str, Any]]:
    """Return digests for non-model state colocated with a checkpoint.

    BrainUICL regularization methods currently save ``regularizer_state.pt``
    next to the three model parameter files.  We record its identity without
    loading or interpreting the object; this keeps EWC/SI/MAS provenance
    auditable while avoiding arbitrary checkpoint execution.
    """

    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(("feature_extractor_parameter_", "feature_encoder_parameter_", "sleep_classifier_parameter_")):
            continue
        rows.append({"path": str(path), "name": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return rows


def _optimizer_state_provenance(root: Path) -> dict[str, Any]:
    """Describe colocated optimizer-state files without loading them.

    BrainUICL's current milestone checkpoints contain model parameter files
    and (for regularized methods) ``regularizer_state.pt``, but not Adam
    moment/scheduler snapshots.  The absence is important evidence: a later
    analysis must not claim to have reconstructed an optimizer trajectory
    from parameters alone.  File names and digests are sufficient here; no
    arbitrary serialized object is deserialized.
    """

    candidates: list[dict[str, Any]] = []
    if root.is_dir():
        pattern = re.compile(r"(?:optimizer|optim[_-]?state|adam|moment(?:um)?|scheduler)", re.IGNORECASE)
        for path in sorted(root.iterdir()):
            if not path.is_file() or not pattern.search(path.name):
                continue
            candidates.append({
                "path": str(path),
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "loaded": False,
            })
    return {
        "status": "available" if candidates else "unavailable",
        "files": candidates,
        "loaded": False,
        "protocol": "checkpoint-optimizer-provenance-v1",
        "reason": None if candidates else "no serialized optimizer/moment/scheduler state colocated with checkpoint",
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _setup_brainuicl_import(root: Path) -> None:
    resolved = str(root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _load_blocks(brainuicl_root: Path, checkpoint_root: Path, seed: int, device, dataset: str):
    """Load the three BrainUICL modules without writing to their repository."""

    import torch
    from model.pretrain_net import FeatureExtractor, FeatureExtractorFACED, SleepMLP, TransformerEncoder

    model_args = SimpleNamespace(dataset=dataset, device=device)
    frontend = FeatureExtractorFACED(model_args) if dataset == "FACED" else FeatureExtractor(model_args)
    blocks = [frontend, TransformerEncoder(model_args), SleepMLP(model_args)]
    checkpoint_paths: list[Path] = []
    for block, name in zip(blocks, CHECKPOINT_NAMES):
        checkpoint = checkpoint_root / f"{name}_parameter_{seed}.pkl"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        # ``weights_only`` prevents arbitrary checkpoint objects from being
        # executed.  The fallback keeps the adapter usable on older PyTorch.
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        block.load_state_dict(state)
        block.eval().to(device)
        checkpoint_paths.append(checkpoint)
    return blocks, checkpoint_paths


def _sample_files(data_root: Path, subject: int, max_files: int) -> list[tuple[Path, Path]]:
    subject_root = data_root / str(subject)
    if not subject_root.is_dir():
        subject_root = data_root / f"sub-{subject:03d}"
    data_dir = subject_root / "data"
    label_dir = subject_root / "label"
    files = sorted(data_dir.glob("*.npy"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no .npy sequence files under {data_dir}")
    pairs: list[tuple[Path, Path]] = []
    for data_path in files:
        label_path = label_dir / data_path.name
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        pairs.append((data_path, label_path))
    return pairs


def _load_batches(pairs: Iterable[tuple[Path, Path]], batch_size: int, dataset: str):
    """Read a bounded number of NPY sequences into CPU tensors."""

    import numpy as np
    import torch

    values: list[Any] = []
    labels: list[Any] = []
    for data_path, label_path in pairs:
        data = np.load(data_path).astype("float32", copy=False)
        label = np.load(label_path)
        expected_shape = (20, 32, 2500) if dataset == "FACED" else (20, 8, 3000)
        if tuple(data.shape) != expected_shape:
            raise ValueError(f"expected {data_path} to have shape {expected_shape}, got {data.shape}")
        if tuple(label.shape) != (20,):
            raise ValueError(f"expected {label_path} to have shape (20,), got {label.shape}")
        values.append(torch.from_numpy(data))
        labels.append(torch.from_numpy(label.astype("int64", copy=False)))
    batches = []
    for start in range(0, len(values), max(1, batch_size)):
        batches.append((torch.stack(values[start : start + batch_size]), torch.stack(labels[start : start + batch_size])))
    return batches


def _forward(blocks, values, dataset: str):
    """Return intermediate representations and logits for one batch."""

    channels = 32 if dataset == "FACED" else 8
    eog_channels = 1 if dataset == "FACED" else 2
    samples = 2500 if dataset == "FACED" else 3000
    flat = values.reshape(-1, channels, samples)
    eog, eeg = flat[:, :eog_channels], flat[:, eog_channels:]
    fused = blocks[0](eeg, eog)
    encoded = blocks[1](fused)
    classifier_input = blocks[2].sleep_stage_mlp(encoded)
    logits = blocks[2].sleep_stage_classifier(classifier_input).permute(0, 2, 1)
    return {
        "fusion": fused,
        "transformer_1": encoded,
        "classifier_input": classifier_input,
        "logits": logits,
    }


def spectral_summary(values, *, center: bool = True) -> dict[str, float | int | bool]:
    """Compute effective/stable rank and spectrum diagnostics.

    The last tensor dimension is treated as the representation dimension.  All
    preceding dimensions are observations.  Centering makes the metric measure
    representation variation rather than the mean activation vector.
    """

    import torch

    matrix = values.detach().float().reshape(-1, values.shape[-1])
    if matrix.shape[0] > 1 and center:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(matrix)
    if singular.numel() == 0:
        return {
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "spectral_entropy": 0.0,
            "spectral_entropy_normalized": 0.0,
            "rank90": 0,
            "rank95": 0,
            "rank99": 0,
            "sigma_max": 0.0,
            "condition_number": 0.0,
            "tail_energy": 0.0,
            "epsilon_rank": 0,
            "frobenius_norm": 0.0,
            "observation_count": int(matrix.shape[0]),
            "feature_dim": int(matrix.shape[1]),
            "centered": bool(center),
        }
    total = singular.sum()
    if float(total.item()) <= EPS:
        return {
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "spectral_entropy": 0.0,
            "spectral_entropy_normalized": 0.0,
            "rank90": 0,
            "rank95": 0,
            "rank99": 0,
            "sigma_max": 0.0,
            "condition_number": 0.0,
            "tail_energy": 0.0,
            "epsilon_rank": 0,
            "frobenius_norm": float(torch.linalg.vector_norm(matrix).item()),
            "observation_count": int(matrix.shape[0]),
            "feature_dim": int(matrix.shape[1]),
            "centered": bool(center),
        }
    probabilities = singular / total
    entropy = -(probabilities * probabilities.clamp_min(EPS).log()).sum()
    squared = singular.square()
    energy_total = squared.sum().clamp_min(EPS)
    cumulative = torch.cumsum(squared / energy_total, dim=0)
    sigma_max = singular[0].clamp_min(EPS)
    epsilon_rank = int((singular > (EPS * sigma_max)).sum().item())
    tail_energy = float((1.0 - cumulative[0]).clamp_min(0.0).item())

    def energy_rank(threshold: float) -> int:
        return int(torch.searchsorted(cumulative, torch.tensor(threshold, device=cumulative.device)).item() + 1)

    # The maximum possible entropy is log(min(observations, features)).
    max_entropy = math.log(max(1, int(singular.numel())))
    return {
        "effective_rank": float(entropy.exp().item()),
        "stable_rank": float((squared.sum() / sigma_max.square()).item()),
        "spectral_entropy": float(entropy.item()),
        "spectral_entropy_normalized": float((entropy / max(max_entropy, EPS)).item()),
        "rank90": energy_rank(0.90),
        "rank95": energy_rank(0.95),
        "rank99": energy_rank(0.99),
        "sigma_max": float(singular[0].item()),
        "condition_number": float((singular[0] / singular[-1].clamp_min(EPS)).item()),
        "tail_energy": tail_energy,
        "epsilon_rank": epsilon_rank,
        "frobenius_norm": float(torch.linalg.vector_norm(matrix).item()),
        "observation_count": int(matrix.shape[0]),
        "feature_dim": int(matrix.shape[1]),
        "centered": bool(center),
    }


def last_layer_jacobian_summary(representations, logits) -> dict[str, Any]:
    """Exact last-linear-layer feature factor and a curvature-weighted proxy.

    For logits ``W h`` the per-example Jacobian with respect to ``W`` is an
    output Kronecker factor of ``h``.  Therefore the feature spectrum is the
    exact nonzero spectrum of the unweighted last-layer Gram up to output
    multiplicity.  The softmax curvature weighting below is a scalar
    per-example proxy (full parameter-space GGN is intentionally not formed).
    """

    import torch

    features = representations.detach().float().reshape(-1, representations.shape[-1])
    result = {"scope": "classifier-last-layer-feature-factor", **spectral_summary(features)}
    probabilities = logits.detach().float().softmax(dim=1).permute(0, 2, 1).reshape(-1, logits.shape[1])
    curvature = (probabilities * (1.0 - probabilities)).mean(dim=1).clamp_min(EPS).sqrt()
    weighted = features * curvature[:, None]
    result["weighted"] = {"scope": "scalar-softmax-curvature-proxy", **spectral_summary(weighted)}
    return result


def local_linearity_diagnostic(blocks, values, device, dataset: str, epsilon: float = 1e-3) -> dict[str, Any]:
    """Estimate a finite-difference Taylor remainder along one parameter ray.

    A small ratio does not prove a global linear regime; it only records how
    well the first-order parameter expansion behaves for this calibration
    batch and perturbation scale.
    """

    import torch

    if epsilon <= 0:
        raise ValueError("linearity epsilon must be positive")
    probe = copy.deepcopy(blocks)
    for block in probe:
        block.to(device).eval()
    parameters = [parameter for block in probe for parameter in block.parameters()]
    directions = [torch.randn_like(parameter) for parameter in parameters]
    norm = torch.sqrt(sum(direction.float().square().sum() for direction in directions)).clamp_min(EPS)
    direction_scale = torch.sqrt(sum(parameter.float().square().sum() for parameter in parameters)).clamp_min(1.0) / norm
    directions = [direction * direction_scale for direction in directions]
    with torch.no_grad():
        base = _forward(probe, values.to(device), dataset)["logits"].float()
        for parameter, direction in zip(parameters, directions):
            parameter.add_(direction * epsilon)
        first_point = _forward(probe, values.to(device), dataset)["logits"].float()
        for parameter, direction in zip(parameters, directions):
            parameter.add_(direction * epsilon)
        second_point = _forward(probe, values.to(device), dataset)["logits"].float()
    first = first_point - base
    second_difference = second_point - 2.0 * first_point + base
    first_norm = torch.linalg.vector_norm(first)
    second_norm = torch.linalg.vector_norm(second_difference)
    return {
        "status": "computed",
        "epsilon_relative_to_parameter_norm": float(epsilon),
        "first_difference_norm": float(first_norm.item()),
        "second_difference_norm": float(second_norm.item()),
        "relative_second_difference": float((second_norm / first_norm.clamp_min(EPS)).item()),
        "interpretation": "local finite-difference diagnostic; not a global linearity proof",
    }


def representation_drift(current, reference) -> dict[str, float | int | str]:
    """Compare two representations collected on exactly the same examples."""

    import torch
    import torch.nn.functional as F

    left = current.detach().float().reshape(-1, current.shape[-1])
    right = reference.detach().float().reshape(-1, reference.shape[-1])
    if left.shape != right.shape:
        raise ValueError(f"representation shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    delta = left - right
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    delta_norm = torch.linalg.vector_norm(delta)
    cosine = F.cosine_similarity(left, right, dim=-1, eps=EPS)
    centroid_delta = left.mean(dim=0) - right.mean(dim=0)
    return {
        "status": "computed",
        "mean_l2": float(torch.linalg.vector_norm(delta, dim=-1).mean().item()),
        "rms": float(torch.sqrt(delta.square().mean()).item()),
        "relative_frobenius": float((delta_norm / right_norm.clamp_min(EPS)).item()),
        "mean_cosine_similarity": float(cosine.mean().item()),
        "mean_cosine_distance": float((1.0 - cosine).mean().item()),
        "centroid_l2": float(torch.linalg.vector_norm(centroid_delta).item()),
        "current_frobenius": float(left_norm.item()),
        "reference_frobenius": float(right_norm.item()),
        "observation_count": int(left.shape[0]),
        "feature_dim": int(left.shape[1]),
    }


class _AttentionCapture:
    """Reconstruct BrainUICL's attention probabilities from a pre-hook.

    BrainUICL does not return attention probabilities.  The pre-hook observes
    the normalized input and uses the module's own Q/K weights to reproduce
    ``softmax(scores, dim=1)`` exactly in eval mode.  This is read-only and
    avoids editing the external model source.
    """

    def __init__(self, block):
        self.records: list[dict[str, Any]] = []
        self.handles = []
        for name, module in block.named_modules():
            if module.__class__.__name__ == "MultiHeadAttention":
                self.handles.append(module.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name: str):
        import torch

        def callback(module, inputs):
            if not inputs:
                return
            x = inputs[0]
            with torch.no_grad():
                batch, sequence, width = x.shape
                heads = int(module.num_head)
                head_width = int(module.input_size // heads)
                query = module.w_query(x).view(batch, -1, heads, head_width).permute(0, 2, 1, 3)
                key = module.w_key(x).view(batch, -1, heads, head_width).permute(0, 2, 1, 3)
                scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(float(width))
                # This is intentionally dim=1: it is BrainUICL's implemented
                # axis, even though conventional attention normalizes keys.
                probs = torch.softmax(scores, dim=1)
                entropy_head = -(probs * probs.clamp_min(EPS).log()).sum(dim=1)
                normalized_head = entropy_head / math.log(max(2, heads))
                key_probs = torch.softmax(scores, dim=-1)
                entropy_key = -(key_probs * key_probs.clamp_min(EPS).log()).sum(dim=-1)
                normalized_key = entropy_key / math.log(max(2, int(sequence)))
                diagonal = key_probs.diagonal(dim1=-2, dim2=-1)
                self.records.append(
                    {
                        "module": name or module.__class__.__name__,
                        "head_axis_entropy": entropy_head.detach().cpu(),
                        "head_axis_entropy_normalized": normalized_head.detach().cpu(),
                        "key_axis_entropy": entropy_key.detach().cpu(),
                        "key_axis_entropy_normalized": normalized_key.detach().cpu(),
                        "max_probability": float(probs.max().item()),
                        "offdiag_mass": float((1.0 - diagonal.mean()).item()),
                        "heads": heads,
                        "sequence_length": int(sequence),
                        "normalization_axis": 1,
                    }
                )

        return callback

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {"status": "unavailable", "reason": "no MultiHeadAttention invocation observed"}
        import torch

        result: dict[str, Any] = {
            "status": "computed",
            "normalization_axis": 1,
            "normalization_axis_description": "BrainUICL softmax(dim=1), across heads",
            "calls": len(self.records),
            "modules": {},
        }
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self.records:
            grouped[record["module"]].append(record)
        for name, records in grouped.items():
            head_entropy = torch.cat([item["head_axis_entropy"].reshape(-1) for item in records])
            head_norm = torch.cat([item["head_axis_entropy_normalized"].reshape(-1) for item in records])
            key_entropy = torch.cat([item["key_axis_entropy"].reshape(-1) for item in records])
            key_norm = torch.cat([item["key_axis_entropy_normalized"].reshape(-1) for item in records])
            result["modules"][name] = {
                "entropy_mean": float(head_entropy.mean().item()),
                "entropy_std": float(head_entropy.std(unbiased=False).item()),
                "entropy_min": float(head_entropy.min().item()),
                "entropy_max": float(head_entropy.max().item()),
                "entropy_normalized_mean": float(head_norm.mean().item()),
                "key_axis_entropy_mean": float(key_entropy.mean().item()),
                "key_axis_entropy_normalized_mean": float(key_norm.mean().item()),
                "max_probability": float(sum(item["max_probability"] for item in records) / len(records)),
                "offdiag_mass": float(sum(item["offdiag_mass"] for item in records) / len(records)),
                "heads": int(records[0]["heads"]),
                "sequence_length": int(records[0]["sequence_length"]),
                "call_count": len(records),
            }
        # A stable alias used by the EdgeForge metric contract.
        first = next(iter(result["modules"].values()))
        result.update(
            {
                "transformer_1": first,
                "entropy_mean": first["entropy_mean"],
                "entropy_normalized_mean": first["entropy_normalized_mean"],
                "key_axis_entropy_mean": first["key_axis_entropy_mean"],
                "max_probability": first["max_probability"],
                "offdiag_mass": first["offdiag_mass"],
            }
        )
        return result


class _ActivationCapture:
    """Streaming activation/dormancy statistics for nonlinear modules."""

    def __init__(self, blocks):
        self.handles = []
        self.stats: dict[str, dict[str, float | int]] = {}
        for block_name, block in zip(CHECKPOINT_NAMES, blocks):
            for name, module in block.named_modules():
                kind = module.__class__.__name__
                if kind not in {"ReLU", "GELU", "SiLU", "Tanh", "Sigmoid"}:
                    continue
                key = f"{block_name}.{name or kind}"
                self.handles.append(module.register_forward_hook(self._hook(key, kind)))

    def _hook(self, key: str, kind: str):
        import torch

        def callback(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            values = output.detach().float()
            item = self.stats.setdefault(key, {"kind": kind, "elements": 0, "zero": 0, "abs_sum": 0.0, "sq_sum": 0.0, "min": float("inf"), "max": float("-inf")})
            flat = values.reshape(-1)
            item["elements"] += int(flat.numel())
            item["zero"] += int(torch.count_nonzero(flat.abs() <= 1e-6).item())
            item["abs_sum"] += float(flat.abs().sum().item())
            item["sq_sum"] += float(flat.square().sum().item())
            item["min"] = min(float(item["min"]), float(flat.min().item()))
            item["max"] = max(float(item["max"]), float(flat.max().item()))

        return callback

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {"status": "computed" if self.stats else "unavailable", "modules": {}}
        for key, item in self.stats.items():
            count = max(1, int(item["elements"]))
            mean_abs = float(item["abs_sum"]) / count
            variance = max(0.0, float(item["sq_sum"]) / count - (float(item["abs_sum"]) / count) ** 2)
            result["modules"][key] = {
                "kind": item["kind"],
                # ``dead`` has a strict dormant-unit interpretation only for
                # ReLU.  For GELU/Tanh/etc. expose the same numerical test as
                # near-zero fraction and do not label it a dead neuron.
                "dead_fraction": float(item["zero"]) / count if item["kind"] == "ReLU" else None,
                "near_zero_fraction": float(item["zero"]) / count,
                "mean_abs": mean_abs,
                "std": math.sqrt(variance),
                "min": float(item["min"]),
                "max": float(item["max"]),
                "element_count": count,
            }
        return result


def activation_metrics(blocks, batches, device, dataset: str) -> dict[str, Any]:
    capture = _ActivationCapture(blocks)
    try:
        import torch

        with torch.no_grad():
            for values, _labels in batches:
                _forward(blocks, values.to(device), dataset)
    finally:
        summary = capture.summary()
        capture.close()
    return summary


def attention_metrics(blocks, batches, device, dataset: str) -> dict[str, Any]:
    capture = _AttentionCapture(blocks[1])
    try:
        import torch

        with torch.no_grad():
            for values, _labels in batches:
                _forward(blocks, values.to(device), dataset)
    finally:
        summary = capture.summary()
        capture.close()
    return summary


def _module_group(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else name


def weight_norms(blocks) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {"by_block": {}, "by_module": {}}
    global_sq = 0.0
    global_count = 0
    for block_name, block in zip(CHECKPOINT_NAMES, blocks):
        block_sq = 0.0
        block_count = 0
        for parameter_name, parameter in block.named_parameters():
            values = parameter.detach().float()
            sq = float(values.square().sum().item())
            count = int(values.numel())
            block_sq += sq
            block_count += count
            global_sq += sq
            global_count += count
            module_name = f"{block_name}.{_module_group(parameter_name)}"
            item = result["by_module"].setdefault(module_name, {"squared_l2": 0.0, "parameter_count": 0})
            item["squared_l2"] += sq
            item["parameter_count"] += count
        result["by_block"][block_name] = {
            "l2": math.sqrt(block_sq),
            "squared_l2": block_sq,
            "parameter_count": block_count,
        }
    for item in result["by_module"].values():
        item["l2"] = math.sqrt(item["squared_l2"])
    result["global_l2"] = math.sqrt(global_sq)
    result["global_parameter_count"] = global_count
    # Flat aliases retain compatibility with the existing post-hoc probe.
    for block_name, item in result["by_block"].items():
        result[f"{block_name}_l2"] = item["l2"]
    return result


def parameter_update_summary(current_blocks, reference_blocks) -> dict[str, Any]:
    """Measure checkpoint parameter movement against a reference state.

    This is a state-difference diagnostic, not an optimizer trajectory: it
    only compares the serialized parameters available at the two checkpoint
    roots.  Recording both absolute and relative norms prevents a large
    representation drift from being attributed to LoP when the underlying
    checkpoint update itself is simply much larger.
    """

    import torch
    import torch.nn.functional as F

    rows: list[dict[str, Any]] = []
    by_block: dict[str, dict[str, float | int]] = {}
    global_delta_sq = 0.0
    global_reference_sq = 0.0
    global_count = 0
    for block_index, (current, reference) in enumerate(zip(current_blocks, reference_blocks)):
        block_name = CHECKPOINT_NAMES[block_index] if block_index < len(CHECKPOINT_NAMES) else f"block_{block_index}"
        delta_sq = 0.0
        reference_sq = 0.0
        count = 0
        for parameter_name, parameter in current.named_parameters():
            reference_parameter = dict(reference.named_parameters()).get(parameter_name)
            if reference_parameter is None:
                continue
            left = parameter.detach().float().reshape(-1)
            right = reference_parameter.detach().float().reshape(-1)
            if left.shape != right.shape:
                continue
            delta = left - right
            delta_norm = float(torch.linalg.vector_norm(delta).item())
            reference_norm = float(torch.linalg.vector_norm(right).item())
            rows.append({
                "name": f"{block_name}.{parameter_name}",
                "delta_l2": delta_norm,
                "reference_l2": reference_norm,
                "relative_update": delta_norm / max(reference_norm, EPS),
                "cosine_to_reference": float(F.cosine_similarity(left[None, :], right[None, :], dim=1, eps=EPS).item()),
                "parameter_count": int(left.numel()),
            })
            delta_sq += float(delta.square().sum().item())
            reference_sq += float(right.square().sum().item())
            count += int(left.numel())
        global_delta_sq += delta_sq
        global_reference_sq += reference_sq
        global_count += count
        by_block[block_name] = {
            "delta_l2": math.sqrt(delta_sq),
            "reference_l2": math.sqrt(reference_sq),
            "relative_update": math.sqrt(delta_sq) / max(math.sqrt(reference_sq), EPS),
            "parameter_count": count,
        }
    rows.sort(key=lambda item: item["delta_l2"], reverse=True)
    return {
        "status": "computed",
        "protocol": "checkpoint-parameter-delta-v1",
        "by_block": by_block,
        "top_parameters": rows[:20],
        "global_delta_l2": math.sqrt(global_delta_sq),
        "global_reference_l2": math.sqrt(global_reference_sq),
        "global_relative_update": math.sqrt(global_delta_sq) / max(math.sqrt(global_reference_sq), EPS),
        "parameter_count": global_count,
    }


def empirical_fisher(blocks, batches, device, max_batches: int, top_k: int, dataset: str) -> dict[str, Any]:
    """Estimate importance as mean squared cross-entropy gradients.

    This is an empirical Fisher proxy, not a claim of exact Fisher information.
    The output records the protocol and batch count so later LoP analyses can
    reject underpowered or incomparable estimates.
    """

    import torch
    import torch.nn.functional as F

    accum: dict[str, dict[str, Any]] = {}
    for block_name, block in zip(CHECKPOINT_NAMES, blocks):
        for parameter_name, parameter in block.named_parameters():
            accum[f"{block_name}.{parameter_name}"] = {
                "squared_grad_sum": 0.0,
                "abs_grad_sum": 0.0,
                "nonzero_grad_count": 0,
                "parameter_count": int(parameter.numel()),
                "seen_batches": 0,
            }
    losses: list[float] = []
    gradient_vectors: list[Any] = []
    used = 0
    for values, labels in batches:
        if max_batches > 0 and used >= max_batches:
            break
        values = values.to(device)
        labels = labels.to(device)
        for block in blocks:
            block.zero_grad(set_to_none=True)
        outputs = _forward(blocks, values, dataset)
        logits = outputs["logits"].permute(0, 2, 1).reshape(-1, outputs["logits"].shape[1])
        target = labels.reshape(-1).long()
        loss = F.cross_entropy(logits, target)
        loss.backward()
        losses.append(float(loss.detach().item()))
        used += 1
        # Keep only a bounded number of flattened vectors; this is a
        # diagnostic Gram/cosine estimate, not a full parameter-space matrix.
        if len(gradient_vectors) < 16:
            gradient_vectors.append(torch.cat([
                parameter.grad.detach().float().reshape(-1)
                for block in blocks for parameter in block.parameters()
                if parameter.grad is not None
            ]).cpu())
        for block_name, block in zip(CHECKPOINT_NAMES, blocks):
            for parameter_name, parameter in block.named_parameters():
                if parameter.grad is None:
                    continue
                key = f"{block_name}.{parameter_name}"
                gradient = parameter.grad.detach().float()
                item = accum[key]
                item["squared_grad_sum"] += float(gradient.square().sum().item())
                item["abs_grad_sum"] += float(gradient.abs().sum().item())
                item["nonzero_grad_count"] += int(torch.count_nonzero(gradient).item())
                item["seen_batches"] += 1
    for block in blocks:
        block.zero_grad(set_to_none=True)
    if used == 0:
        return {"status": "unavailable", "reason": "no importance batches", "protocol": "empirical-fisher-cross-entropy"}

    by_block: dict[str, dict[str, float | int]] = {}
    by_module: dict[str, dict[str, float | int]] = {}
    rows: list[dict[str, Any]] = []
    for name, item in accum.items():
        mean_sq = item["squared_grad_sum"] / max(1, item["seen_batches"] * item["parameter_count"])
        row = {
            "name": name,
            "mean_squared_grad": mean_sq,
            "squared_grad_sum": item["squared_grad_sum"],
            "mean_abs_grad": item["abs_grad_sum"] / max(1, item["seen_batches"] * item["parameter_count"]),
            "nonzero_fraction": item["nonzero_grad_count"] / max(1, item["seen_batches"] * item["parameter_count"]),
            "parameter_count": item["parameter_count"],
            "seen_batches": item["seen_batches"],
        }
        rows.append(row)
        block_name = name.split(".", 1)[0]
        block_item = by_block.setdefault(block_name, {"fisher_trace": 0.0, "parameter_count": 0, "nonzero_grad_count": 0, "max_mean_squared_grad": 0.0})
        block_item["fisher_trace"] += item["squared_grad_sum"] / max(1, item["seen_batches"])
        block_item["parameter_count"] += item["parameter_count"]
        block_item["nonzero_grad_count"] += item["nonzero_grad_count"]
        block_item["max_mean_squared_grad"] = max(float(block_item["max_mean_squared_grad"]), float(mean_sq))
        module_name = _module_group(name)
        module_item = by_module.setdefault(module_name, {"fisher_trace": 0.0, "parameter_count": 0})
        module_item["fisher_trace"] += item["squared_grad_sum"] / max(1, item["seen_batches"])
        module_item["parameter_count"] += item["parameter_count"]
    total = sum(float(item["fisher_trace"]) for item in by_block.values())
    for item in by_block.values():
        item["fisher_fraction"] = float(item["fisher_trace"] / max(total, EPS))
        item["mean_squared_grad"] = float(item["fisher_trace"] / max(item["parameter_count"], 1))
        item["nonzero_fraction"] = float(item["nonzero_grad_count"] / max(1, item["parameter_count"] * used))
    for item in by_module.values():
        item["fisher_fraction"] = float(item["fisher_trace"] / max(total, EPS))
        item["mean_squared_grad"] = float(item["fisher_trace"] / max(item["parameter_count"], 1))
    rows.sort(key=lambda row: row["mean_squared_grad"], reverse=True)
    gradient_summary: dict[str, Any] = {"status": "unavailable", "reason": "fewer than two gradient batches"}
    if len(gradient_vectors) >= 2:
        matrix = torch.stack(gradient_vectors)
        norms = torch.linalg.vector_norm(matrix, dim=1).clamp_min(EPS)
        cosine = (matrix @ matrix.T) / (norms[:, None] * norms[None, :])
        mask = ~torch.eye(cosine.shape[0], dtype=torch.bool)
        offdiag = cosine[mask]
        gradient_summary = {
            "status": "computed",
            "batch_count": int(matrix.shape[0]),
            "parameter_dim": int(matrix.shape[1]),
            "norm_mean": float(norms.mean().item()),
            "norm_std": float(norms.std(unbiased=False).item()),
            "cosine_mean": float(offdiag.mean().item()),
            "cosine_min": float(offdiag.min().item()),
            "cosine_max": float(offdiag.max().item()),
            "negative_fraction": float((offdiag < 0).float().mean().item()),
            "effective_rank": float(spectral_summary(matrix, center=True)["effective_rank"]),
        }
    return {
        "status": "computed",
        "protocol": "empirical-fisher-cross-entropy",
        "label_source": "processed EEG labels",
        "batch_count": used,
        "mean_loss": sum(losses) / len(losses),
        "fisher_trace": total,
        "by_block": by_block,
        "by_module": by_module,
        "top_parameters": rows[: max(1, top_k)],
        "gradient": gradient_summary,
    }


def _metric_unit(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("fraction", "ratio", "similarity", "distance", "gain")):
        return "ratio"
    # Effective/stable rank are dimensionless spectral statistics.  Only the
    # explicit energy-rank thresholds and observation/parameter counts are
    # counts; treating ``effective_rank`` as a count would make the metric
    # envelope misleading to consumers.
    if any(token in lowered for token in ("rank90", "rank95", "rank99", "count", "batches")):
        return "count"
    return "scalar"


def _flatten_metrics(value: Any, prefix: str, *, stage: int, context: dict[str, Any], output: list[dict[str, Any]]) -> None:
    if isinstance(value, bool):
        return
    number = _finite(value)
    if number is not None:
        row = {"namespace": "raeeg.research", "name": prefix, "value": number, "step": stage, "unit": _metric_unit(prefix), "context": context}
        key = (row["name"], row["step"], json.dumps(row["context"], sort_keys=True, separators=(",", ":")))
        if not any((item.get("name"), item.get("step"), json.dumps(item.get("context") or {}, sort_keys=True, separators=(",", ":"))) == key for item in output):
            output.append(row)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                _flatten_metrics(item, f"{prefix}.{key}" if prefix else key, stage=stage, context=context, output=output)


def _collect_representations(blocks, batches, device, max_batches: int, dataset: str):
    import torch

    collected: dict[str, list[Any]] = {name: [] for name in COMPONENTS}
    used = 0
    with torch.no_grad():
        for values, _labels in batches:
            if max_batches > 0 and used >= max_batches:
                break
            outputs = _forward(blocks, values.to(device), dataset)
            for name in COMPONENTS:
                collected[name].append(outputs[name].detach().cpu())
            used += 1
    if used == 0:
        raise RuntimeError("no batches available for representation collection")
    return {name: torch.cat(values, dim=0) for name, values in collected.items()}, used


def _canonical_metric_views(spectra: dict[str, Any], drift: dict[str, Any], attention: dict[str, Any], importance: dict[str, Any], norms: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Add stable LoP-facing aliases while retaining detailed diagnostics."""

    representation = drift
    if isinstance(drift, dict) and any(isinstance(value, dict) and value.get("status") == "computed" for value in drift.values()):
        representation = {key: dict(value) if isinstance(value, dict) else value for key, value in drift.items()}
        for component, item in list(representation.items()):
            if not isinstance(item, dict) or item.get("status") != "computed":
                continue
            # Names used in the LoP protocol are deliberately semantic rather
            # than tied to one implementation's internal field names.
            item.setdefault("drift_from_previous", item.get("mean_l2"))
            item.setdefault("cosine_to_previous", item.get("mean_cosine_similarity"))
            item.setdefault("frobenius_relative_change", item.get("relative_frobenius"))

    attention_view = dict(attention)
    transformer_attention = attention.get("transformer_1") if isinstance(attention, dict) else None
    if isinstance(transformer_attention, dict):
        attention_view.setdefault("layer_1", {
            "entropy": transformer_attention.get("entropy_mean"),
            "head_entropy": transformer_attention.get("entropy_normalized_mean"),
            "max_probability": transformer_attention.get("max_probability"),
            "offdiag_mass": transformer_attention.get("offdiag_mass"),
            "normalization_axis": attention.get("normalization_axis"),
        })

    importance_view = dict(importance)
    for block_name, item in (importance.get("by_block") or {}).items() if isinstance(importance, dict) else []:
        if not isinstance(item, dict):
            continue
        importance_view.setdefault(block_name, {
            "mean": item.get("mean_squared_grad"),
            "max": item.get("max_mean_squared_grad"),
            "nonzero_fraction": item.get("nonzero_fraction"),
            "fisher_fraction": item.get("fisher_fraction"),
        })

    norm_view = {key: value for key, value in (norms.get("by_block") or {}).items()} if isinstance(norms, dict) else {}
    return representation, attention_view, importance_view, norm_view


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brainuicl-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ISRUC", "FACED"), default="ISRUC")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--checkpoint-stage", type=int, default=0)
    parser.add_argument("--method", default="instrumentation")
    parser.add_argument("--split", default="subject-eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--importance-batches", type=int, default=1)
    parser.add_argument("--importance-top-k", type=int, default=20)
    parser.add_argument("--linearity-epsilon", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.checkpoint_stage < 0:
        raise SystemExit("--checkpoint-stage must be non-negative")
    _setup_brainuicl_import(args.brainuicl_root)
    import torch

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    pairs = _sample_files(args.data_root, args.subject, args.max_files)
    batches = _load_batches(pairs, args.batch_size, args.dataset)
    blocks, checkpoint_paths = _load_blocks(args.brainuicl_root, args.checkpoint_root, args.seed, device, args.dataset)
    representations, representation_batches = _collect_representations(blocks, batches, device, args.max_batches, args.dataset)
    spectra = {name: spectral_summary(values) for name, values in representations.items()}
    # Re-run only the bounded calibration batches to obtain logits for the
    # last-layer Jacobian/curvature proxy; no full parameter Jacobian is
    # materialized.
    import torch as _torch
    with _torch.no_grad():
        logits_for_jacobian = _torch.cat([
            _forward(blocks, values.to(device), args.dataset)["logits"].detach().cpu()
            for values, _labels in batches[: max(1, args.max_batches or len(batches))]
        ], dim=0)
    jacobian = last_layer_jacobian_summary(representations["classifier_input"], logits_for_jacobian)
    linearity = local_linearity_diagnostic(blocks, batches[0][0], device, args.dataset, args.linearity_epsilon)
    attention = attention_metrics(blocks, batches[: max(1, args.max_batches or len(batches))], device, args.dataset)
    activation = activation_metrics(blocks, batches[: max(1, args.max_batches or len(batches))], device, args.dataset)
    importance = empirical_fisher(blocks, batches, device, args.importance_batches, args.importance_top_k, args.dataset)
    norms = weight_norms(blocks)

    drift: dict[str, Any] = {
        "status": "not-computed",
        "reason": "--baseline-checkpoint-root was not supplied",
    }
    parameter_updates: dict[str, Any] = {
        "status": "not-computed",
        "reason": "--baseline-checkpoint-root was not supplied",
    }
    baseline_digests: list[dict[str, str]] = []
    baseline_auxiliary_files: list[dict[str, Any]] = []
    optimizer_state = _optimizer_state_provenance(args.checkpoint_root.resolve())
    baseline_optimizer_state = {"status": "not-computed", "loaded": False, "files": []}
    if args.baseline_checkpoint_root is not None:
        baseline_blocks, baseline_paths = _load_blocks(args.brainuicl_root, args.baseline_checkpoint_root, args.seed, device, args.dataset)
        baseline_representations, _ = _collect_representations(baseline_blocks, batches, device, args.max_batches, args.dataset)
        drift = {name: representation_drift(representations[name], baseline_representations[name]) for name in COMPONENTS}
        parameter_updates = parameter_update_summary(blocks, baseline_blocks)
        baseline_digests = [{"path": str(path), "sha256": _sha256(path)} for path in baseline_paths]
        baseline_auxiliary_files = _auxiliary_checkpoint_files(args.baseline_checkpoint_root.resolve())
        baseline_optimizer_state = _optimizer_state_provenance(args.baseline_checkpoint_root.resolve())
        del baseline_blocks

    representation_view, attention_view, importance_view, norm_view = _canonical_metric_views(spectra, drift, attention, importance, norms)
    context = {
        "dataset": str(args.dataset),
        "subject": str(args.subject),
        "method": str(args.method),
        "split": str(args.split),
        "probe_budget": representation_batches,
        "measurement_protocol": "brainuicl-checkpoint-instrumentation-v1",
    }
    task = {
        "task": args.checkpoint_stage,
        "stage": args.checkpoint_stage,
        "dataset": str(args.dataset),
        "subject": args.subject,
        "checkpoint_stage": args.checkpoint_stage,
        "split": str(args.split),
        "method": str(args.method),
        "probe_budget": representation_batches,
        "measurement_protocol": "brainuicl-checkpoint-instrumentation-v1",
        "spectra": spectra,
        "jacobian": jacobian,
        "local_linearity": linearity,
        "representation": representation_view,
        "parameter_updates": parameter_updates,
        "optimizer_state": optimizer_state,
        "attention": attention_view,
        "activation": activation,
        "importance": importance_view,
        "gradient": importance.get("gradient", {"status": "unavailable"}),
        "weight_norms": norms,
        "weight_norm": norm_view,
        # ``norms`` is a short alias for consumers that use task.norms.*.
        "norms": norms,
    }
    metrics: list[dict[str, Any]] = []
    _flatten_metrics(task["spectra"], "task.spectra", stage=args.checkpoint_stage, context=context, output=metrics)
    _flatten_metrics(task["jacobian"], "task.jacobian", stage=args.checkpoint_stage, context=context, output=metrics)
    _flatten_metrics(task["local_linearity"], "task.local_linearity", stage=args.checkpoint_stage, context=context, output=metrics)
    _flatten_metrics(task["representation"], "task.representation", stage=args.checkpoint_stage, context=context, output=metrics)
    _flatten_metrics(task["parameter_updates"], "task.parameter_updates", stage=args.checkpoint_stage, context=context, output=metrics)
    _flatten_metrics(task["attention"], "task.attention", stage=args.checkpoint_stage, context=context, output=metrics)
    _flatten_metrics(task["activation"], "task.activation", stage=args.checkpoint_stage, context=context, output=metrics)
    _flatten_metrics(task["importance"], "task.importance", stage=args.checkpoint_stage, context=context, output=metrics)
    _flatten_metrics(task["gradient"], "task.gradient", stage=args.checkpoint_stage, context=context, output=metrics)
    _flatten_metrics(task["weight_norms"], "task.weight_norms", stage=args.checkpoint_stage, context=context, output=metrics)
    _flatten_metrics(task["weight_norm"], "task.weight_norm", stage=args.checkpoint_stage, context=context, output=metrics)
    # ``task.spectra.transformer_1.effective_rank`` is emitted by the flatten
    # above.  Keep a single row so downstream stores cannot mistake duplicate
    # values for independent observations.
    predictor = spectra["transformer_1"]["effective_rank"]
    result = {
        "schema_version": 1,
        "instrumentation": "brainuicl-checkpoint-metrics-v1",
        "status": "succeeded",
        "read_only": True,
        "config": {
            "dataset": args.dataset,
            "subject": args.subject,
            "seed": args.seed,
            "checkpoint_stage": args.checkpoint_stage,
            "method": str(args.method),
            "split": str(args.split),
            "device": str(device),
            "max_files": args.max_files,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "importance_batches": args.importance_batches,
            "importance_top_k": args.importance_top_k,
            "linearity_epsilon": args.linearity_epsilon,
        },
        "source": {
            "brainuicl_root": str(args.brainuicl_root.resolve()),
            "checkpoint_root": str(args.checkpoint_root.resolve()),
            "checkpoint_digests": [{"path": str(path), "sha256": _sha256(path)} for path in checkpoint_paths],
            "checkpoint_auxiliary_files": _auxiliary_checkpoint_files(args.checkpoint_root.resolve()),
            "optimizer_state_provenance": optimizer_state,
            "baseline_checkpoint_digests": baseline_digests,
            "baseline_checkpoint_auxiliary_files": baseline_auxiliary_files,
            "baseline_optimizer_state_provenance": baseline_optimizer_state,
            "input_files": [{"data": str(data), "label": str(label), "data_sha256": _sha256(data), "label_sha256": _sha256(label)} for data, label in pairs],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "tasks": [task],
        "metrics": metrics,
        "summary": {
            "task_count": 1,
            "transformer_effective_rank": float(predictor),
            "transformer_stable_rank": float(spectra["transformer_1"]["stable_rank"]),
            "representation_drift_status": drift.get("status", "computed") if isinstance(drift, dict) else "computed",
            "attention_entropy_status": attention.get("status", "unavailable"),
            "importance_status": importance.get("status", "unavailable"),
            "optimizer_state_status": optimizer_state.get("status", "unavailable"),
            "weight_norm_global_l2": float(norms["global_l2"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
