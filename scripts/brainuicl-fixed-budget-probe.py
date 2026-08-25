#!/usr/bin/env python3
"""Run a fixed-budget fresh-vs-checkpoint EEG plasticity probe.

This is an EdgeForge-side adapter.  It reads BrainUICL definitions,
checkpoints and processed EEG but does not modify the external checkout.  The
probe is deliberately labelled ``supervised-oracle``: labels are used for a
controlled offline diagnostic and held-out evaluation.  It is not a claim
about an online unlabeled deployment protocol.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
_loader = importlib.util.spec_from_file_location("edgeforge_brainuicl_instrumentation", HERE / "brainuicl-instrumentation.py")
if _loader is None or _loader.loader is None:
    raise RuntimeError("cannot load instrumentation adapter")
instrumentation = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(instrumentation)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_steps(value: str) -> list[int]:
    steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not steps or steps[0] != 0 or steps[-1] <= 0:
        raise ValueError("probe steps must include 0 and at least one positive step")
    return steps


def clone_and_reset(blocks):
    import torch.nn as nn

    result = copy.deepcopy(blocks)

    def reset(module):
        # Do not reset containers; leaf modules with reset_parameters provide
        # the framework's standard fresh initialization.
        if not isinstance(module, (nn.Sequential, nn.ModuleList)) and hasattr(module, "reset_parameters"):
            module.reset_parameters()

    for block in result:
        block.apply(reset)
        block.eval()
    return result


def batches_from_files(pairs: list[tuple[Path, Path]], batch_size: int, dataset: str):
    batches = instrumentation._load_batches(pairs, batch_size, dataset)
    return batches


def split_pairs(pairs: list[tuple[Path, Path]], fraction: float) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if len(pairs) < 2:
        raise ValueError("fixed-budget probe needs at least two sequence files for train/eval split")
    pivot = min(len(pairs) - 1, max(1, int(round(len(pairs) * fraction))))
    return pairs[:pivot], pairs[pivot:]


def _metrics_from_logits(logits, labels, classes: int) -> tuple[float, float, float]:
    import torch

    prediction = logits.argmax(dim=1).reshape(-1).detach().cpu().numpy()
    target = labels.reshape(-1).detach().cpu().numpy()
    accuracy = float((prediction == target).mean()) if target.size else 0.0
    f1_values = []
    for cls in range(classes):
        true_positive = int(((prediction == cls) & (target == cls)).sum())
        false_positive = int(((prediction == cls) & (target != cls)).sum())
        false_negative = int(((prediction != cls) & (target == cls)).sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1_values.append(2 * precision * recall / max(1e-12, precision + recall))
    return accuracy, float(sum(f1_values) / max(1, classes)), float(prediction.size)


def _summarize_curve(rows: list[dict[str, Any]], keys: tuple[str, ...], prefix: str = "") -> dict[str, Any]:
    """Return deterministic initial/final/gain/AULC values for a probe curve.

    ``rows`` always contains an explicit optimizer step.  Keeping this helper
    separate from the main probe makes optional retention measurements follow
    exactly the same fixed-budget integration rule as the plasticity curve.
    """

    x = np.asarray([row["step"] for row in rows], dtype=np.float64)
    result: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        name = f"{prefix}{key}" if prefix else key
        result[name + "_initial"] = float(values[0])
        result[name + "_final"] = float(values[-1])
        result[name + "_gain"] = float(values[-1] - values[0])
        # NumPy < 2.0 exposes the same composite trapezoid rule as ``trapz``;
        # keep the probe runnable in the board/reference environments without
        # changing the metric definition.
        # Do not pass ``np.trapz`` as getattr's default: Python evaluates the
        # default eagerly, and NumPy 2.5 removed the legacy alias even when the
        # preferred ``trapezoid`` implementation is available.
        integrate = getattr(np, "trapezoid", None)
        if integrate is None:
            integrate = getattr(np, "trapz", None)
        if integrate is None:
            raise RuntimeError("NumPy must provide trapezoid integration")
        result[name + "_aulc"] = float(integrate(values, x) / max(1.0, x[-1]))
    return result


def evaluate(blocks, batches, device, dataset: str, classes: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    for block in blocks:
        block.eval()
    losses: list[float] = []
    accuracies: list[float] = []
    f1s: list[float] = []
    with torch.no_grad():
        for values, labels in batches:
            values = values.to(device)
            labels = labels.to(device)
            output = instrumentation._forward(blocks, values, dataset)["logits"]
            # output is [batch, classes, sequence].
            flat = output.permute(0, 2, 1).reshape(-1, classes)
            target = labels.reshape(-1).long()
            losses.append(float(F.cross_entropy(flat, target).item()))
            acc, mf1, _count = _metrics_from_logits(flat, labels, classes)
            accuracies.append(acc)
            f1s.append(mf1)
    if not losses:
        raise ValueError("evaluation split produced no batches")
    return {"loss": float(sum(losses) / len(losses)), "acc": float(sum(accuracies) / len(accuracies)), "mf1": float(sum(f1s) / len(f1s))}


def run_probe(
    initial_blocks,
    train_batches,
    eval_batches,
    *,
    device,
    dataset: str,
    classes: int,
    steps: list[int],
    lr: float,
    weight_decay: float,
    retention_batches=None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    blocks = copy.deepcopy(initial_blocks)
    for block in blocks:
        block.to(device).eval()
    parameters = [parameter for block in blocks for parameter in block.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    rows: list[dict[str, Any]] = []

    def record(step: int, train_loss: float | None = None):
        values = evaluate(blocks, eval_batches, device, dataset, classes)
        row = {"step": int(step), "loss": values["loss"], "acc": values["acc"], "mf1": values["mf1"], "train_loss": train_loss}
        if retention_batches is not None:
            retained = evaluate(blocks, retention_batches, device, dataset, classes)
            row.update({
                "retention_loss": retained["loss"],
                "retention_acc": retained["acc"],
                "retention_mf1": retained["mf1"],
            })
        rows.append(row)

    record(0)
    iterator = iter(train_batches)
    for step in range(1, steps[-1] + 1):
        try:
            values, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_batches)
            values, labels = next(iterator)
        for block in blocks:
            block.train()
        values, labels = values.to(device), labels.to(device)
        output = instrumentation._forward(blocks, values, dataset)["logits"]
        flat = output.permute(0, 2, 1).reshape(-1, classes)
        loss = F.cross_entropy(flat, labels.reshape(-1).long())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in steps:
            record(step, float(loss.detach().item()))
    result = _summarize_curve(rows, ("loss", "acc", "mf1"))
    if retention_batches is not None:
        retention = _summarize_curve(rows, ("retention_loss", "retention_acc", "retention_mf1"))
        # Positive drop means that the fixed-budget adaptation harmed the
        # retained task.  This is a retention diagnostic, not the LoP gap.
        retention["retention_loss_delta"] = retention["retention_loss_final"] - retention["retention_loss_initial"]
        retention["retention_acc_drop"] = retention["retention_acc_initial"] - retention["retention_acc_final"]
        retention["retention_mf1_drop"] = retention["retention_mf1_initial"] - retention["retention_mf1_final"]
        result["retention"] = retention
    result["curve"] = rows
    result["steps"] = steps
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brainuicl-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ISRUC", "FACED"), required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--fresh-checkpoint-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--checkpoint-stage", type=int, default=0)
    parser.add_argument("--method", default="finetune")
    parser.add_argument("--split", default="subject-eval")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--probe-steps", default="0,5,10")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--retention-data-root",
        type=Path,
        default=None,
        help="Optional labelled old-task evaluation root; no retention update is performed.",
    )
    parser.add_argument(
        "--retention-subject",
        type=int,
        action="append",
        default=[],
        help="Subject id to include in the fixed old-task retention set (repeatable).",
    )
    parser.add_argument(
        "--retention-max-files",
        type=int,
        default=0,
        help="Maximum files per retention subject; 0 uses --max-files.",
    )
    parser.add_argument(
        "--retention-batch-size",
        type=int,
        default=0,
        help="Retention evaluation batch size; 0 uses --batch-size.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.train_fraction < 1:
        raise SystemExit("train-fraction must be in (0,1)")
    steps = parse_steps(args.probe_steps)
    instrumentation._setup_brainuicl_import(args.brainuicl_root)
    import torch

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    classes = 9 if args.dataset == "FACED" else 5
    pairs = instrumentation._sample_files(args.data_root, args.subject, args.max_files)
    train_pairs, eval_pairs = split_pairs(pairs, args.train_fraction)
    train_batches = batches_from_files(train_pairs, args.batch_size, args.dataset)
    eval_batches = batches_from_files(eval_pairs, args.batch_size, args.dataset)
    retention_pairs: list[tuple[Path, Path]] = []
    if args.retention_subject:
        if args.retention_data_root is None:
            raise SystemExit("--retention-data-root is required when --retention-subject is supplied")
        retention_root = args.retention_data_root.resolve()
        if not retention_root.is_dir():
            raise SystemExit(f"retention data root does not exist: {retention_root}")
        retention_limit = args.retention_max_files if args.retention_max_files > 0 else args.max_files
        for retention_subject in sorted(set(args.retention_subject)):
            retention_pairs.extend(instrumentation._sample_files(retention_root, retention_subject, retention_limit))
        if not retention_pairs:
            raise SystemExit("retention set contains no sequence files")
    retention_batches = (
        batches_from_files(
            retention_pairs,
            args.retention_batch_size if args.retention_batch_size > 0 else args.batch_size,
            args.dataset,
        )
        if retention_pairs
        else None
    )
    blocks, checkpoint_paths = instrumentation._load_blocks(args.brainuicl_root, args.checkpoint_root, args.seed, device, args.dataset)
    old = run_probe(blocks, train_batches, eval_batches, device=device, dataset=args.dataset, classes=classes, steps=steps, lr=args.lr, weight_decay=args.weight_decay, retention_batches=retention_batches)
    if args.fresh_checkpoint_root is not None:
        fresh_blocks, fresh_paths = instrumentation._load_blocks(args.brainuicl_root, args.fresh_checkpoint_root, args.seed, device, args.dataset)
        fresh_mode = "checkpoint"
    else:
        torch.manual_seed(args.seed + 100000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + 100000)
        fresh_blocks = clone_and_reset(blocks)
        fresh_paths = []
        fresh_mode = "random-initialization"
    fresh = run_probe(fresh_blocks, train_batches, eval_batches, device=device, dataset=args.dataset, classes=classes, steps=steps, lr=args.lr, weight_decay=args.weight_decay, retention_batches=retention_batches)

    probe = {
        "protocol": "supervised-oracle-fixed-budget-heldout-v1",
        "fresh_mode": fresh_mode,
        "budget_steps": steps,
        "optimizer": {
            "name": "Adam",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "update_steps": int(steps[-1]),
            "scope": "target-train-only",
            "state_persisted": False,
            "state_interpretation": "probe-local optimizer state; not a checkpoint trajectory",
        },
        "train_sequences": len(train_pairs),
        "eval_sequences": len(eval_pairs),
        "old": old,
        "fresh": fresh,
        "loss_gap_final": float(old["loss_final"] - fresh["loss_final"]),
        "acc_gap_final": float(fresh["acc_final"] - old["acc_final"]),
        "mf1_gap_final": float(fresh["mf1_final"] - old["mf1_final"]),
        "loss_aulc_gap": float(old["loss_aulc"] - fresh["loss_aulc"]),
        "acc_aulc_gap": float(fresh["acc_aulc"] - old["acc_aulc"]),
        "mf1_aulc_gap": float(fresh["mf1_aulc"] - old["mf1_aulc"]),
    }
    if "retention" in old:
        probe["retention"] = {
            "protocol": "old-task-retention-eval-v1",
            "subjects": sorted(set(args.retention_subject)),
            "sequence_count": len(retention_pairs),
            "labels_loaded": True,
            "optimizer_update": False,
            "optimizer_scope": "target-train-only",
            "checkpoint": old["retention"],
            "fresh": fresh.get("retention"),
            "checkpoint_acc_drop": old["retention"]["retention_acc_drop"],
            "checkpoint_mf1_drop": old["retention"]["retention_mf1_drop"],
            "checkpoint_loss_delta": old["retention"]["retention_loss_delta"],
            "fresh_acc_drop": fresh["retention"]["retention_acc_drop"],
            "fresh_mf1_drop": fresh["retention"]["retention_mf1_drop"],
            "fresh_loss_delta": fresh["retention"]["retention_loss_delta"],
            "interpretation": "fixed old-task retention diagnostic; labels are evaluation-only and no retention optimizer update is performed; not a LoP outcome or BWT estimate",
        }
    context = {
        "dataset": args.dataset,
        "subject": str(args.subject),
        "method": args.method,
        "split": args.split,
        "probe_budget": steps[-1],
        "measurement_protocol": "supervised-oracle-fixed-budget-heldout-v1",
    }
    task = {
        "task": args.checkpoint_stage,
        "stage": args.checkpoint_stage,
        "subject": args.subject,
        "split": args.split,
        "probe_budget": steps[-1],
        "measurement_protocol": "supervised-oracle-fixed-budget-heldout-v1",
        "plasticity": {
            "acc_gain": old["acc_gain"],
            "mf1_gain": old["mf1_gain"],
            "loss_reduction": -old["loss_gain"],
            "aulc": old["acc_aulc"],
            "loss_aulc": old["loss_aulc"],
            "fresh_gap": probe["acc_gap_final"],
            "fresh_loss_gap": probe["loss_gap_final"],
        },
        "forgetting": probe.get("retention", {"status": "not-computed", "reason": "retention set was not supplied"}),
        "optimizer": probe["optimizer"],
        "probe": probe,
    }
    metrics: list[dict[str, Any]] = []
    # The normalizer handles tasks/probe curve rows and assigns their explicit
    # optimizer step; this direct envelope is useful for edgeforge-bundle-v1.
    def flatten(value: Any, prefix: str, step_override: int | None = None):
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics.append({"namespace": "raeeg.research", "name": prefix, "value": float(value), "step": args.checkpoint_stage if step_override is None else step_override, "unit": "ratio" if any(token in prefix for token in ("acc", "mf1", "gain", "gap")) else "scalar", "context": context})
        elif isinstance(value, dict):
            for key, item in value.items():
                flatten(item, f"{prefix}.{key}", step_override=step_override)
        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            for item in value:
                row_step = item.get("step", step_override if step_override is not None else args.checkpoint_stage)
                for key, item_value in item.items():
                    if key == "step":
                        continue
                    flatten(item_value, f"{prefix}.{key}", step_override=int(row_step))
    flatten(task["plasticity"], "task.plasticity")
    flatten(task["forgetting"], "task.forgetting")
    flatten(task["probe"], "task.probe")
    # Canonical/legacy aliases are both explicit in this bundle; stores that
    # use the old default outcome continue to work without guessing.
    metrics.append({"namespace": "raeeg.research", "name": "plasticity.acc_gain", "value": float(old["acc_gain"]), "step": args.checkpoint_stage, "unit": "ratio", "context": context})
    metrics.append({"namespace": "raeeg.research", "name": "task.plasticity.acc_gain", "value": float(old["acc_gain"]), "step": args.checkpoint_stage, "unit": "ratio", "context": context})
    result = {
        "schema_version": 1,
        "instrumentation": "brainuicl-fixed-budget-probe-v1",
        "status": "succeeded",
        "read_only": True,
        "config": {"dataset": args.dataset, "subject": args.subject, "seed": args.seed, "checkpoint_stage": args.checkpoint_stage, "method": args.method, "split": args.split, "probe_steps": steps, "lr": args.lr, "weight_decay": args.weight_decay, "device": str(device), "retention_data_root": (None if args.retention_data_root is None else str(args.retention_data_root.resolve())), "retention_subjects": sorted(set(args.retention_subject)), "retention_max_files": args.retention_max_files, "retention_batch_size": args.retention_batch_size},
        "source": {
            "checkpoint_digests": [{"path": str(path), "sha256": digest(path)} for path in checkpoint_paths],
            "fresh_checkpoint_digests": [{"path": str(path), "sha256": digest(path)} for path in fresh_paths],
            "input_files": [{"data": str(data), "label": str(label), "data_sha256": digest(data), "label_sha256": digest(label)} for data, label in pairs],
            "retention_input_files": [{"data": str(data), "label": str(label), "data_sha256": digest(data), "label_sha256": digest(label)} for data, label in retention_pairs],
        },
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)},
        "tasks": [task],
        "metrics": metrics,
        "summary": {"dataset": args.dataset, "method": args.method, "stage": args.checkpoint_stage, "fresh_mode": fresh_mode, "acc_gain": old["acc_gain"], "mf1_gain": old["mf1_gain"], "acc_gap_final": probe["acc_gap_final"], "loss_gap_final": probe["loss_gap_final"], "loss_aulc_gap": probe["loss_aulc_gap"], "retention_enabled": bool(retention_pairs), "retention_labels_loaded_for_eval": bool(retention_pairs), "retention_optimizer_update": False, "retention_acc_drop": (None if "retention" not in old else old["retention"]["retention_acc_drop"]), "retention_mf1_drop": (None if "retention" not in old else old["retention"]["retention_mf1_drop"]), "scientific_conclusion_allowed": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
