#!/usr/bin/env python3
"""Subject identity probes at epoch and sequence granularity.

This companion experiment makes the closed/open-set boundary explicit.  A
deterministic fraction of subjects is enrolled; for each enrolled subject,
half of complete sequence/trial groups are train and the other half are held
out.  All groups from the remaining subjects are *unseen-subject* test data.
The probe can therefore report concrete subject-id accuracy only for enrolled
subjects, and unknown-subject rejection separately.

No sleep/emotion labels are loaded.  Transformer token probes are contextual:
an epoch token after attention was produced from all 20 tokens in its source
sequence.  Raw/frontend epoch probes are independent of the other epochs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subject_identity_probe import _load_blocks, _records  # noqa: E402


STAGES = (
    "raw_waveform",
    "raw_waveform_gain_normalized",
    "eeg_branch",
    "eog_branch",
    "fusion",
    "transformer_layer1",
    "transformer_layer2",
    "transformer_layer3",
    "classifier_input",
    "logits",
)


def _raw_epoch_representations(values: np.ndarray, points: int = 128) -> dict[str, np.ndarray]:
    """Return one vector per epoch, preserving the 20-epoch dimension."""

    raw = np.asarray(values, dtype=np.float32)
    epochs, channels, samples = raw.shape
    edges = np.linspace(0, samples, points + 1, dtype=np.int64)
    pooled = np.empty((epochs, channels, points), dtype=np.float32)
    for index in range(points):
        left, right = int(edges[index]), int(edges[index + 1])
        right = max(left + 1, right)
        pooled[:, :, index] = raw[:, :, left:right].mean(axis=-1)
    gain = np.sqrt(np.mean(np.square(raw), axis=(1, 2), keepdims=True)).astype(np.float32)
    gain = np.maximum(gain, 1e-12)
    return {
        "raw_waveform": pooled.reshape(epochs, -1),
        "raw_waveform_gain_normalized": (pooled / gain).reshape(epochs, -1),
    }


def _encoded_epoch_representations(blocks, values: np.ndarray, dataset: str, torch) -> dict[str, np.ndarray]:
    """Extract token-level taps; attention taps remain sequence-contextual."""

    channels = 8 if dataset == "ISRUC" else 32
    samples = 3000 if dataset == "ISRUC" else 2500
    flat = values.reshape(20, channels, samples)
    if dataset == "ISRUC":
        eeg = flat[:, 2:]
        eog = flat[:, :2]
        eeg_conv = blocks[0].FEBlock_EEG(eeg)
        eog_conv = blocks[0].FEBlock_EOG(eog)
        eeg_pooled = blocks[0].avg(eeg_conv).reshape(20, 512)
        eog_pooled = blocks[0].avg(eog_conv).reshape(20, 512)
        fused = blocks[0].fusion(torch.cat((eeg_pooled[:, None, :], eog_pooled[:, None, :]), dim=2)).reshape(20, 512)
    else:
        eog = flat[:, :1]
        eeg = flat[:, 1:]
        conv = blocks[0].features(torch.cat((eog, eeg), dim=1))
        fused = blocks[0].avg(conv).reshape(20, 512)
        eeg_pooled = fused
        eog_pooled = fused
    # Keep the original 20-token sequence as a batch of one.  This is what
    # makes the Transformer taps contextual: each epoch token can attend to
    # the other 19 epochs from the same complete group.
    encoded = fused[None, :, :]
    stage_values: dict[str, torch.Tensor] = {
        "eeg_branch": eeg_pooled,
        "eog_branch": eog_pooled,
        "fusion": fused,
    }
    for index in range(1, 4):
        encoded = blocks[1].encoder.encoder(encoded)
        stage_values[f"transformer_layer{index}"] = encoded[0]
    classifier_input = blocks[2].sleep_stage_mlp(encoded)
    logits = blocks[2].sleep_stage_classifier(classifier_input).permute(0, 2, 1).contiguous()
    stage_values["classifier_input"] = classifier_input[0]
    stage_values["logits"] = logits[0].transpose(0, 1)
    return {name: tensor.detach().float().cpu().numpy() for name, tensor in stage_values.items()}


def _known_open_split(rows: list[tuple[int, str, Path]], fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    """Return group-level train, known-test, unknown-test masks."""

    by_subject: dict[int, list[int]] = {}
    for index, (subject, _group, _path) in enumerate(rows):
        by_subject.setdefault(subject, []).append(index)
    subjects = np.asarray(sorted(by_subject), dtype=np.int64)
    if not 0 < fraction <= 1:
        raise ValueError("known subject fraction must be in (0, 1]")
    known_count = max(2, min(len(subjects), int(round(len(subjects) * fraction))))
    order = subjects.copy()
    np.random.default_rng(seed).shuffle(order)
    known = sorted(int(x) for x in order[:known_count])
    unknown = sorted(int(x) for x in order[known_count:])
    train = np.zeros(len(rows), dtype=bool)
    known_test = np.zeros(len(rows), dtype=bool)
    unknown_test = np.zeros(len(rows), dtype=bool)
    for subject in known:
        indices = np.asarray(by_subject[subject], dtype=np.int64)
        np.random.default_rng(seed + subject * 1009).shuffle(indices)
        cutoff = max(1, min(len(indices) - 1, len(indices) // 2))
        train[indices[:cutoff]] = True
        known_test[indices[cutoff:]] = True
    for subject in unknown:
        unknown_test[np.asarray(by_subject[subject], dtype=np.int64)] = True
    if not train.any() or not known_test.any() or (unknown and not unknown_test.any()):
        raise RuntimeError("open-set split needs at least two groups for known subjects and one unseen subject")
    return train, known_test, unknown_test, known, unknown


def _probe(matrix: np.ndarray, subjects: np.ndarray, train: np.ndarray, known_test: np.ndarray, unknown_test: np.ndarray, probe_dim: int) -> dict[str, Any]:
    labels = np.asarray(sorted(np.unique(subjects[train])), dtype=np.int64)
    steps = [StandardScaler()]
    if probe_dim > 0 and matrix.shape[1] > probe_dim:
        steps.append(PCA(n_components=min(probe_dim, matrix.shape[1], max(1, int(train.sum()) - 1)), svd_solver="randomized", random_state=0))
    steps.append(LogisticRegression(max_iter=350, solver="lbfgs"))
    model = make_pipeline(*steps)
    model.fit(matrix[train], subjects[train])
    known_prob = model.predict_proba(matrix[known_test])
    known_prediction = model.classes_[known_prob.argmax(axis=1)]
    train_prob = model.predict_proba(matrix[train])
    unknown_prob = model.predict_proba(matrix[unknown_test]) if unknown_test.any() else np.zeros((0, len(labels)))
    known_confidence = known_prob.max(axis=1)
    unknown_confidence = unknown_prob.max(axis=1)
    # A threshold is calibrated from enrollment confidence only.  It rejects
    # the lowest 5% of enrollment scores, avoiding a test-set threshold leak.
    threshold = float(np.quantile(train_prob.max(axis=1), 0.05))
    unknown_score = np.concatenate([1.0 - known_confidence, 1.0 - unknown_confidence])
    binary_target = np.concatenate([np.zeros(len(known_confidence)), np.ones(len(unknown_confidence))])
    result: dict[str, Any] = {
        "feature_dim": int(matrix.shape[1]),
        "probe_dim": int(probe_dim),
        "known_subject_count": int(len(labels)),
        "unknown_subject_count": int(len(np.unique(subjects[unknown_test]))),
        "known_test_top1": float(accuracy_score(subjects[known_test], known_prediction)),
        "known_test_balanced_top1": float(balanced_accuracy_score(subjects[known_test], known_prediction)),
        "known_test_top5": float(top_k_accuracy_score(subjects[known_test], known_prob, k=min(5, len(labels)), labels=labels)),
        "chance_top1_known": float(1.0 / len(labels)),
        "unknown_predicted_as_known_fraction": float((unknown_confidence >= threshold).mean()) if unknown_test.any() else None,
        "unknown_rejection_rate_at_enrollment_5pct_threshold": float((unknown_confidence < threshold).mean()) if unknown_test.any() else None,
        "unknown_score_auroc": float(roc_auc_score(binary_target, unknown_score)) if len(np.unique(binary_target)) == 2 else None,
        "enrollment_confidence_threshold": threshold,
        "train_groups": int(train.sum()),
        "known_test_groups": int(known_test.sum()),
        "unknown_test_groups": int(unknown_test.sum()),
    }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    rows = _records(args.data_root.resolve(), args.dataset)
    if args.max_sequences > 0 and args.max_sequences < len(rows):
        # Round-robin groups over subjects while keeping two groups per subject
        # whenever possible, so smoke runs have a valid train/test split.
        by_subject: dict[int, list[tuple[int, str, Path]]] = {}
        for row in rows:
            by_subject.setdefault(row[0], []).append(row)
        selected: list[tuple[int, str, Path]] = []
        for subject in sorted(by_subject):
            if len(selected) + 2 > args.max_sequences:
                break
            selected.extend(by_subject[subject][:2])
        cursor = {subject: 2 for subject in by_subject if len(by_subject[subject]) >= 2}
        while len(selected) < args.max_sequences:
            advanced = False
            for subject in sorted(by_subject):
                index = cursor.get(subject, 0)
                if index < len(by_subject[subject]) and len(selected) < args.max_sequences:
                    selected.append(by_subject[subject][index])
                    cursor[subject] = index + 1
                    advanced = True
            if not advanced:
                break
        rows = selected
    train_groups, known_test_groups, unknown_test_groups, known_subjects, unknown_subjects = _known_open_split(rows, args.known_subject_fraction, args.seed)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    blocks = _load_blocks(args.brainuicl_root, args.checkpoint_root, args.dataset, args.checkpoint_seed, device)
    stage_names = tuple(name.strip() for name in args.stages.split(",") if name.strip())
    invalid = sorted(set(stage_names) - set(STAGES))
    if invalid:
        raise ValueError(f"unknown stages: {invalid}")
    collected: dict[str, list[np.ndarray]] = {name: [] for name in stage_names}
    subjects: list[int] = []
    for index, (subject, _group, path) in enumerate(rows, start=1):
        values = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        expected = (20, 8, 3000) if args.dataset == "ISRUC" else (20, 32, 2500)
        if tuple(values.shape) != expected:
            raise ValueError(f"expected {path} to have shape {expected}, got {values.shape}")
        raw = _raw_epoch_representations(values)
        with torch.inference_mode():
            encoded = _encoded_epoch_representations(blocks, torch.from_numpy(values).to(device), args.dataset, torch)
        for name in stage_names:
            vectors = raw[name] if name in raw else encoded[name]
            collected[name].append(np.asarray(vectors, dtype=np.float32))
        subjects.append(subject)
        if index % 100 == 0 or index == len(rows):
            print(json.dumps({"event": "sequence_done", "index": index, "total": len(rows), "dataset": args.dataset}), flush=True)

    group_subjects = np.asarray(subjects, dtype=np.int64)
    units = ("epoch", "sequence") if args.unit == "both" else (args.unit,)
    results: dict[str, Any] = {}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for unit in units:
        unit_results: dict[str, Any] = {}
        for name in stage_names:
            if unit == "epoch":
                matrix = np.concatenate(collected[name], axis=0)
                labels = np.repeat(group_subjects, 20)
                train = np.repeat(train_groups, 20)
                known_test = np.repeat(known_test_groups, 20)
                unknown_test = np.repeat(unknown_test_groups, 20)
            else:
                matrix = np.asarray([np.median(vectors, axis=0) for vectors in collected[name]], dtype=np.float32)
                labels = group_subjects
                train, known_test, unknown_test = train_groups, known_test_groups, unknown_test_groups
            unit_results[name] = _probe(matrix, labels, train, known_test, unknown_test, args.probe_dim)
            if args.save_vectors:
                np.save(output / f"{unit}-{name}-vectors.npy", matrix)
        results[unit] = unit_results

    aggregation_curve: dict[str, Any] = {}
    curve_ks = tuple(sorted({int(value) for value in args.aggregation_ks.split(",") if value.strip()})) if args.aggregation_ks.strip() else ()
    curve_stages = tuple(name.strip() for name in args.curve_stages.split(",") if name.strip())
    invalid_curve = sorted(set(curve_stages) - set(stage_names))
    if invalid_curve:
        raise ValueError(f"curve stages must be included in --stages: {invalid_curve}")
    for k in curve_ks:
        if k < 1 or k > 20:
            raise ValueError("aggregation k must be between 1 and 20")
        k_results: dict[str, Any] = {}
        for name in curve_stages:
            representatives = []
            for group_index, vectors in enumerate(collected[name]):
                order = np.random.default_rng(args.seed + group_index * 9176 + k * 37).permutation(20)[:k]
                representatives.append(np.median(vectors[order], axis=0))
            matrix = np.asarray(representatives, dtype=np.float32)
            k_results[name] = _probe(matrix, group_subjects, train_groups, known_test_groups, unknown_test_groups, args.probe_dim)
        aggregation_curve[str(k)] = k_results

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    summary = {
        "schema_version": 1,
        "created_at": created_at,
        "experiment": "subject-identity-granularity-open-set-v1",
        "dataset": args.dataset,
        "data_root": str(args.data_root.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "checkpoint_seed": int(args.checkpoint_seed),
        "device": str(device),
        "sequence_count": int(len(rows)),
        "epoch_count": int(len(rows) * 20),
        "known_subjects": known_subjects,
        "unknown_subjects": unknown_subjects,
        "known_subject_fraction": float(args.known_subject_fraction),
        "target": "subject_id_only; source sleep/emotion labels were not loaded",
        "group_split": "known subjects: half complete groups train and half known-test; unselected subjects: all groups unknown-test",
        "context_note": "raw/frontend epoch vectors are independent per epoch; transformer-layer epoch vectors are contextualized by all 20 tokens in the source sequence",
        "probe": "StandardScaler -> randomized PCA(probe_dim) -> multinomial LogisticRegression",
        "stages": results,
        "aggregation_curve": aggregation_curve,
        "aggregation_curve_definition": "For each complete group, choose k of 20 epochs with a deterministic seed and median-pool them; train/test remain group-disjoint.",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__},
        "scientific_conclusion_allowed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": created_at,
        "script": "scripts/subject_identity_granularity.py",
        "experiment": summary["experiment"],
        "dataset": args.dataset,
        "arguments": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "sequence_count": summary["sequence_count"],
        "epoch_count": summary["epoch_count"],
        "known_subject_count": len(summary["known_subjects"]),
        "unknown_subject_count": len(summary["unknown_subjects"]),
        "runtime": summary["runtime"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        f"# {args.dataset} subject identity granularity/open-set probe",
        "",
        f"This run contains {len(rows):,} complete groups ({len(rows) * 20:,} epochs). It enrolls {len(known_subjects)} subjects and reserves {len(unknown_subjects)} subjects as unseen identities.",
        "",
        "Known subjects use half of their complete groups for enrollment and half for held-out identity testing. Unseen subjects are never present in the probe training classes; their concrete subject id is therefore not a valid prediction target. Unknown rejection is reported separately.",
        "",
        "| unit | stage | known top-1 | known top-5 | chance | unknown rejection | unknown AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for unit, stage_results in results.items():
        for name in stage_names:
            row = stage_results[name]
            report.append(f"| {unit} | {name} | {row['known_test_top1']:.4f} | {row['known_test_top5']:.4f} | {row['chance_top1_known']:.4f} | {row['unknown_rejection_rate_at_enrollment_5pct_threshold'] if row['unknown_rejection_rate_at_enrollment_5pct_threshold'] is not None else float('nan'):.4f} | {row['unknown_score_auroc'] if row['unknown_score_auroc'] is not None else float('nan'):.4f} |")
    if aggregation_curve:
        report.extend(["", "## Epoch-count aggregation curve", "", "| k epochs per group | stage | known top-1 | known top-5 |", "| ---: | --- | ---: | ---: |"])
        for k, stage_results in aggregation_curve.items():
            for name in curve_stages:
                row = stage_results[name]
                report.append(f"| {k} | {name} | {row['known_test_top1']:.4f} | {row['known_test_top5']:.4f} |")
    report.extend([
        "",
        "A high known top-1 means a held-out group from an enrolled subject can be assigned its concrete subject id. For an unseen subject, a classifier trained only on known ids cannot produce the true new id; unknown rejection/AUROC instead quantify whether it can detect that the sample is outside enrollment.",
        "",
        "This is a subject-information audit, not a LoP outcome. Transformer epoch taps are sequence-contextual and should not be described as a strictly isolated single-epoch measurement.",
    ])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ISRUC", "FACED"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--brainuicl-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-seed", type=int, default=4321)
    parser.add_argument("--batch-size", type=int, default=1, help="reserved for interface compatibility; extraction is sequence-wise")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--known-subject-fraction", type=float, default=0.8)
    parser.add_argument("--unit", choices=("epoch", "sequence", "both"), default="both")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--probe-dim", type=int, default=128)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--save-vectors", action="store_true")
    parser.add_argument("--aggregation-ks", default="1,2,5,10,20", help="comma-separated epoch counts for group aggregation curves; empty disables")
    parser.add_argument("--curve-stages", default="raw_waveform,eeg_branch,transformer_layer1,classifier_input")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    summary = run(args)
    print(json.dumps({"status": "ok", "dataset": summary["dataset"], "sequence_count": summary["sequence_count"], "known_subject_count": len(summary["known_subjects"]), "unknown_subject_count": len(summary["unknown_subjects"]), "output": str(args.output.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
