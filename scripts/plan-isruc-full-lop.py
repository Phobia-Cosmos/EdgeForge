#!/usr/bin/env python3
"""Build a deterministic, subject-disjoint full ISRUC LoP experiment plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "edgeforge.isruc-full-lop-plan.v1"


def _npy_sort_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = (left.astype(np.float64) + 1e-12) / (float(left.sum()) + 1e-12 * len(left))
    right = (right.astype(np.float64) + 1e-12) / (float(right.sum()) + 1e-12 * len(right))
    midpoint = 0.5 * (left + right)
    return float(0.5 * np.sum(left * np.log(left / midpoint)) + 0.5 * np.sum(right * np.log(right / midpoint)))


def inspect_subject(root: Path, subject: int) -> dict[str, Any]:
    subject_root = root / str(subject)
    data_files = sorted((subject_root / "data").glob("*.npy"), key=_npy_sort_key)
    label_files = sorted((subject_root / "label").glob("*.npy"), key=_npy_sort_key)
    if not data_files:
        raise ValueError(f"subject {subject} has no data files")
    data_names = {path.name for path in data_files}
    label_names = {path.name for path in label_files}
    missing_labels = sorted(data_names - label_names)
    missing_data = sorted(label_names - data_names)
    if missing_labels:
        raise FileNotFoundError(f"missing label pair: {subject_root / 'label' / missing_labels[0]}")
    if missing_data:
        raise FileNotFoundError(f"missing data pair: {subject_root / 'data' / missing_data[0]}")
    counts = np.zeros(5, dtype=np.int64)
    pair_digests: list[tuple[str, str]] = []
    payload_bytes = 0
    for data_path in data_files:
        label_path = subject_root / "label" / data_path.name
        if not label_path.is_file():
            raise FileNotFoundError(f"missing label pair: {label_path}")
        data = np.load(data_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(label_path, allow_pickle=False).reshape(-1)
        if data.shape != (20, 8, 3000) or labels.shape != (20,):
            raise ValueError(f"invalid ISRUC pair {data_path}: data={data.shape}, labels={labels.shape}")
        if np.any((labels < 0) | (labels > 4)):
            raise ValueError(f"labels outside 0..4: {label_path}")
        counts += np.bincount(labels.astype(np.int64), minlength=5)
        pair_digests.append((_sha256(data_path), _sha256(label_path)))
        payload_bytes += data_path.stat().st_size + label_path.stat().st_size
    digest_input = "\n".join(
        f"{path.name}:{path.stat().st_size}:{data_digest}:{label_digest}"
        for path, (data_digest, label_digest) in zip(data_files, pair_digests)
    )
    probabilities = counts / max(1, int(counts.sum()))
    entropy = -float(np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])))
    return {
        "subject": int(subject),
        "files": len(data_files),
        "epochs": int(counts.sum()),
        "payload_bytes": int(payload_bytes),
        "class_counts": counts.tolist(),
        "class_entropy_bits": entropy,
        "pair_manifest_sha256": hashlib.sha256(digest_input.encode("utf-8")).hexdigest(),
    }


def build_plan(
    source_root: str | Path,
    *,
    split_seed: int = 20260904,
    source_count: int = 30,
    target_count: int = 50,
    portable: bool = False,
) -> dict[str, Any]:
    root = Path(source_root).resolve()
    subjects = sorted(int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    if source_count < 1 or target_count < 2 or source_count + target_count >= len(subjects):
        raise ValueError("role counts must leave at least one retention subject and two target subjects")
    profiles = [inspect_subject(root, subject) for subject in subjects]
    profile_by_subject = {int(row["subject"]): row for row in profiles}

    shuffled = list(subjects)
    random.Random(int(split_seed)).shuffle(shuffled)
    source_subjects = sorted(shuffled[:source_count])
    target_subjects = sorted(shuffled[source_count : source_count + target_count])
    retention_subjects = sorted(shuffled[source_count + target_count :])
    if set(source_subjects) & set(target_subjects) or set(source_subjects) & set(retention_subjects) or set(target_subjects) & set(retention_subjects):
        raise AssertionError("subject roles unexpectedly overlap")

    source_counts = np.sum([profile_by_subject[item]["class_counts"] for item in source_subjects], axis=0)
    for subject in target_subjects:
        row = profile_by_subject[subject]
        row["label_js_from_source"] = _js_divergence(np.asarray(row["class_counts"]), source_counts)
    random_a = list(target_subjects)
    random_b = list(target_subjects)
    random.Random(int(split_seed) + 1).shuffle(random_a)
    random.Random(int(split_seed) + 2).shuffle(random_b)
    if random_b == random_a:
        random_b = random_b[1:] + random_b[:1]
    shift_ascending = sorted(target_subjects, key=lambda item: (profile_by_subject[item]["label_js_from_source"], item))
    orders = {
        "random-a": random_a,
        "random-b": random_b,
        "label-shift-ascending": shift_ascending,
        "label-shift-descending": list(reversed(shift_ascending)),
    }
    total_files = sum(int(row["files"]) for row in profiles)
    total_epochs = sum(int(row["epochs"]) for row in profiles)
    total_bytes = sum(int(row["payload_bytes"]) for row in profiles)
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": "ISRUC-Sleep Group I",
        "source_root": None if portable else str(root),
        "layout": "flat-subject-directories",
        "split_seed": int(split_seed),
        "subjects_available": subjects,
        "missing_subject_ids": sorted(set(range(min(subjects), max(subjects) + 1)) - set(subjects)),
        "roles": {
            "source": source_subjects,
            "target": target_subjects,
            "retention": retention_subjects,
        },
        "orders": orders,
        "profiles": profiles,
        "totals": {
            "subjects": len(subjects),
            "files": total_files,
            "epochs": total_epochs,
            "payload_bytes": total_bytes,
        },
        "protocol": {
            "source_split": "last 20% of files per source subject held out",
            "target_split": "first 10 epochs per file adapt; last 10 held out",
            "retention": "subject-disjoint fixed evaluation reservoir",
            "primary_outcome": "fresh_accuracy - warm_accuracy at fixed budget; positive means LoP direction",
            "primary_orders": ["random-a", "random-b"],
            "data_composition_orders": ["label-shift-ascending", "label-shift-descending"],
            "scientific_conclusion_allowed": False,
        },
    }
    digest_payload = dict(plan)
    digest_payload.pop("source_root", None)
    plan["plan_digest"] = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return plan


def materialize_view(plan: dict[str, Any], source_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in output.iterdir() if path.name != "PLAN.json"]
    if unexpected:
        raise FileExistsError(f"refusing to modify non-empty role view: {output}")
    for role, subjects in plan["roles"].items():
        for subject in subjects:
            link = output / role / str(subject)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to((source_root / str(subject)).resolve(), target_is_directory=True)
    (output / "PLAN.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260904)
    parser.add_argument("--source-count", type=int, default=30)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--portable", action="store_true")
    parser.add_argument("--materialize-view", type=Path)
    args = parser.parse_args()
    plan = build_plan(
        args.source_root,
        split_seed=args.split_seed,
        source_count=args.source_count,
        target_count=args.target_count,
        portable=args.portable,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.materialize_view:
        materialize_view(plan, args.source_root.resolve(), args.materialize_view.resolve())
    print(json.dumps({"status": "ok", "output": str(args.output.resolve()), "totals": plan["totals"], "role_counts": {key: len(value) for key, value in plan["roles"].items()}, "plan_digest": plan["plan_digest"]}, sort_keys=True))


if __name__ == "__main__":
    main()
