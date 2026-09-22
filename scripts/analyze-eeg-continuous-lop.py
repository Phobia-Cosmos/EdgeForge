#!/usr/bin/env python3
"""Profile the continuous EEG data and relate its features to LoP gaps.

This is a descriptive diagnostic.  It computes subject-level signal/label
features, joins them with the continuous runner's fresh-gap curves, and writes
JSON/Markdown artifacts.  It does not retrain models, alter data, or claim
causality from correlations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _subject_files(root: Path, group: str, subject: int) -> list[Path]:
    paths = sorted((root / group / str(subject) / "data").glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not paths:
        raise FileNotFoundError(f"no data for {group}/{subject}")
    return paths


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = (left + 1e-12) / (left.sum() + 1e-12 * len(left))
    right = (right + 1e-12) / (right.sum() + 1e-12 * len(right))
    midpoint = 0.5 * (left + right)
    return float(0.5 * np.sum(left * np.log(left / midpoint)) + 0.5 * np.sum(right * np.log(right / midpoint)))


def _entropy(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def profile_subject(root: Path, group: str, subject: int, classes: int = 5) -> dict[str, Any]:
    total_values = 0
    sum_values = 0.0
    sum_squares = 0.0
    sum_abs = 0.0
    channel_squares = np.zeros(8, dtype=np.float64)
    diff_squares = np.zeros(8, dtype=np.float64)
    diff_values = 0
    counts = np.zeros(classes, dtype=np.int64)
    train_counts = np.zeros(classes, dtype=np.int64)
    eval_counts = np.zeros(classes, dtype=np.int64)
    epoch_rms: list[float] = []
    files = _subject_files(root, group, subject)
    for data_path in files:
        label_path = data_path.parent.parent / "label" / data_path.name
        data = np.load(data_path, allow_pickle=False).astype(np.float64, copy=False)
        labels = np.load(label_path, allow_pickle=False).reshape(-1).astype(np.int64, copy=False)
        if data.shape != (20, 8, 3000) or labels.shape != (20,):
            raise ValueError(f"invalid pair {data_path}: data={data.shape}, labels={labels.shape}")
        flat = data.reshape(-1)
        total_values += flat.size
        sum_values += float(flat.sum())
        sum_squares += float(np.square(flat).sum())
        sum_abs += float(np.abs(flat).sum())
        channel_squares += np.square(data).sum(axis=(0, 2))
        differences = np.diff(data, axis=-1)
        diff_squares += np.square(differences).sum(axis=(0, 2))
        diff_values += int(differences.shape[0] * differences.shape[2])
        counts += np.bincount(labels, minlength=classes)
        train_counts += np.bincount(labels[:10], minlength=classes)
        eval_counts += np.bincount(labels[10:], minlength=classes)
        epoch_rms.extend(np.sqrt(np.mean(np.square(data), axis=(1, 2))).tolist())
    mean = sum_values / total_values
    variance = max(0.0, sum_squares / total_values - mean * mean)
    rms = math.sqrt(max(0.0, sum_squares / total_values))
    return {
        "group": group,
        "subject": int(subject),
        "files": len(files),
        "epochs": int(len(epoch_rms)),
        "class_counts": counts.tolist(),
        "train_class_counts": train_counts.tolist(),
        "eval_class_counts": eval_counts.tolist(),
        "class_entropy_bits": _entropy(counts),
        "train_class_entropy_bits": _entropy(train_counts),
        "eval_class_entropy_bits": _entropy(eval_counts),
        "train_eval_label_js": _js_divergence(train_counts, eval_counts),
        "train_majority_class_fraction": float(train_counts.max() / max(1, train_counts.sum())),
        "eval_majority_class_fraction": float(eval_counts.max() / max(1, eval_counts.sum())),
        "minority_class_count": int(counts.min()),
        "majority_class_fraction": float(counts.max() / max(1, counts.sum())),
        "mean": mean,
        "std": math.sqrt(variance),
        "rms": rms,
        "abs_mean": sum_abs / total_values,
        "channel_rms": np.sqrt(channel_squares / (len(epoch_rms) * 3000)).tolist(),
        "temporal_diff_rms": np.sqrt(diff_squares / diff_values).tolist(),
        "epoch_rms_mean": float(np.mean(epoch_rms)),
        "epoch_rms_std": float(np.std(epoch_rms)),
        "epoch_rms_min": float(np.min(epoch_rms)),
        "epoch_rms_max": float(np.max(epoch_rms)),
    }


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = float(rank)
    return result


def _correlation(left: list[float], right: list[float]) -> dict[str, float | None]:
    if len(left) != len(right) or len(left) < 3:
        return {"pearson": None, "spearman": None, "n": len(left)}
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return {"pearson": None, "spearman": None, "n": len(left)}
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(np.asarray(_rank(left)), np.asarray(_rank(right)))[0, 1]),
        "n": len(left),
    }


def _gap_rows(summary: dict[str, Any], profiles: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    target_subjects = [int(item) for item in metadata.get("target_subject_order", [])]
    rows: list[dict[str, Any]] = []
    for run in summary.get("runs", []):
        if not isinstance(run, dict):
            continue
        architecture = str(run.get("architecture"))
        seed = int(run["seed"])
        for index, stage in enumerate(run.get("stages", [])):
            subject = int(target_subjects[index])
            profile = profiles[("target", subject)]
            for gap in stage.get("gaps", []):
                budget = int(gap["step"])
                rows.append({
                    "architecture": architecture,
                    "seed": seed,
                    "stage": index,
                    "subject": subject,
                    "budget": budget,
                    "fresh_gap": float(gap["fresh_gap"]),
                    "feature_distance_to_source": float(profile["feature_distance_to_source"]),
                    "target_rms": float(profile["rms"]),
                    "target_epoch_rms_std": float(profile["epoch_rms_std"]),
                    "target_class_entropy_bits": float(profile["class_entropy_bits"]),
                    "target_train_eval_label_js": float(profile["train_eval_label_js"]),
                    "target_majority_class_fraction": float(profile["majority_class_fraction"]),
                    "target_eval_majority_class_fraction": float(profile["eval_majority_class_fraction"]),
                    "target_minority_class_count": int(profile["minority_class_count"]),
                })
    return rows


def analyze(summary_path: str | Path, data_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    summary_file = Path(summary_path).resolve()
    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    groups = {
        "source": [int(item) for item in metadata.get("source_subjects", [])],
        "target": [int(item) for item in metadata.get("target_subject_order", [])],
        "retention": [int(item) for item in metadata.get("retention_subjects", [])],
    }
    profiles: dict[tuple[str, int], dict[str, Any]] = {}
    for group, subjects in groups.items():
        for subject in subjects:
            profiles[(group, subject)] = profile_subject(root, group, subject)
    source_vectors = []
    for subject in groups["source"]:
        profile = profiles[("source", subject)]
        source_vectors.append(np.log10(np.asarray([profile["rms"], *profile["channel_rms"], *profile["temporal_diff_rms"]]) + 1e-30))
    source_array = np.asarray(source_vectors, dtype=np.float64)
    source_center = source_array.mean(axis=0)
    source_scale = np.maximum(source_array.std(axis=0), 1e-12)
    for (group, subject), profile in profiles.items():
        vector = np.log10(np.asarray([profile["rms"], *profile["channel_rms"], *profile["temporal_diff_rms"]]) + 1e-30)
        profile["feature_distance_to_source"] = float(np.sqrt(np.mean(np.square((vector - source_center) / source_scale))))
        profile["feature_distance_basis"] = "z-scored log10(global_rms, channel_rms[8], temporal_diff_rms[8]) vs source-subject center"
    rows = _gap_rows(summary, profiles)
    source_counts = np.sum([profiles[("source", subject)]["class_counts"] for subject in groups["source"]], axis=0).astype(int)
    source_fractions = source_counts / max(1, int(source_counts.sum()))
    source_majority_fraction = float(source_fractions.max())
    source_baselines = []
    for run in summary.get("runs", []):
        if not isinstance(run, dict):
            continue
        source_metrics = run.get("source") if isinstance(run.get("source"), dict) else {}
        accuracy = source_metrics.get("accuracy")
        macro_f1 = source_metrics.get("macro_f1")
        if not isinstance(accuracy, (int, float)):
            continue
        class_index = int(np.argmin(np.abs(source_fractions - float(accuracy))))
        class_fraction = float(source_fractions[class_index])
        one_class_macro_f1 = float((2.0 * class_fraction / (1.0 + class_fraction)) / len(source_counts))
        source_baselines.append({
            "architecture": str(run.get("architecture")),
            "seed": int(run.get("seed")),
            "accuracy": float(accuracy),
            "macro_f1": (None if not isinstance(macro_f1, (int, float)) else float(macro_f1)),
            "closest_constant_class": class_index,
            "constant_class_accuracy": class_fraction,
            "constant_class_macro_f1": one_class_macro_f1,
            "accuracy_delta_from_constant": float(accuracy) - class_fraction,
            "macro_f1_delta_from_constant": (None if not isinstance(macro_f1, (int, float)) else float(macro_f1) - one_class_macro_f1),
        })
    correlations: dict[str, Any] = {}
    for architecture in sorted({row["architecture"] for row in rows}):
        correlations[architecture] = {}
        for budget in sorted({row["budget"] for row in rows}):
            selected = [row for row in rows if row["architecture"] == architecture and row["budget"] == budget]
            correlations[architecture][str(budget)] = {
                feature: _correlation([float(row[feature]) for row in selected], [float(row["fresh_gap"]) for row in selected])
                for feature in (
                    "feature_distance_to_source",
                    "target_rms",
                    "target_epoch_rms_std",
                    "target_class_entropy_bits",
                    "target_train_eval_label_js",
                    "target_majority_class_fraction",
                    "target_eval_majority_class_fraction",
                    "target_minority_class_count",
                )
            }
    result = {
        "schema_version": 1,
        "analysis": "eeg-continuous-data-profile-v1",
        "summary_path": str(summary_file),
        "data_root": str(root),
        "groups": groups,
        "source_label_counts": source_counts.tolist(),
        "source_majority_fraction": source_majority_fraction,
        "source_baseline_comparison": source_baselines,
        "profiles": [profiles[key] for key in sorted(profiles, key=lambda item: (item[0], item[1]))],
        "gap_rows": rows,
        "gap_feature_correlations": correlations,
        "interpretation": "descriptive feature/gap association only; no causal or scientific LoP conclusion",
        "scientific_conclusion_allowed": False,
    }
    (output / "data-profile.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Continuous EEG data profile and LoP diagnostics",
        "",
        "This report describes subject-level signal/label properties and their association with fresh-gap. Correlations are exploratory and do not establish causality.",
        "",
        "## Subject profile",
        "",
        "| group | subject | epochs | RMS | epoch RMS SD | class entropy | train/eval JS | majority fraction | source distance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in result["profiles"]:
        lines.append(
            f"| {profile['group']} | {profile['subject']} | {profile['epochs']} | {profile['rms']:.3e} | "
            f"{profile['epoch_rms_std']:.3e} | {profile['class_entropy_bits']:.3f} | {profile['train_eval_label_js']:.4f} | "
            f"{profile['majority_class_fraction']:.3f} | {profile['feature_distance_to_source']:.3f} |"
        )
    lines.extend(["", "## Source learning baseline", "", f"Combined source labels are {result['source_label_counts']}; the majority-class accuracy baseline is {result['source_majority_fraction']:.4f}. These rows compare each source metric with the closest constant-class predictor.", "", "| architecture | seed | accuracy | macro-F1 | constant class | constant accuracy | accuracy delta | macro-F1 delta |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for item in result["source_baseline_comparison"]:
        lines.append(f"| {item['architecture']} | {item['seed']} | {item['accuracy']:.4f} | {item['macro_f1']:.4f} | {item['closest_constant_class']} | {item['constant_class_accuracy']:.4f} | {item['accuracy_delta_from_constant']:+.4f} | {item['macro_f1_delta_from_constant']:+.4f} |")
    lines.extend(["", "## Fresh-gap feature associations", "", "Each cell is Pearson / Spearman correlation over 24 stage-seed rows at that budget.", ""])
    for architecture, by_budget in correlations.items():
        lines.extend([f"### `{architecture}`", "", "| budget | feature distance | target RMS | epoch RMS SD | class entropy | train/eval JS | majority fraction | eval majority fraction | minority count |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for budget, values in by_budget.items():
            def fmt(item: dict[str, Any]) -> str:
                p, s = item["pearson"], item["spearman"]
                return "n/a" if p is None else f"{p:+.3f}/{s:+.3f}"
            lines.append("| " + " | ".join([budget, *(fmt(values[key]) for key in ("feature_distance_to_source", "target_rms", "target_epoch_rms_std", "target_class_entropy_bits", "target_train_eval_label_js", "target_majority_class_fraction", "target_eval_majority_class_fraction", "target_minority_class_count"))]) + " |")
        lines.append("")
    lines.extend(["## Limitations", "", "The data are a diversity-selected development subset. Subject-level features are aggregate summaries; they do not replace representation spectra, effective-rank measurements, gradient/Jacobian diagnostics, or a controlled shift experiment. All conclusions remain descriptive.", ""])
    (output / "DATA_PROFILE.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.summary, args.data_root, args.output_dir)
    print(json.dumps({"status": "ok", "profiles": len(result["profiles"]), "gap_rows": len(result["gap_rows"]), "output_dir": str(args.output_dir.resolve()), "scientific_conclusion_allowed": False}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
