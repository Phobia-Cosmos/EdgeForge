#!/usr/bin/env python3
"""Audit signal usability of a derived EEG condition against clean data.

This is a read-only paired audit. It checks shape/finiteness, byte-identical
labels, waveform correlation, RMS ratio and normalized 0.5--40 Hz spectral
Jensen-Shannon distance. It does not assess clinical validity or task utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import welch


FS_HZ = 100.0


def _files(root: Path, group: str, subject: int) -> list[Path]:
    return sorted((root / group / str(subject) / "data").glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def _subjects(root: Path, group: str) -> list[int]:
    return sorted(int(p.name) for p in (root / group).iterdir() if p.is_dir() and p.name.isdigit())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _js(left: np.ndarray, right: np.ndarray) -> float:
    left = (left + 1e-30) / (left.sum() + 1e-30 * len(left))
    right = (right + 1e-30) / (right.sum() + 1e-30 * len(right))
    midpoint = 0.5 * (left + right)
    return float(0.5 * np.sum(left * np.log(left / midpoint)) + 0.5 * np.sum(right * np.log(right / midpoint)))


def _normalized_psd(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channel_psds = []
    frequencies = None
    for channel in values.reshape(-1, values.shape[-1]):
        frequencies, power = welch(channel, fs=FS_HZ, nperseg=1024)
        channel_psds.append(power)
    if frequencies is None:
        raise ValueError("empty EEG array")
    power = np.mean(np.asarray(channel_psds), axis=0)
    mask = (frequencies >= 0.5) & (frequencies <= 40.0)
    frequencies = frequencies[mask]
    power = power[mask]
    power = power / max(float(np.trapezoid(power, frequencies)), 1e-30)
    return frequencies, power


def _pair(clean: Path, derived: Path) -> dict[str, Any]:
    clean_values = np.load(clean, allow_pickle=False)
    derived_values = np.load(derived, allow_pickle=False)
    clean_label = clean.parent.parent / "label" / clean.name
    derived_label = derived.parent.parent / "label" / derived.name
    shape_ok = clean_values.shape == derived_values.shape
    finite = bool(np.isfinite(derived_values).all())
    data_byte_identical = _sha256(clean) == _sha256(derived)
    labels_equal = _sha256(clean_label) == _sha256(derived_label)
    clean_flat = clean_values.astype(np.float64, copy=False).reshape(-1)
    derived_flat = derived_values.astype(np.float64, copy=False).reshape(-1)
    if clean_flat.size and np.std(clean_flat) > 0.0 and np.std(derived_flat) > 0.0:
        correlation = float(np.corrcoef(clean_flat, derived_flat)[0, 1])
    else:
        correlation = 1.0 if np.array_equal(clean_values, derived_values) else 0.0
    clean_rms = float(np.sqrt(np.mean(np.square(clean_flat))))
    derived_rms = float(np.sqrt(np.mean(np.square(derived_flat))))
    _, clean_psd = _normalized_psd(clean_values)
    _, derived_psd = _normalized_psd(derived_values)
    return {
        "file": clean.name,
        "shape": list(clean_values.shape),
        "shape_ok": shape_ok,
        "finite": finite,
        "labels_byte_identical": labels_equal,
        "data_byte_identical": data_byte_identical,
        "clean_rms": clean_rms,
        "derived_rms": derived_rms,
        "rms_ratio": derived_rms / max(clean_rms, 1e-30),
        "correlation": correlation,
        "spectral_js": _js(clean_psd, derived_psd),
        "clean_abs_max": float(np.max(np.abs(clean_values))),
        "derived_abs_max": float(np.max(np.abs(derived_values))),
    }


def audit(clean_root: str | Path, derived_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    clean = Path(clean_root).resolve()
    derived = Path(derived_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for group in ("source", "target", "retention"):
        for subject in _subjects(clean, group):
            for clean_path in _files(clean, group, subject):
                derived_path = derived / group / str(subject) / "data" / clean_path.name
                if not derived_path.is_file():
                    raise FileNotFoundError(derived_path)
                row = _pair(clean_path, derived_path)
                row.update({"group": group, "subject": subject})
                rows.append(row)
    correlation = np.asarray([row["correlation"] for row in rows], dtype=np.float64)
    spectral_js = np.asarray([row["spectral_js"] for row in rows], dtype=np.float64)
    rms_ratio = np.asarray([row["rms_ratio"] for row in rows], dtype=np.float64)
    labels_ok = all(row["labels_byte_identical"] for row in rows)
    finite_ok = all(row["finite"] and row["shape_ok"] for row in rows)
    groups: dict[str, Any] = {}
    for group in ("source", "target", "retention"):
        group_rows = [row for row in rows if row["group"] == group]
        if not group_rows:
            continue
        group_correlations = np.asarray([row["correlation"] for row in group_rows], dtype=np.float64)
        group_spectral_js = np.asarray([row["spectral_js"] for row in group_rows], dtype=np.float64)
        group_rms = np.asarray([row["rms_ratio"] for row in group_rows], dtype=np.float64)
        groups[group] = {
            "files": len(group_rows),
            "data_byte_identical": bool(all(row["data_byte_identical"] for row in group_rows)),
            "changed_files": int(sum(not row["data_byte_identical"] for row in group_rows)),
            "correlation_min": float(group_correlations.min()),
            "correlation_median": float(np.median(group_correlations)),
            "spectral_js_max": float(group_spectral_js.max()),
            "spectral_js_median": float(np.median(group_spectral_js)),
            "rms_ratio_min": float(group_rms.min()),
            "rms_ratio_max": float(group_rms.max()),
        }
    changed_rows = [row for row in rows if not row["data_byte_identical"]]
    changed_correlations = np.asarray([row["correlation"] for row in changed_rows], dtype=np.float64)
    changed_spectral_js = np.asarray([row["spectral_js"] for row in changed_rows], dtype=np.float64)
    changed_rms = np.asarray([row["rms_ratio"] for row in changed_rows], dtype=np.float64)
    result = {
        "schema_version": 1,
        "analysis": "eeg-condition-quality-audit-v1",
        "clean_root": str(clean),
        "derived_root": str(derived),
        "files": len(rows),
        "labels_byte_identical": labels_ok,
        "shape_and_finite_ok": finite_ok,
        "correlation_min": float(correlation.min()),
        "correlation_median": float(np.median(correlation)),
        "spectral_js_max": float(spectral_js.max()),
        "spectral_js_median": float(np.median(spectral_js)),
        "rms_ratio_min": float(rms_ratio.min()),
        "rms_ratio_max": float(rms_ratio.max()),
        "groups": groups,
        "changed_data_files": len(changed_rows),
        "changed_data_correlation_min": (float(changed_correlations.min()) if len(changed_rows) else None),
        "changed_data_correlation_median": (float(np.median(changed_correlations)) if len(changed_rows) else None),
        "changed_data_spectral_js_max": (float(changed_spectral_js.max()) if len(changed_rows) else None),
        "changed_data_spectral_js_median": (float(np.median(changed_spectral_js)) if len(changed_rows) else None),
        "changed_data_rms_ratio_min": (float(changed_rms.min()) if len(changed_rows) else None),
        "changed_data_rms_ratio_max": (float(changed_rms.max()) if len(changed_rows) else None),
        "rows": rows,
        "preferred_waveform_correlation_threshold": 0.98,
        "scientific_conclusion_allowed": False,
    }
    result["preferred_waveform_gate_pass"] = bool(labels_ok and finite_ok and result["correlation_min"] >= 0.98)
    (output / "quality-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# EEG condition quality audit", "", f"Clean: `{clean}`", f"Derived: `{derived}`", "", "| check | value |", "| --- | --- |", f"| files | {len(rows)} |", f"| labels byte-identical | `{labels_ok}` |", f"| shape and finite | `{finite_ok}` |", f"| all-file correlation min / median | {result['correlation_min']:.6f} / {result['correlation_median']:.6f} |", f"| all-file spectral JS max / median | {result['spectral_js_max']:.6f} / {result['spectral_js_median']:.6f} |", f"| all-file RMS ratio min / max | {result['rms_ratio_min']:.6f} / {result['rms_ratio_max']:.6f} |", f"| changed data files | {result['changed_data_files']} |", f"| changed-file correlation min / median | {result['changed_data_correlation_min'] if result['changed_data_correlation_min'] is not None else 'n/a'} / {result['changed_data_correlation_median'] if result['changed_data_correlation_median'] is not None else 'n/a'} |", f"| changed-file spectral JS max / median | {result['changed_data_spectral_js_max'] if result['changed_data_spectral_js_max'] is not None else 'n/a'} / {result['changed_data_spectral_js_median'] if result['changed_data_spectral_js_median'] is not None else 'n/a'} |", f"| changed-file RMS ratio min / max | {result['changed_data_rms_ratio_min'] if result['changed_data_rms_ratio_min'] is not None else 'n/a'} / {result['changed_data_rms_ratio_max'] if result['changed_data_rms_ratio_max'] is not None else 'n/a'} |", f"| preferred waveform gate (corr >= 0.98) | `{result['preferred_waveform_gate_pass']}` |", "", "Group-level details are stored in `quality-audit.json`; changed-file statistics isolate the intended target-only perturbation. This audit measures preservation of the recorded signal representation; it does not establish clinical validity, label quality, or LoP.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.clean_root, args.derived_root, args.output_dir)
    print(json.dumps({"status": "ok", "files": result["files"], "preferred_waveform_gate_pass": result["preferred_waveform_gate_pass"], "scientific_conclusion_allowed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
