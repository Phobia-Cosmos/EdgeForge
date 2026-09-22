#!/usr/bin/env python3
"""Read-only alignment of continual-learning paper claims with RA-EEG results.

The script intentionally treats BrainUICL as an external source of truth.  It
does not alter experiment files; it emits a compact JSON/JSONL evidence table
that can be archived with an EdgeForge version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def walk_json(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.name in {"metrics.json", "summary.json", "RESULTS.json"}
    )


def flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten_numbers(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


def classify(path: Path) -> tuple[str, str]:
    text = str(path).lower()
    if "edgeforge_runs/lop" in text:
        return "LoP probe / spectral correlates", "raeeg-lop-posthoc-v1"
    if "attack_smoke" in text or "proxy_" in text or "dynamic_proxy" in text:
        return "poisoning or proxy shift", "proxy/attack (not clean baseline)"
    if "t2t" in text or "robust_feature" in text:
        return "monitoring or robust-feature defense", "defense exploratory"
    if "persist" in text or "order" in text or "trajectory" in text:
        return "order/persistence", "continual-learning trajectory"
    if "regularization" in text or "clean" in text or "aligned" in text:
        return "clean continual learning", "aligned or historical; inspect path"
    return "other", "unclassified"


def evidence_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in walk_json(root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        numbers = flatten_numbers(payload)
        family, protocol = classify(path)
        for metric in (
            "final_old_acc", "old_fr", "bwt_acc", "final_seen_acc",
            "mean_current_acc_gain", "robust_mean_protected_fraction",
            "t2t_detected_pairs", "t2t_rejected_updates", "effective_rank",
            "stable_rank", "sigma_max", "plasticity.acc_gain",
        ):
            matches = [(key, value) for key, value in numbers.items() if key.endswith(metric)]
            for key, value in matches:
                rows.append({
                    "paper_claim": {
                        "clean continual learning": "regularization/replay may trade stability, plasticity and forgetting",
                        "LoP probe / spectral correlates": "plasticity loss may co-vary with probe performance and representation spectrum",
                        "poisoning or proxy shift": "continual learners can be harmed by poisoned or shifted streams",
                        "monitoring or robust-feature defense": "monitoring/feature constraints may reject or reduce harmful updates",
                        "order/persistence": "task order and persistence change forgetting and transfer",
                    }.get(family, "result requires paper-specific interpretation"),
                    "protocol": protocol,
                    "dataset": "faced" if "/faced" in str(path).lower() else "isruc" if "/isruc" in str(path).lower() else "unknown",
                    "method": key.rsplit(".", 1)[0].split(".")[-1],
                    "metric": metric,
                    "value": value,
                    "evidence_path": str(path),
                    "comparability": "exploratory" if "smoke" in str(path).lower() or "lop" in str(path).lower() else "protocol-scoped",
                    "status": "observed",
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brainuicl", type=Path, default=Path("/home/undefined/Desktop/bci/code/tta_security/BrainUICL"))
    parser.add_argument("--out", type=Path, default=Path("logs/paper-claims-gaeeg-20260820.json"))
    args = parser.parse_args()
    rows = evidence_rows(args.brainuicl)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema_version": 1, "source": str(args.brainuicl), "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
