#!/usr/bin/env python3
"""Summarize fixed-probe subject-ID scores across continual-learning checkpoints."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


STAGES = ("eeg_branch", "transformer_layer1", "transformer_layer2", "transformer_layer3", "classifier_input", "logits")


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "sequence" in payload.get("stages", {}):
        stages = payload["stages"]["sequence"]
        values = {stage: {"known_test_top1": stages[stage].get("known_test_top1"), "unknown_score_auroc": stages[stage].get("unknown_score_auroc")} for stage in STAGES if stage in stages}
        checkpoint = path.parent.name
        protocol = "80-percent known subjects + 20-percent unknown subjects; group-disjoint known test"
    else:
        stages = payload["stages"]
        values = {stage: {"known_test_top1": stages[stage].get("accuracy_top1"), "unknown_score_auroc": None} for stage in STAGES if stage in stages}
        checkpoint = path.parent.name
        protocol = "all subjects enrolled; group-disjoint closed-set test"
    return {"checkpoint": checkpoint, "dataset": payload.get("dataset", "ISRUC"), "sequence_count": payload.get("sequence_count"), "protocol": protocol, "stages": values, "source": str(path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [read(path.resolve()) for path in args.summaries]
    rows.sort(key=lambda row: row["checkpoint"])
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": "subject-id-continual-learning-checkpoint-tracking-v1",
        "rows": rows,
        "scientific_conclusion_allowed": False,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ISRUC continual-learning subject-ID checkpoint tracking (2026-09-16)",
        "",
        "The checkpoint source objective remains sleep-stage prediction; subject ID is an external audit target. The 2026-09-15 rows use the open-set robustness protocol, while the 2026-09-16 rows use the all-subject closed-set probe. They must not be compared as if they were the same split or decoder; use the within-protocol trajectory for checkpoint comparisons.",
        "",
    ]
    for protocol in sorted({row["protocol"] for row in rows}):
        lines += [f"### {protocol}", "", "| checkpoint | EEG branch | Transformer L1 | Transformer L2 | Transformer L3 | classifier input | logits |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in rows:
            if row["protocol"] != protocol:
                continue
            vals = [row["stages"].get(stage, {}).get("known_test_top1") for stage in STAGES]
            fmt = ["n/a" if value is None else f"{value:.4f}" for value in vals]
            lines.append("| " + row["checkpoint"] + " | " + " | ".join(fmt) + " |")
        lines.append("")
    lines += [
        "",
        "The table measures representation decodability, not biometric uniqueness. An increase or decrease across checkpoints is not by itself LoP; it must be paired with task accuracy, fresh-subject adaptation, retention, effective rank/spectrum, and gradient-coverage measurements.",
        "",
        "The ISRUC dataset currently exposes processed subject/group files but no explicit recording/session identifier in the canonical copy. Therefore this remains group-disjoint rather than session-disjoint; a session-level rerun is required before making biological invariance claims.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "checkpoints": [row["checkpoint"] for row in rows], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
