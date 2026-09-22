"""Build a safe import catalog from existing BrainUICL/RA-EEG results."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


RESULT_NAMES = {"metrics.json", "RESULTS.json"}
MAX_RESULT_BYTES = 8 * 1024 * 1024


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return value[:100] or "result"


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _method(relative: str, raw: dict[str, Any] | None) -> str:
    if isinstance(raw, dict) and isinstance(raw.get("method"), str) and raw["method"].strip():
        return _slug(raw["method"].lower())
    if isinstance(raw, dict) and isinstance(raw.get("config"), dict) and isinstance(raw["config"].get("method"), str):
        return _slug(raw["config"]["method"].lower())
    lowered = relative.lower()
    tokens = (
        ("puridiver", "puridiver_eeg"),
        ("spr", "spr_eeg"),
        ("online_ewc", "online_ewc"),
        ("ewc", "ewc"),
        ("finetune", "finetune"),
        ("plain_er", "plain_er"),
        ("brainuicl", "brainuicl"),
    )
    for needle, result in tokens:
        if needle in lowered:
            return result
    return "historical_import"


def _dataset(relative: str) -> str:
    lowered = relative.lower()
    if "faced" in lowered:
        return "FACED"
    if "isruc" in lowered or "rttdp" in lowered or "regularization_cl_eeg" in lowered or "edgeforge_runs" in lowered:
        return "ISRUC-Group-I"
    return "BrainUICL-local"


def _protocol_and_group(relative: str) -> tuple[str, str]:
    lowered = relative.lower()
    if "aligned" in lowered or "regularization_cl_eeg_runs" in lowered:
        return "eeg-cl-v1-aligned", "aligned-full49"
    if "rttdp_brainuicl_runs" in lowered and any(token in lowered for token in ("spr", "puridiver")):
        return "method-transfer-v1", "method-transfer"
    if "unlabeled" in lowered or "proxy" in lowered or "attack" in lowered or "defense" in lowered:
        return "raeeg-historical-transfer-v1", "method-transfer"
    return "raeeg-historical-import-v1", "historical-unclassified"


def _load_small_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > MAX_RESULT_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def build_entry(root: Path, path: Path, *, dataset_manifest_digest: str | None = None) -> dict[str, Any]:
    root = root.resolve()
    path = path.resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("result path must be a file below catalog root")
    relative = path.relative_to(root).as_posix()
    raw = _load_small_json(path)
    method = _method(relative, raw)
    protocol, comparison_group = _protocol_and_group(relative)
    digest = _digest(path)
    dataset_name = _dataset(relative)
    model_name = "BrainUICL"
    if method == "spr_eeg":
        model_name = "SPR-EEG"
    elif method == "puridiver_eeg":
        model_name = "PuriDivER-EEG"
    identity = f"{relative}:{digest[:16]}"
    return {
        "experiment_id": f"migrated-{_slug(relative.rsplit('/', 1)[-2] if '/' in relative else relative)}-{digest[:12]}",
        "workload": "raeeg",
        "dataset": {
            "name": dataset_name,
            "root": str(root),
            "result_root": str(root),
            "manifest_digest": dataset_manifest_digest or "unresolved-local-dataset-manifest",
        },
        "model": {"name": model_name, "source": "BrainUICL-local", "result_digest": digest},
        "protocol": protocol,
        "method": method,
        "seed": _seed(relative),
        "runner": {"mode": "import", "result_path": relative, "adapter": "raeeg-metrics-v1"},
        "metadata": {
            "comparison_group": comparison_group,
            "migration": "brainuicl-results-v1",
            "source_path": relative,
            "source_digest": digest,
            "source_size_bytes": path.stat().st_size,
            "scientific_conclusion_allowed": False,
            "note": "Historical import; rerun and protocol review required before capability gate.",
        },
    }


def _seed(relative: str) -> int:
    match = re.search(r"seed(\d+)", relative.lower())
    return int(match.group(1)) if match else 0


def discover_results(root: Path, *, pattern: str | None = None, limit: int | None = None) -> list[Path]:
    root = root.resolve()
    candidates = [path for path in root.rglob("*") if path.is_file() and path.name in RESULT_NAMES]
    if pattern:
        candidates = [path for path in candidates if pattern.lower() in path.relative_to(root).as_posix().lower()]
    candidates.sort(key=lambda path: path.relative_to(root).as_posix())
    if limit is not None:
        candidates = candidates[: max(0, int(limit))]
    return candidates


def build_catalog(root: Path, paths: Iterable[Path], *, dataset_manifest_digest: str | None = None) -> dict[str, Any]:
    root = root.resolve()
    entries = [build_entry(root, path, dataset_manifest_digest=dataset_manifest_digest) for path in paths]
    return {
        "schema_version": 1,
        "worker_work_root": str(root),
        "source": {"name": "BrainUICL-local", "root": str(root), "result_count": len(entries)},
        "defaults": {},
        "experiments": entries,
    }
