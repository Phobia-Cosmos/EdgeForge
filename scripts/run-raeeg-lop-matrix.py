#!/usr/bin/env python3
"""Run a reproducible FACED/ISRUC LoP evidence matrix.

The runner is intentionally an EdgeForge-side orchestration layer.  It only
reads BrainUICL source/checkpoints/data and writes into an explicit run
directory.  A matrix cell is one ``dataset × method × condition × subject ×
checkpoint-stage`` observation.  Every successful cell contains the raw
instrumentation/probe JSON plus an ``edgeforge-bundle-v1`` result that can be
imported by the normal experiment API.  Failed cells are retained in the
catalog manifest and event log rather than silently dropped.

Manifest format (schema ``raeeg-lop-matrix-v1``)::

    {
      "schema_version": 1,
      "matrix_id": "isruc-lop-v014",
      "version": "0.14.0",
      "brainuicl_root": "/path/to/BrainUICL",
      "output_root": "/path/to/run-directory",
      "datasets": [{
        "name": "ISRUC",
        "data_root": "/path/to/isruc",
        "subjects": [1],
        "stages": [0, 10, 25, 49],
        "conditions": [{"name": "clean"},
                       {"name": "noise-s0p5", "data_root": "/path/to/shift"}],
        "methods": [{
          "name": "finetune",
          "checkpoint_root_template": "/runs/{method}/checkpoints/individual_{stage}",
          "baseline_checkpoint_root": "/model_parameter/ISRUC/Pretrain",
          "fresh_checkpoint_root": "/model_parameter/ISRUC/Pretrain"
        }]
      }],
      "defaults": {"seed": 4321, "device": "cuda:0", "max_files": 1,
                   "max_batches": 1, "importance_batches": 1,
                   "probe_steps": "0,5,10",
                   "retention": {"data_root": "/path/to/data", "subjects": [1]}}
    }

For larger experiments, ``cells`` may be supplied instead of ``datasets``;
each cell then contains the expanded fields shown in the generated
``matrix-plan.json``.  Unknown fields are preserved in cell metadata but are
not interpolated into subprocess arguments.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
EXPECTED_CHECKPOINT_FILES = (
    "feature_extractor_parameter_{seed}.pkl",
    "feature_encoder_parameter_{seed}.pkl",
    "sleep_classifier_parameter_{seed}.pkl",
)


class MatrixError(ValueError):
    """An invalid matrix manifest or unsafe output layout."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: Any, *, limit: int = 64) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() or char in "._-" else "-" for char in text]
    result = "".join(chars).strip("-") or "item"
    return result[:limit]


def _write_json(path: Path, payload: Any, *, allow_replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if path.exists() and not allow_replace:
        temporary.unlink(missing_ok=True)
        raise MatrixError(f"refusing to overwrite existing result: {path}")
    temporary.replace(path)


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp": _utc(), **event}, ensure_ascii=False, sort_keys=True) + "\n")


def _path(value: Any, *, variables: dict[str, Any] | None = None) -> Path:
    if value is None or str(value).strip() == "":
        raise MatrixError("path value must not be empty")
    text = os.path.expandvars(os.path.expanduser(str(value)))
    if variables:
        class _Values(dict[str, Any]):
            def __missing__(self, key: str) -> str:
                raise MatrixError(f"unknown path template variable: {{{key}}}")

        try:
            text = text.format_map(_Values(variables))
        except KeyError as error:  # pragma: no cover - defensive for custom mappings
            raise MatrixError(f"unknown path template variable: {error}") from error
    return Path(text).resolve(strict=False)


def _subject_int(value: Any) -> int:
    text = str(value).strip()
    if text.lower().startswith("sub-"):
        text = text[4:]
    try:
        return int(text)
    except (TypeError, ValueError) as error:
        raise MatrixError(f"subject must be an integer or sub-NNN, got {value!r}") from error


def _ensure_list(value: Any, field: str, *, default: list[Any] | None = None) -> list[Any]:
    if value is None and default is not None:
        return list(default)
    if not isinstance(value, list) or not value:
        raise MatrixError(f"{field} must be a non-empty array")
    return list(value)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate_manifest(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise MatrixError("matrix manifest must be a JSON object")
    if int(raw.get("schema_version") or 0) != SCHEMA_VERSION:
        raise MatrixError(f"unsupported matrix schema_version; expected {SCHEMA_VERSION}")
    matrix_id = str(raw.get("matrix_id") or "")
    version = str(raw.get("version") or "")
    if not matrix_id or not version:
        raise MatrixError("matrix_id and version are required")
    if not isinstance(raw.get("brainuicl_root"), str) or not raw["brainuicl_root"].strip():
        raise MatrixError("brainuicl_root is required")
    if not isinstance(raw.get("output_root"), str) or not raw["output_root"].strip():
        raise MatrixError("output_root is required")
    if not raw.get("cells") and not raw.get("datasets"):
        raise MatrixError("manifest requires either cells[] or datasets[]")
    if raw.get("cells") is not None and (not isinstance(raw["cells"], list) or not raw["cells"]):
        raise MatrixError("cells must be a non-empty array when supplied")
    if raw.get("datasets") is not None and (not isinstance(raw["datasets"], list) or not raw["datasets"]):
        raise MatrixError("datasets must be a non-empty array when supplied")
    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise MatrixError("defaults must be an object")
    result = copy.deepcopy(raw)
    result["defaults"] = defaults
    return result


def _condition_rows(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = dataset.get("conditions")
    if conditions is None:
        return [{"name": "clean", "data_root": dataset.get("data_root")}]
    if not isinstance(conditions, list) or not conditions:
        raise MatrixError(f"dataset {dataset.get('name')} conditions must be a non-empty array")
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        if isinstance(condition, str):
            condition = {"name": condition}
        if not isinstance(condition, dict) or not str(condition.get("name") or "").strip():
            raise MatrixError("each condition requires a non-empty name")
        rows.append(dict(condition))
    return rows


def _method_rows(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    methods = dataset.get("methods")
    if not isinstance(methods, list) or not methods:
        raise MatrixError(f"dataset {dataset.get('name')} methods must be a non-empty array")
    rows: list[dict[str, Any]] = []
    for method in methods:
        if isinstance(method, str):
            method = {"name": method}
        if not isinstance(method, dict) or not str(method.get("name") or "").strip():
            raise MatrixError("each method requires a non-empty name")
        rows.append(dict(method))
    return rows


def _resolve_checkpoint_root(
    dataset: dict[str, Any], method: dict[str, Any], stage: int, variables: dict[str, Any]
) -> Path:
    roots = method.get("checkpoint_roots", dataset.get("checkpoint_roots"))
    value: Any = None
    if isinstance(roots, dict):
        value = roots.get(str(stage), roots.get(stage))
    if value is None:
        value = method.get("checkpoint_root_template", dataset.get("checkpoint_root_template"))
    if value is None:
        value = method.get("checkpoint_root", dataset.get("checkpoint_root"))
    if value is None and stage == 0:
        value = method.get("pretrain_checkpoint_root", dataset.get("pretrain_checkpoint_root"))
    if value is None:
        raise MatrixError(f"no checkpoint root configured for stage {stage} / method {method.get('name')}")
    return _path(value, variables=variables)


def _resolve_optional_root(value: Any, variables: dict[str, Any]) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return _path(value, variables=variables)


def expand_cells(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a compact manifest into explicit, deterministic cell specs."""

    defaults = dict(manifest.get("defaults") or {})
    if isinstance(manifest.get("cells"), list):
        rows: list[dict[str, Any]] = []
        for raw in manifest["cells"]:
            if not isinstance(raw, dict):
                raise MatrixError("each explicit cell must be an object")
            rows.append(_merge(defaults, raw))
        return rows

    rows = []
    for dataset in manifest["datasets"]:
        if not isinstance(dataset, dict) or not str(dataset.get("name") or "").strip():
            raise MatrixError("each dataset requires a non-empty name")
        dataset = _merge(defaults, dataset)
        name = str(dataset["name"])
        subjects = _ensure_list(dataset.get("subjects"), f"dataset {name}.subjects")
        configured_seeds = dataset.get("seeds")
        if configured_seeds is None:
            configured_seeds = [dataset.get("seed", defaults.get("seed", 4321))]
        if not isinstance(configured_seeds, list) or not configured_seeds:
            raise MatrixError(f"dataset {name}.seeds must be a non-empty array")
        try:
            seeds = sorted({int(item) for item in configured_seeds})
        except (TypeError, ValueError) as error:
            raise MatrixError(f"dataset {name}.seeds must contain integers") from error
        stages = _ensure_list(dataset.get("stages"), f"dataset {name}.stages")
        try:
            stages = sorted({int(item) for item in stages})
        except (TypeError, ValueError) as error:
            raise MatrixError(f"dataset {name}.stages must contain integers") from error
        if stages[0] < 0:
            raise MatrixError("checkpoint stages must be non-negative")
        for method in _method_rows(dataset):
            for condition in _condition_rows(dataset):
                for subject in subjects:
                    for seed in seeds:
                        for stage in stages:
                            cell = _merge(dataset, method)
                            cell["dataset"] = name
                            cell["method"] = str(method["name"])
                            cell["condition"] = dict(condition)
                            cell["subject"] = subject
                            cell["seed"] = seed
                            cell["checkpoint_stage"] = stage
                            # Preserve the dataset-level defaults while allowing a
                            # method-level override for the fresh/reference model.
                            cell["baseline_checkpoint_root"] = method.get(
                                "baseline_checkpoint_root",
                                dataset.get("baseline_checkpoint_root", dataset.get("pretrain_checkpoint_root")),
                            )
                            cell["fresh_checkpoint_root"] = method.get(
                                "fresh_checkpoint_root", dataset.get("fresh_checkpoint_root")
                            )
                            rows.append(cell)
    if not rows:
        raise MatrixError("manifest expansion produced no cells")
    return rows


def _input_roots(cell: dict[str, Any], variables: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    for key in ("brainuicl_root", "data_root", "checkpoint_root", "baseline_checkpoint_root", "fresh_checkpoint_root"):
        value = cell.get(key)
        if value is None:
            continue
        try:
            root = _path(value, variables=variables)
        except MatrixError:
            continue
        roots.append(root)
    condition = cell.get("condition")
    if isinstance(condition, dict) and condition.get("data_root"):
        roots.append(_path(condition["data_root"], variables=variables))
    retention = cell.get("retention")
    if isinstance(retention, dict) and retention.get("data_root"):
        roots.append(_path(retention["data_root"], variables=variables))
    return roots


def _assert_disjoint(output_root: Path, roots: Iterable[Path]) -> None:
    output_root = output_root.resolve(strict=False)
    for root in roots:
        root = root.resolve(strict=False)
        if output_root == root or output_root.is_relative_to(root) or root.is_relative_to(output_root):
            raise MatrixError(f"output_root must be disjoint from input root: {output_root} vs {root}")


def _checkpoint_files(root: Path, seed: int) -> list[Path]:
    return [root / name.format(seed=seed) for name in EXPECTED_CHECKPOINT_FILES]


def _check_cell_inputs(cell: dict[str, Any], variables: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    brainuicl_root = _path(cell["brainuicl_root"], variables=variables)
    if not brainuicl_root.is_dir():
        errors.append(f"brainuicl_root missing: {brainuicl_root}")
    data_root = _path(cell["data_root"], variables=variables)
    if not data_root.is_dir():
        errors.append(f"data_root missing: {data_root}")
    checkpoint_root = _path(cell["checkpoint_root"], variables=variables)
    if not checkpoint_root.is_dir():
        errors.append(f"checkpoint_root missing: {checkpoint_root}")
    seed = int(cell["seed"])
    for path in _checkpoint_files(checkpoint_root, seed):
        if not path.is_file():
            errors.append(f"checkpoint file missing: {path}")
    for key in ("baseline_checkpoint_root", "fresh_checkpoint_root"):
        if cell.get(key):
            root = _path(cell[key], variables=variables)
            if not root.is_dir():
                errors.append(f"{key} missing: {root}")
            if key in {"baseline_checkpoint_root", "fresh_checkpoint_root"}:
                for path in _checkpoint_files(root, seed):
                    if not path.is_file():
                        errors.append(f"{key} file missing: {path}")
    condition = cell.get("condition")
    if isinstance(condition, dict) and condition.get("manifest"):
        manifest = _path(condition["manifest"], variables=variables)
        if not manifest.is_file():
            errors.append(f"condition manifest missing: {manifest}")
    retention = cell.get("retention")
    if retention is not None:
        if not isinstance(retention, dict):
            errors.append("retention must be an object")
        else:
            retention_root_value = retention.get("data_root")
            if not retention_root_value:
                errors.append("retention.data_root is required")
            else:
                retention_root = _path(retention_root_value, variables=variables)
                if not retention_root.is_dir():
                    errors.append(f"retention data_root missing: {retention_root}")
                subjects = retention.get("subjects")
                if not isinstance(subjects, list) or not subjects:
                    errors.append("retention.subjects must be a non-empty array")
                else:
                    limit = int(retention.get("max_files") or 0)
                    for raw_subject in subjects:
                        try:
                            subject = _subject_int(raw_subject)
                        except MatrixError as error:
                            errors.append(str(error))
                            continue
                        subject_root = retention_root / str(subject)
                        if not subject_root.is_dir():
                            subject_root = retention_root / f"sub-{subject:03d}"
                        data_dir = subject_root / "data"
                        label_dir = subject_root / "label"
                        files = sorted(data_dir.glob("*.npy")) if data_dir.is_dir() else []
                        if limit > 0:
                            files = files[:limit]
                        if not files:
                            errors.append(f"retention subject data missing: {data_dir}")
                        for data_path in files:
                            if not (label_dir / data_path.name).is_file():
                                errors.append(f"retention label missing: {label_dir / data_path.name}")
    return errors


def _cell_variables(manifest: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    condition = cell.get("condition") if isinstance(cell.get("condition"), dict) else {}
    subject = _subject_int(cell.get("subject"))
    return {
        "matrix_id": manifest.get("matrix_id"),
        "version": manifest.get("version"),
        "dataset": cell.get("dataset"),
        "method": cell.get("method"),
        "condition": condition.get("name", cell.get("condition", "clean")),
        "subject": subject,
        "subject_label": str(cell.get("subject")),
        "seed": int(cell.get("seed")),
        "stage": int(cell.get("checkpoint_stage")),
    }


def normalize_cells(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve compact cells and validate their required runtime fields."""

    result: list[dict[str, Any]] = []
    for raw in expand_cells(manifest):
        cell = copy.deepcopy(raw)
        if isinstance(cell.get("condition"), str):
            cell["condition"] = {"name": cell["condition"]}
        condition = cell.get("condition") or {"name": "clean"}
        if not isinstance(condition, dict) or not str(condition.get("name") or "").strip():
            raise MatrixError("cell condition must have a name")
        cell["condition"] = condition
        cell["dataset"] = str(cell.get("dataset") or cell.get("dataset_name") or "")
        cell["method"] = str(cell.get("method") or "")
        if not cell["dataset"] or not cell["method"]:
            raise MatrixError("each cell requires dataset and method")
        cell["subject"] = _subject_int(cell.get("subject"))
        try:
            cell["checkpoint_stage"] = int(cell.get("checkpoint_stage", cell.get("stage", 0)))
            cell["seed"] = int(cell.get("seed", 4321))
        except (TypeError, ValueError) as error:
            raise MatrixError("cell seed and checkpoint_stage must be integers") from error
        if cell["checkpoint_stage"] < 0:
            raise MatrixError("checkpoint_stage must be non-negative")
        condition_root = condition.get("data_root")
        if condition_root:
            cell["data_root"] = condition_root
        if not cell.get("data_root"):
            raise MatrixError(f"cell {cell['dataset']}/{cell['method']} has no data_root")
        variables = _cell_variables(manifest, cell)
        # A compact method configuration can defer checkpoint path expansion
        # until after stage/method variables are known.
        if not cell.get("checkpoint_root"):
            dataset_stub = {"checkpoint_root": cell.get("checkpoint_root")}
            method_stub = cell
            cell["checkpoint_root"] = str(_resolve_checkpoint_root(dataset_stub, method_stub, cell["checkpoint_stage"], variables))
        if cell.get("baseline_checkpoint_root"):
            cell["baseline_checkpoint_root"] = str(_path(cell["baseline_checkpoint_root"], variables=variables))
        if cell.get("fresh_checkpoint_root"):
            cell["fresh_checkpoint_root"] = str(_path(cell["fresh_checkpoint_root"], variables=variables))
        retention = cell.get("retention")
        if retention is not None:
            if not isinstance(retention, dict):
                raise MatrixError("cell retention must be an object")
            retention = copy.deepcopy(retention)
            if not retention.get("data_root"):
                raise MatrixError("retention.data_root is required")
            retention["data_root"] = str(_path(retention["data_root"], variables=variables))
            subjects = retention.get("subjects")
            if not isinstance(subjects, list) or not subjects:
                raise MatrixError("retention.subjects must be a non-empty array")
            try:
                retention["subjects"] = sorted({_subject_int(item) for item in subjects})
            except MatrixError:
                raise
            for key in ("max_files", "batch_size"):
                if key in retention and retention[key] is not None:
                    try:
                        retention[key] = int(retention[key])
                    except (TypeError, ValueError) as error:
                        raise MatrixError(f"retention.{key} must be an integer") from error
                    if retention[key] < 0:
                        raise MatrixError(f"retention.{key} cannot be negative")
            cell["retention"] = retention
        cell["data_root"] = str(_path(cell["data_root"], variables=variables))
        cell["checkpoint_root"] = str(_path(cell["checkpoint_root"], variables=variables))
        split = cell.get("split") or manifest.get("split") or "subject-eval"
        cell["split"] = f"{split}:{condition['name']}" if condition["name"] != "clean" else str(split)
        result.append(cell)
    return result


def _cell_id(manifest: dict[str, Any], cell: dict[str, Any]) -> str:
    identity = {
        "matrix_id": manifest.get("matrix_id"),
        "dataset": cell["dataset"],
        "method": cell["method"],
        "condition": cell["condition"].get("name"),
        "subject": cell["subject"],
        "seed": cell["seed"],
        "checkpoint_stage": cell["checkpoint_stage"],
    }
    return "-".join(
        (
            _slug(manifest["matrix_id"]),
            _slug(cell["dataset"]),
            _slug(cell["method"]),
            _slug(cell["condition"].get("name")),
            f"subject{cell['subject']}",
            f"seed{cell['seed']}",
            f"stage{cell['checkpoint_stage']}",
            _digest_bytes(_canonical(identity))[:10],
        )
    )


def _command_common(
    script: Path,
    python: str,
    repo: Path,
    cell: dict[str, Any],
    *,
    output: Path,
    include_retention: bool = False,
) -> list[str]:
    args = [python, str(script), "--brainuicl-root", str(_path(cell["brainuicl_root"])), "--dataset", str(cell["dataset"]), "--checkpoint-root", str(_path(cell["checkpoint_root"])), "--data-root", str(_path(cell["data_root"])), "--subject", str(cell["subject"]), "--seed", str(cell["seed"]), "--checkpoint-stage", str(cell["checkpoint_stage"]), "--method", str(cell["method"]), "--split", str(cell["split"]), "--device", str(cell.get("device", "cuda:0")), "--max-files", str(int(cell.get("max_files", 1))), "--batch-size", str(int(cell.get("batch_size", 1))), "--output", str(output)]
    retention = cell.get("retention")
    if include_retention and isinstance(retention, dict):
        args.extend(["--retention-data-root", str(_path(retention["data_root"]))])
        for subject in retention.get("subjects") or []:
            args.extend(["--retention-subject", str(int(subject))])
        if "max_files" in retention:
            args.extend(["--retention-max-files", str(int(retention["max_files"]))])
        if "batch_size" in retention:
            args.extend(["--retention-batch-size", str(int(retention["batch_size"]))])
    return args


def build_commands(repo: Path, python: str, cell: dict[str, Any], cell_dir: Path, *, skip_instrumentation: bool = False, skip_probe: bool = False, with_unlabeled_diagnostics: bool = False) -> list[tuple[str, list[str], Path]]:
    commands: list[tuple[str, list[str], Path]] = []
    if not skip_instrumentation:
        output = cell_dir / "instrumentation.json"
        script = repo / "scripts" / "brainuicl-instrumentation.py"
        argv = _command_common(script, python, repo, cell, output=output)
        for name in ("max_batches", "importance_batches", "importance_top_k", "linearity_epsilon"):
            if name in cell:
                argv.extend([f"--{name.replace('_', '-')}", str(cell[name])])
        if int(cell["checkpoint_stage"]) > 0 and cell.get("baseline_checkpoint_root"):
            argv.extend(["--baseline-checkpoint-root", str(_path(cell["baseline_checkpoint_root"]))])
        commands.append(("instrumentation", argv, output))
    if not skip_probe:
        output = cell_dir / "probe.json"
        script = repo / "scripts" / "brainuicl-fixed-budget-probe.py"
        argv = _command_common(script, python, repo, cell, output=output, include_retention=True)
        for name in ("train_fraction", "probe_steps", "lr", "weight_decay"):
            if name in cell:
                argv.extend([f"--{name.replace('_', '-')}", str(cell[name])])
        if cell.get("fresh_checkpoint_root"):
            argv.extend(["--fresh-checkpoint-root", str(_path(cell["fresh_checkpoint_root"]))])
        commands.append(("probe", argv, output))
    if with_unlabeled_diagnostics:
        output = cell_dir / "unlabeled.json"
        script = repo / "scripts" / "brainuicl-unlabeled-diagnostics.py"
        argv = _command_common(script, python, repo, cell, output=output)
        # The label-free adapter has the same data/checkpoint identity but a
        # separate perturbation protocol and no probe optimizer settings.
        if "max_batches" in cell:
            argv.extend(["--max-batches", str(cell["max_batches"])])
        argv.extend(["--noise-severity", str(cell.get("unlabeled_noise_severity", 0.05))])
        commands.append(("unlabeled", argv, output))
    return commands


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MatrixError(f"cannot read JSON result {path}: {error}") from error
    if not isinstance(value, dict):
        raise MatrixError(f"result must be an object: {path}")
    return value


def _merge_bundle(manifest: dict[str, Any], cell: dict[str, Any], cell_id: str, cell_dir: Path, raw_results: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    condition = cell["condition"]
    condition_name = str(condition.get("name"))
    metrics: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, str]] = set()
    source_results: list[dict[str, Any]] = []
    for role, raw in raw_results:
        source_name = {"predictor": "instrumentation.json", "outcome": "probe.json", "diagnostic": "unlabeled.json"}.get(role)
        if source_name is None:
            continue
        source = cell_dir / source_name
        source_results.append({"role": role, "path": str(source.relative_to(cell_dir.parent.parent)), "sha256": _digest_file(source), "size_bytes": source.stat().st_size})
        for item in raw.get("metrics") or []:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            row = copy.deepcopy(item)
            metric_role = role
            # Retention/forgetting is deliberately a separate evidence role:
            # it describes old-task stability and must never be selected as
            # the fresh-vs-checkpoint plasticity outcome by an audit.
            if role == "outcome" and (
                str(row["name"]).startswith("task.forgetting.")
                or str(row["name"]).startswith("task.probe.retention.")
            ):
                metric_role = "retention"
            elif role == "outcome" and (
                str(row["name"]).startswith("task.optimizer.")
                or str(row["name"]).startswith("task.probe.optimizer.")
            ):
                metric_role = "diagnostic"
            context = dict(row.get("context") or {})
            context.update({"condition": condition_name, "metric_role": metric_role, "matrix_id": manifest["matrix_id"]})
            row["context"] = context
            key = (str(row["name"]), row.get("step"), json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            if key not in seen:
                seen.add(key)
                metrics.append(row)
    comparison_group = f"{cell['dataset']}:{condition_name}"
    relative_bundle = str((cell_dir / "bundle.json").relative_to(cell_dir.parent.parent))
    probe_steps = cell.get("probe_steps", "0,5,10")
    try:
        probe_budget = max(int(item.strip()) for item in str(probe_steps).split(",") if item.strip())
    except (TypeError, ValueError):
        probe_budget = None
    task_order = []
    for item in cell.get("stages", []):
        try:
            task_order.append(int(item))
        except (TypeError, ValueError):
            continue
    spec = {
        "schema_version": 1,
        "experiment_id": cell_id,
        "workload": "raeeg",
        "dataset": {"name": cell["dataset"], "root": cell["data_root"], "condition": condition_name},
        "model": {"name": "BrainUICL", "source": str(_path(cell["brainuicl_root"]))},
        "protocol": "eeg-lop-matrix-v1",
        "method": cell["method"],
        "seed": int(cell["seed"]),
        "runner": {"mode": "import", "result_path": relative_bundle, "adapter": "edgeforge-bundle-v1"},
        "metadata": {
            "comparison_group": comparison_group,
            "matrix_id": manifest["matrix_id"],
            "condition": condition_name,
            "subject": str(cell["subject"]),
            "checkpoint_stage": int(cell["checkpoint_stage"]),
            "split": cell["split"],
            "task_order": task_order,
            "probe_budget": probe_budget,
            "optimizer": {"name": "Adam", "lr": cell.get("lr"), "weight_decay": cell.get("weight_decay", 0.0), "scope": "target-train-only", "state_persisted": False},
            "lr": cell.get("lr"),
            "model_structure": {"name": "BrainUICL"},
            "fresh_warm_protocol": "supervised-oracle-fixed-budget-heldout-v1",
            "retention": copy.deepcopy(cell.get("retention")),
            "scientific_conclusion_allowed": False,
            "note": "matrix cell; instrumentation/probe evidence is descriptive and single-cell results are not a LoP conclusion",
        },
    }
    return {
        "schema_version": 1,
        "instrumentation": "edgeforge-bundle-v1",
        "status": "succeeded",
        "experiment_id": cell_id,
        "spec": spec,
        "metrics": metrics,
        "summary": {
            "matrix_id": manifest["matrix_id"],
            "dataset": cell["dataset"],
            "method": cell["method"],
            "condition": condition_name,
            "subject": cell["subject"],
            "seed": cell["seed"],
            "checkpoint_stage": cell["checkpoint_stage"],
            "metric_count": len(metrics),
            "scientific_conclusion_allowed": False,
        },
        "source_results": source_results,
        "environment": {"python": platform.python_version(), "edgeforge_root": str(repo_root())},
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_run_dir(run_dir: Path, manifest_digest: str, *, resume: bool) -> None:
    marker = run_dir / "run-manifest.json"
    if not run_dir.exists():
        return
    if not marker.is_file():
        if any(run_dir.iterdir()):
            raise MatrixError(f"output_root exists without run marker; refusing to mix results: {run_dir}")
        return
    previous = _load_json(marker)
    if previous.get("manifest_digest") != manifest_digest:
        raise MatrixError("output_root belongs to a different manifest; choose a new versioned directory")
    if not resume:
        raise MatrixError("output_root already exists; pass --resume to continue it")


def _run_one(
    manifest: dict[str, Any], cell: dict[str, Any], *, run_dir: Path, python: str, event_log: Path,
    timeout: int, resume: bool, dry_run: bool, skip_instrumentation: bool, skip_probe: bool,
    stop_on_error: bool, with_unlabeled_diagnostics: bool,
) -> dict[str, Any]:
    cell_id = _cell_id(manifest, cell)
    cell_dir = run_dir / "cells" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    variables = _cell_variables(manifest, cell)
    input_errors = _check_cell_inputs(cell, variables)
    status_path = cell_dir / "status.json"
    config_digest = _digest_bytes(_canonical({
        "cell": {key: value for key, value in cell.items() if key not in {"_runtime"}},
        "skip_instrumentation": bool(skip_instrumentation),
        "skip_probe": bool(skip_probe),
        "with_unlabeled_diagnostics": bool(with_unlabeled_diagnostics),
    }))
    if resume and status_path.is_file():
        try:
            previous = _load_json(status_path)
        except MatrixError:
            previous = {}
        if previous.get("status") == "succeeded" and previous.get("config_digest") == config_digest and (cell_dir / "bundle.json").is_file():
            event = {"event": "cell.skip", "cell_id": cell_id, "status": "succeeded", "reason": "resume"}
            _append_event(event_log, event)
            return {**previous, "cell_id": cell_id, "status": "skipped"}
    if input_errors:
        result = {"cell_id": cell_id, "status": "blocked", "config_digest": config_digest, "errors": input_errors, "cell": cell}
        _write_json(status_path, result)
        _append_event(event_log, {"event": "cell.blocked", "cell_id": cell_id, "errors": input_errors})
        return result
    commands = build_commands(repo_root(), python, cell, cell_dir, skip_instrumentation=skip_instrumentation, skip_probe=skip_probe, with_unlabeled_diagnostics=with_unlabeled_diagnostics)
    if not commands:
        raise MatrixError("at least one of instrumentation/probe must be enabled")
    command_rows: list[dict[str, Any]] = []
    raw_results: list[tuple[str, dict[str, Any]]] = []
    started = time.monotonic()
    for role, argv, expected in commands:
        command_path = cell_dir / f"{role}.command.json"
        stdout_path = cell_dir / f"{role}.stdout.log"
        stderr_path = cell_dir / f"{role}.stderr.log"
        _write_json(command_path, {"role": role, "argv": argv, "cwd": str(repo_root()), "started_at": _utc()})
        _append_event(event_log, {"event": "command.start", "cell_id": cell_id, "role": role, "argv": argv})
        code = -1
        error: str | None = None
        command_started = time.monotonic()
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(repo_root() / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(argv, cwd=repo_root(), env=env, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
                code = int(completed.returncode)
        except subprocess.TimeoutExpired:
            error = f"timeout after {timeout}s"
        except OSError as exc:
            error = str(exc)
        row = {"role": role, "argv": argv, "returncode": code, "error": error, "duration_s": time.monotonic() - command_started, "output": str(expected.relative_to(run_dir))}
        command_rows.append(row)
        _append_event(event_log, {"event": "command.finish", "cell_id": cell_id, **row})
        if code != 0 or error:
            result = {"cell_id": cell_id, "status": "failed", "config_digest": config_digest, "duration_s": time.monotonic() - started, "commands": command_rows, "cell": cell}
            _write_json(status_path, result)
            if stop_on_error:
                result["stop_requested"] = True
            return result
        try:
            result_role = {"instrumentation": "predictor", "probe": "outcome", "unlabeled": "diagnostic"}[role]
            raw_results.append((result_role, _load_json(expected)))
        except MatrixError as exc:
            result = {"cell_id": cell_id, "status": "failed", "config_digest": config_digest, "duration_s": time.monotonic() - started, "commands": command_rows, "errors": [str(exc)], "cell": cell}
            _write_json(status_path, result)
            return result
    bundle = _merge_bundle(manifest, cell, cell_id, cell_dir, raw_results)
    _write_json(cell_dir / "bundle.json", bundle)
    result = {"cell_id": cell_id, "status": "succeeded", "config_digest": config_digest, "duration_s": time.monotonic() - started, "commands": command_rows, "bundle": str((cell_dir / "bundle.json").relative_to(run_dir)), "metric_count": len(bundle["metrics"]), "cell": cell}
    _write_json(status_path, result)
    _append_event(event_log, {"event": "cell.finish", "cell_id": cell_id, "status": "succeeded", "metric_count": len(bundle["metrics"]), "duration_s": result["duration_s"]})
    return result


def _catalog(run_dir: Path, manifest: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    experiments: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") not in {"succeeded", "skipped"}:
            continue
        bundle_path = run_dir / str(result.get("bundle"))
        if not bundle_path.is_file():
            continue
        bundle = _load_json(bundle_path)
        spec = copy.deepcopy(bundle.get("spec") or {})
        if spec:
            experiments.append(spec)
    return {
        "schema_version": 1,
        "worker_work_root": str(run_dir),
        "source": {"name": "EdgeForge-RAEEG-LoP-matrix", "matrix_id": manifest["matrix_id"], "result_count": len(experiments)},
        "defaults": {},
        "experiments": experiments,
    }


def _all_cell_results(run_dir: Path, current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the complete persisted cell set, preserving partial-run state.

    ``--max-cells`` is useful for local smoke tests and recovery.  Rebuilding
    a catalog from only the cells selected in the latest invocation would
    silently erase previously successful stages, so persisted status files are
    merged with the current results before catalog/audit generation.
    """

    by_id: dict[str, dict[str, Any]] = {}
    for item in current:
        if item.get("cell_id") is not None:
            by_id[str(item["cell_id"])] = item
    cells_root = run_dir / "cells"
    if cells_root.is_dir():
        for status_path in sorted(cells_root.glob("*/status.json")):
            try:
                item = _load_json(status_path)
            except MatrixError:
                continue
            if item.get("cell_id") is None:
                item["cell_id"] = status_path.parent.name
            by_id.setdefault(str(item["cell_id"]), item)
    return [by_id[key] for key in sorted(by_id)]


def _trajectory_catalog(run_dir: Path, manifest: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one multi-stage experiment per subject/seed trajectory.

    Cell bundles are intentionally one checkpoint observation each so they
    can be resumed and imported independently.  ``lop_analysis`` expects an
    experiment to contain the complete ordered stage trajectory, however; if
    every stage is represented as a separate experiment, the audit correctly
    sees duplicate seeds and cannot form a lagged pair.  This derived catalog
    keeps the cell artifacts unchanged and supplies the analysis-shaped view.
    """

    groups: dict[tuple[str, str, str, str, str, int, int], list[tuple[dict[str, Any], dict[str, Any], Path]]] = {}
    for result in results:
        if result.get("status") not in {"succeeded", "skipped"}:
            continue
        bundle_path = run_dir / str(result.get("bundle"))
        if not bundle_path.is_file():
            continue
        bundle = _load_json(bundle_path)
        spec = bundle.get("spec") or {}
        dataset = str(spec.get("dataset", {}).get("name") or "")
        condition = str(spec.get("dataset", {}).get("condition") or "clean")
        method = str(spec.get("method") or "")
        seed = int(spec.get("seed"))
        metadata = spec.get("metadata") or {}
        subject = int(str(metadata.get("subject") or bundle.get("summary", {}).get("subject") or 0))
        stage = int(metadata.get("checkpoint_stage", bundle.get("summary", {}).get("checkpoint_stage", 0)))
        # A trajectory may only combine stages with the same split and
        # retention design.  Including this digest prevents a later stage
        # measured on a different old-task set from silently being treated as
        # the same longitudinal experiment.
        trajectory_context = _digest_bytes(_canonical({
            "split": metadata.get("split"),
            "retention": metadata.get("retention"),
        }))[:12]
        key = (dataset, method, condition, str(spec.get("protocol") or "eeg-lop-matrix-v1"), trajectory_context, subject, seed)
        groups.setdefault(key, []).append((spec, bundle, bundle_path))

    experiments: list[dict[str, Any]] = []
    trajectory_root = run_dir / "trajectories"
    for key, items in sorted(groups.items()):
        dataset, method, condition, protocol, trajectory_context, subject, seed = key
        items.sort(key=lambda item: int((item[0].get("metadata") or {}).get("checkpoint_stage", item[1].get("summary", {}).get("checkpoint_stage", 0))))
        stages = [int((spec.get("metadata") or {}).get("checkpoint_stage", bundle.get("summary", {}).get("checkpoint_stage", 0))) for spec, bundle, _path_value in items]
        trajectory_id = "-".join((_slug(manifest["matrix_id"]), _slug(dataset), _slug(method), _slug(condition), f"subject{subject}", f"seed{seed}", f"ctx{trajectory_context}", _digest_bytes(_canonical(stages))[:10]))
        trajectory_dir = trajectory_root / trajectory_id
        trajectory_path = trajectory_dir / "bundle.json"
        metrics: list[dict[str, Any]] = []
        seen: set[tuple[str, int | None, str]] = set()
        source_results: list[dict[str, Any]] = []
        for spec, bundle, bundle_path in items:
            source_results.append({"path": str(bundle_path.relative_to(run_dir)), "sha256": _digest_file(bundle_path), "stages": [int((spec.get("metadata") or {}).get("checkpoint_stage", 0))]})
            for item in bundle.get("metrics") or []:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    continue
                row = copy.deepcopy(item)
                context = dict(row.get("context") or {})
                context.setdefault("condition", condition)
                context.setdefault("matrix_id", manifest["matrix_id"])
                row["context"] = context
                identity = (str(row["name"]), row.get("step"), json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                if identity not in seen:
                    seen.add(identity)
                    metrics.append(row)
        base_spec = copy.deepcopy(items[0][0])
        trajectory_relative = str(trajectory_path.relative_to(run_dir))
        base_spec["experiment_id"] = trajectory_id
        base_spec["runner"] = {"mode": "import", "result_path": trajectory_relative, "adapter": "edgeforge-bundle-v1"}
        metadata = dict(base_spec.get("metadata") or {})
        metadata.update({"trajectory": True, "trajectory_stages": stages, "trajectory_cell_count": len(items), "scientific_conclusion_allowed": False})
        base_spec["metadata"] = metadata
        trajectory_bundle = {
            "schema_version": 1,
            "instrumentation": "edgeforge-trajectory-bundle-v1",
            "status": "succeeded",
            "experiment_id": trajectory_id,
            "spec": base_spec,
            "metrics": metrics,
            "summary": {"matrix_id": manifest["matrix_id"], "trajectory": True, "dataset": dataset, "method": method, "condition": condition, "subject": subject, "seed": seed, "stages": stages, "metric_count": len(metrics), "scientific_conclusion_allowed": False},
            "source_results": source_results,
            "environment": {"python": platform.python_version(), "edgeforge_root": str(repo_root())},
        }
        _write_json(trajectory_path, trajectory_bundle)
        experiments.append(base_spec)
    return {
        "schema_version": 1,
        "worker_work_root": str(run_dir),
        "source": {"name": "EdgeForge-RAEEG-LoP-trajectory", "matrix_id": manifest["matrix_id"], "result_count": len(experiments)},
        "defaults": {},
        "experiments": experiments,
    }


def _write_descriptive_report(run_dir: Path, trajectory_catalog_path: Path) -> dict[str, Any] | None:
    """Write the stable report sidecar without making it a gate."""

    script = repo_root() / "scripts" / "summarize-raeeg-lop-matrix.py"
    spec = importlib.util.spec_from_file_location("edgeforge_raeeg_matrix_report", script)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.build_report(trajectory_catalog_path)
    _write_json(run_dir / "matrix-report.json", report)
    (run_dir / "matrix-report.md").write_text(module.markdown(report), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, help="override manifest output_root")
    parser.add_argument("--python", dest="python_executable", help="Python used for adapter subprocesses (default: current interpreter)")
    parser.add_argument("--log-root", type=Path, default=None, help="versioned JSONL event log root (default: EDGEFORGE_LOG_DIR or logs)")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--skip-instrumentation", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--with-unlabeled-diagnostics", action="store_true", help="also run label-free confidence/consistency diagnostics per cell")
    parser.add_argument("--no-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest.resolve()
    raw_bytes = manifest_path.read_bytes()
    manifest = _validate_manifest(json.loads(raw_bytes.decode("utf-8")))
    manifest_digest = _digest_bytes(raw_bytes)
    repo = repo_root()
    manifest["brainuicl_root"] = str(_path(manifest["brainuicl_root"]))
    output_root = _path(args.output_root or manifest["output_root"])
    cells = normalize_cells(manifest)
    cell_ids = [_cell_id(manifest, cell) for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        duplicates = sorted({item for item in cell_ids if cell_ids.count(item) > 1})
        raise MatrixError(f"matrix expansion produced duplicate cell ids: {duplicates}")
    # Add common runtime fields after expansion; explicit cells may already
    # carry their own values.
    for cell in cells:
        cell.setdefault("brainuicl_root", manifest["brainuicl_root"])
        for key, value in (manifest.get("defaults") or {}).items():
            cell.setdefault(key, value)
        cell["brainuicl_root"] = str(_path(cell["brainuicl_root"]))
        _assert_disjoint(output_root, _input_roots(cell, _cell_variables(manifest, cell)))
    if args.timeout <= 0:
        raise MatrixError("timeout must be positive")
    python = str(args.python_executable or manifest.get("python") or sys.executable)
    run_dir = output_root
    _validate_run_dir(run_dir, manifest_digest, resume=args.resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_marker = run_dir / "run-manifest.json"
    if not run_marker.exists():
        _write_json(run_marker, {"schema_version": 1, "matrix_schema_version": SCHEMA_VERSION, "matrix_id": manifest["matrix_id"], "version": manifest["version"], "manifest": str(manifest_path), "manifest_digest": manifest_digest, "created_at": _utc(), "run_id": f"{_run_stamp()}-{os.getpid()}-{secrets.token_hex(4)}", "python": python, "cell_count": len(cells)}, allow_replace=False)
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "matrix-plan.json", {"schema_version": 1, "matrix_id": manifest["matrix_id"], "manifest_digest": manifest_digest, "cells": [{"cell_id": _cell_id(manifest, cell), **cell} for cell in cells]})
    log_root = _path(args.log_root or os.environ.get("EDGEFORGE_LOG_DIR", "logs"))
    event_log = log_root / f"v{manifest['version']}" / "raeeg-lop-matrix" / f"{_run_stamp()}-{os.getpid()}-{secrets.token_hex(4)}.jsonl"
    _append_event(event_log, {"event": "matrix.start", "matrix_id": manifest["matrix_id"], "version": manifest["version"], "manifest_digest": manifest_digest, "cell_count": len(cells), "output_root": str(run_dir), "dry_run": args.dry_run})
    if args.dry_run:
        dry_cells = []
        for cell in cells:
            cell_id = _cell_id(manifest, cell)
            errors = _check_cell_inputs(cell, _cell_variables(manifest, cell))
            dry_cells.append({
                "cell_id": cell_id,
                "status": "ready" if not errors else "blocked",
                "input_errors": errors,
                "commands": [argv for _role, argv, _out in build_commands(repo, python, cell, run_dir / "cells" / cell_id, skip_instrumentation=args.skip_instrumentation, skip_probe=args.skip_probe, with_unlabeled_diagnostics=args.with_unlabeled_diagnostics)],
            })
        payload = {"schema_version": 1, "matrix_id": manifest["matrix_id"], "version": manifest["version"], "cell_count": len(cells), "ready_count": sum(item["status"] == "ready" for item in dry_cells), "blocked_count": sum(item["status"] == "blocked" for item in dry_cells), "event_log": str(event_log), "cells": dry_cells}
        _write_json(run_dir / "dry-run.json", payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    selected = cells[: max(0, args.max_cells)] if args.max_cells is not None else cells
    results: list[dict[str, Any]] = []
    for index, cell in enumerate(selected, start=1):
        _append_event(event_log, {"event": "cell.queue", "index": index, "cell_id": _cell_id(manifest, cell)})
        result = _run_one(manifest, cell, run_dir=run_dir, python=python, event_log=event_log, timeout=args.timeout, resume=args.resume, dry_run=False, skip_instrumentation=args.skip_instrumentation, skip_probe=args.skip_probe, stop_on_error=args.stop_on_error, with_unlabeled_diagnostics=args.with_unlabeled_diagnostics)
        results.append(result)
        if args.stop_on_error and result.get("status") == "failed":
            break
    persisted_results = _all_cell_results(run_dir, results)
    catalog = _catalog(run_dir, manifest, persisted_results)
    _write_json(run_dir / "catalog.json", catalog)
    trajectory_catalog = _trajectory_catalog(run_dir, manifest, persisted_results)
    _write_json(run_dir / "trajectory-catalog.json", trajectory_catalog)
    report = _write_descriptive_report(run_dir, run_dir / "trajectory-catalog.json")
    audit: dict[str, Any] | None = None
    if not args.no_audit and trajectory_catalog["experiments"]:
        try:
            sys.path.insert(0, str(repo / "src"))
            from edgeforge.lop_audit import audit_catalog

            audit = audit_catalog(run_dir / "trajectory-catalog.json", context_policy=str(manifest.get("audit", {}).get("context_policy", "exact")), minimum_pairs=int(manifest.get("audit", {}).get("minimum_pairs", 3)), minimum_seeds=int(manifest.get("audit", {}).get("minimum_seeds", 3)))
            _write_json(run_dir / "audit.json", audit)
        except Exception as error:  # audit failure is evidence, not a lost run
            audit = {"status": "audit-error", "error": str(error), "scientific_conclusion_allowed": False}
            _write_json(run_dir / "audit.json", audit)
    counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status"))
        counts[status] = counts.get(status, 0) + 1
    summary = {"schema_version": 1, "matrix_id": manifest["matrix_id"], "version": manifest["version"], "manifest_digest": manifest_digest, "requested_cells": len(cells), "executed_cells": len(selected), "status_counts": dict(sorted(counts.items())), "catalog": str(run_dir / "catalog.json"), "trajectory_catalog": str(run_dir / "trajectory-catalog.json"), "report": str(run_dir / "matrix-report.json") if report is not None else None, "audit": str(run_dir / "audit.json") if audit is not None else None, "event_log": str(event_log), "scientific_conclusion_allowed": False}
    _write_json(run_dir / "matrix-summary.json", summary)
    _append_event(event_log, {"event": "matrix.finish", **summary})
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if counts.get("failed") or counts.get("blocked"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
