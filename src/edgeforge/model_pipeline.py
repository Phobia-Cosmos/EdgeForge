"""Model-level frontend, transform, compiler and runtime pipeline.

The module deliberately treats frameworks as external tools.  This keeps the
control plane dependency-free while giving PyTorch, ONNX Runtime, Triton and
IREE adapters one stable contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from edgeforge.backend_registry import validate_backend_target


SCHEMA_VERSION = 1
STAGES = ("export", "transform", "compile", "run", "correctness", "benchmark")


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _command(value: Any, name: str, *, required: bool = False) -> list[str] | None:
    if value is None and not required:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must be a non-empty argv string array")
    return list(value)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def normalize_model_manifest(payload: Any) -> dict[str, Any]:
    value = _object(payload, "model pipeline payload")
    model = _object(value.get("model"), "model")
    dataset = _object(value.get("dataset"), "dataset")
    if not str(model.get("name") or "").strip():
        raise ValueError("model.name is required")
    if not str(dataset.get("name") or "").strip():
        raise ValueError("dataset.name is required")
    for field in ("checkpoint", "format"):
        if field in model and not isinstance(model[field], str):
            raise ValueError(f"model.{field} must be a string")
    if "manifest_digest" in dataset and not isinstance(dataset["manifest_digest"], str):
        raise ValueError("dataset.manifest_digest must be a string")
    if "transform_command" in dataset:
        dataset["transform_command"] = _command(dataset.get("transform_command"), "dataset.transform_command")
    transforms = value.get("transforms") or []
    if not isinstance(transforms, list) or len(transforms) > 128:
        raise ValueError("transforms must be an array with at most 128 entries")
    clean_transforms = []
    for index, transform in enumerate(transforms):
        item = _object(transform, f"transforms[{index}]")
        if not str(item.get("name") or "").strip() or not str(item.get("version") or "").strip():
            raise ValueError(f"transforms[{index}] requires name and version")
        config = item.get("config") or {}
        if not isinstance(config, dict):
            raise ValueError(f"transforms[{index}].config must be an object")
        clean_transforms.append({"name": str(item["name"]), "version": str(item["version"]), "config": config})

    frontend = _object(value.get("frontend") or {}, "frontend")
    compiler = _object(value.get("compiler") or {}, "compiler")
    runtime = _object(value.get("runtime") or {}, "runtime")
    correctness = _object(value.get("correctness") or {}, "correctness")
    benchmark = _object(value.get("benchmark") or {}, "benchmark")
    for section, obj in (("frontend", frontend), ("compiler", compiler), ("runtime", runtime), ("correctness", correctness), ("benchmark", benchmark)):
        if "command" in obj:
            obj["command"] = _command(obj.get("command"), f"{section}.command")
    frontend.setdefault("name", "external")
    compiler.setdefault("backend", "python-reference")
    runtime.setdefault("name", compiler.get("backend", "python-reference"))
    repeats = int(benchmark.get("repeats", 1) or 1)
    if repeats < 1 or repeats > 100:
        raise ValueError("benchmark.repeats must be between 1 and 100")
    benchmark["repeats"] = repeats
    target = _object(value.get("target") or {}, "target")
    if "architecture" in target and not isinstance(target["architecture"], str):
        raise ValueError("target.architecture must be a string")
    normalized = {
        "schema_version": int(value.get("schema_version", SCHEMA_VERSION)),
        "model": model,
        "dataset": dataset,
        "transforms": clean_transforms,
        "frontend": frontend,
        "compiler": compiler,
        "runtime": runtime,
        "correctness": correctness,
        "benchmark": benchmark,
        "target": target,
        "metadata": _object(value.get("metadata") or {}, "metadata"),
    }
    if normalized["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported model pipeline schema_version: {normalized['schema_version']}")
    normalized["backend_spec"] = validate_backend_target(str(compiler.get("backend") or ""), target)
    normalized["transform_digest"] = _digest(clean_transforms)
    normalized["manifest_digest"] = _digest({k: normalized[k] for k in normalized if k not in {"manifest_digest"}})
    return normalized


def _parse_json_output(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    try:
        value = json.loads(text.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {}
    return value if isinstance(value, dict) else {}


def run_model_pipeline(
    payload: dict[str, Any],
    *,
    work_root: Path,
    run_command: Callable[[list[str], str | None, dict[str, str], float], dict[str, Any]],
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = normalize_model_manifest(payload)
    root = work_root.resolve()
    started = time.perf_counter()
    stages: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    for stage in STAGES:
        section = manifest["frontend"] if stage == "export" else {"command": manifest["dataset"].get("transform_command")} if stage == "transform" else manifest["compiler"] if stage == "compile" else manifest["runtime"] if stage == "run" else manifest["correctness"] if stage == "correctness" else manifest["benchmark"]
        argv = section.get("command")
        stage_started = time.perf_counter()
        if argv is None:
            stage_result = {"stage": stage, "status": "skipped", "exit_code": 0, "elapsed_ms": 0.0}
        else:
            env = {
                "EDGEFORGE_MODEL_MANIFEST_DIGEST": manifest["manifest_digest"],
                "EDGEFORGE_TRANSFORM_DIGEST": manifest["transform_digest"],
                "EDGEFORGE_MODEL_NAME": str(manifest["model"]["name"]),
                "EDGEFORGE_DATASET_NAME": str(manifest["dataset"]["name"]),
                "EDGEFORGE_MODEL_BACKEND": str(manifest["compiler"].get("backend") or "python-reference"),
                "EDGEFORGE_TARGET_ARCHITECTURE": str(manifest["target"].get("architecture") or ""),
                "EDGEFORGE_TARGET_DEVICE": str(manifest["target"].get("device") or ""),
                "EDGEFORGE_STAGE": stage,
            }
            if stage == "benchmark":
                env["EDGEFORGE_BENCHMARK_REPEATS"] = str(manifest["benchmark"].get("repeats", 1))
            result_path = section.get("result_path")
            if result_path:
                if not isinstance(result_path, str):
                    raise ValueError(f"{stage}.result_path must be a string")
                candidate = (root / result_path).resolve()
                if not candidate.is_relative_to(root):
                    raise ValueError(f"{stage}.result_path escapes work root")
                env["EDGEFORGE_RESULT_PATH"] = str(candidate)
            execution = run_command(list(argv), section.get("cwd"), env, float(section.get("timeout_seconds", 86_400.0)))
            stage_result = {"stage": stage, **execution}
            parsed = _parse_json_output(str(execution.get("stdout", "")))
            if parsed:
                stage_result["parsed"] = parsed
                outputs[stage] = parsed
                if stage == "correctness" and "correctness" in parsed and not bool(parsed["correctness"]):
                    stage_result["exit_code"] = 1
                    stage_result["validation_error"] = "correctness reported false"
        stage_result["elapsed_ms"] = round((time.perf_counter() - stage_started) * 1000.0, 3)
        stages.append(stage_result)
        if stage_result.get("exit_code", 1) != 0:
            break

    benchmark = outputs.get("benchmark") or {}
    correctness = outputs.get("correctness") or {}
    compile_output = outputs.get("compile") or {}
    compile_stage_ms = next((item.get("elapsed_ms") for item in stages if item["stage"] == "compile"), None)
    result = {
        "exit_code": 0 if stages and all(item.get("exit_code", 1) == 0 for item in stages) else 1,
        "manifest": manifest,
        "pipeline": stages,
        "correctness": bool(correctness.get("correctness", correctness.get("passed", False))) if correctness else None,
        "benchmark": benchmark,
        "compile_ms": compile_output.get("compile_ms", compile_stage_ms),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "environment": environment or {},
    }
    artifact = json.dumps({"schema_version": 1, "manifest": manifest, "result": result}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result["artifact_upload"] = {
        "content_base64": __import__("base64").b64encode(artifact).decode("ascii"),
        "kind": "model-compiler-manifest",
        "media_type": "application/json",
        "name": f"{manifest['model']['name']}-{manifest['manifest_digest'][:12]}.json",
        "metadata": {"manifest_digest": manifest["manifest_digest"], "transform_digest": manifest["transform_digest"], "backend": manifest["compiler"].get("backend")},
    }
    return result
