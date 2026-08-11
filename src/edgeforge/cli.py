"""EdgeForge command-line interface."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from edgeforge import __version__
from edgeforge.api import serve
from edgeforge.client import APIError, Client
from edgeforge.logging_utils import configure_logging
from edgeforge.worker import Worker, default_worker_id


def _add_client_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--control-url",
        default=os.environ.get("EDGEFORGE_CONTROL_URL", "http://127.0.0.1:8080"),
        help="control plane base URL",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("EDGEFORGE_TOKEN"),
        help="shared bearer token (or EDGEFORGE_TOKEN)",
    )


def _labels(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise ValueError(f"label must use key=value syntax: {value}")
        result[key] = item
    return result


def _client(args: argparse.Namespace) -> Client:
    if not args.token:
        raise ValueError("a token is required; pass --token or set EDGEFORGE_TOKEN")
    return Client(args.control_url, args.token)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgeforge", description="EdgeForge heterogeneous edge runtime")
    parser.add_argument("--log-level", default=os.environ.get("EDGEFORGE_LOG_LEVEL", "INFO"))
    parser.add_argument("--log-dir", default=os.environ.get("EDGEFORGE_LOG_DIR", "./logs"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    control = subparsers.add_parser("control", help="run the control plane")
    control.add_argument("--bind", default=os.environ.get("EDGEFORGE_BIND", "127.0.0.1"))
    control.add_argument("--port", type=int, default=int(os.environ.get("EDGEFORGE_PORT", "8080")))
    control.add_argument("--database", default=os.environ.get("EDGEFORGE_DATABASE", "./edgeforge.db"))
    control.add_argument("--artifact-dir", default=os.environ.get("EDGEFORGE_ARTIFACT_DIR", "./artifacts"))
    control.add_argument("--token", default=os.environ.get("EDGEFORGE_TOKEN"))
    control.add_argument("--worker-timeout", type=float, default=30.0)

    worker = subparsers.add_parser("worker", help="run a worker agent")
    _add_client_options(worker)
    worker.add_argument("--worker-id", default=os.environ.get("EDGEFORGE_WORKER_ID") or default_worker_id())
    worker.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")
    worker.add_argument("--work-root", default=os.environ.get("EDGEFORGE_WORK_ROOT", "./work"))
    worker.add_argument("--interval", type=float, default=2.0)
    worker.add_argument(
        "--allow-command",
        action="append",
        default=["uname", "python3", "true", "false"],
        help="allow an executable name or absolute path; repeat as needed",
    )
    worker.add_argument("--allow-any-command", action="store_true", help="allow any executable (trusted CI workers only)")

    workers = subparsers.add_parser("workers", help="list registered workers")
    _add_client_options(workers)

    tasks = subparsers.add_parser("tasks", help="list recent tasks")
    _add_client_options(tasks)
    tasks.add_argument("--limit", type=int, default=100)

    task = subparsers.add_parser("task", help="show one task")
    _add_client_options(task)
    task.add_argument("task_id")

    submit = subparsers.add_parser("submit", help="submit a command or benchmark task")
    _add_client_options(submit)
    submit.add_argument("--kind", choices=("command", "benchmark"), default="command")
    submit.add_argument("--arch", action="append", default=[])
    submit.add_argument("--accelerator", action="append", default=[])
    submit.add_argument("--prefer-accelerator", action="append", default=[])
    submit.add_argument("--worker-id", action="append", default=[])
    submit.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")
    submit.add_argument("--min-memory-mb", type=int, default=0)
    submit.add_argument("--priority", type=int, default=0)
    submit.add_argument("--max-attempts", type=int, default=1)
    submit.add_argument("--timeout", type=float, default=300.0)
    submit.add_argument("--repeats", type=int, default=5)
    submit.add_argument("--cwd")
    submit.add_argument("--wait", action="store_true")
    submit.add_argument("argv", nargs=argparse.REMAINDER, help="command after --")

    operator = subparsers.add_parser("operator-benchmark", help="run an Operator IR benchmark")
    _add_client_options(operator)
    operator.add_argument("--operator", required=True, choices=("matmul", "softmax", "rmsnorm", "silu"))
    operator.add_argument("--shape", required=True, help="comma-separated dimensions; matmul uses M,K,N")
    operator.add_argument("--dtype", default="fp32", choices=("fp32", "fp16", "bf16"))
    operator.add_argument("--backend", default="python-reference")
    operator.add_argument("--kernel-id")
    operator.add_argument("--arch", action="append", default=[])
    operator.add_argument("--worker-id", action="append", default=[])
    operator.add_argument("--accelerator", action="append", default=[])
    operator.add_argument("--repeats", type=int, default=3)
    operator.add_argument("--priority", type=int, default=0)
    operator.add_argument("--wait", action="store_true")

    events = subparsers.add_parser("events", help="list persisted versioned events")
    _add_client_options(events)
    events.add_argument("--version")
    events.add_argument("--limit", type=int, default=100)

    benchmarks = subparsers.add_parser("benchmarks", help="list persisted operator benchmark results")
    _add_client_options(benchmarks)
    benchmarks.add_argument("--operator")
    benchmarks.add_argument("--limit", type=int, default=100)

    releases = subparsers.add_parser("releases", help="list recorded releases")
    _add_client_options(releases)

    release = subparsers.add_parser("release", help="record a release in the control-plane ledger")
    _add_client_options(release)
    release.add_argument("version")
    release.add_argument("--summary", required=True)
    release.add_argument("--status", default="active", choices=("active", "deprecated", "retired"))
    release.add_argument("--metadata", default="{}", help="JSON object")

    kernels = subparsers.add_parser("kernels", help="list registered kernels")
    _add_client_options(kernels)
    kernels.add_argument("--operator")

    kernel = subparsers.add_parser("kernel-register", help="register a kernel candidate")
    _add_client_options(kernel)
    kernel.add_argument("--id")
    kernel.add_argument("--operator", required=True)
    kernel.add_argument("--backend", required=True)
    kernel.add_argument("--version", required=True)
    kernel.add_argument("--arch", action="append", default=[])
    kernel.add_argument("--accelerator", action="append", default=[])
    kernel.add_argument("--dtype", action="append", default=[])
    kernel.add_argument("--metadata", default="{}")
    kernel.add_argument("--compiler", default="{}")
    kernel.add_argument("--artifact-digest")

    regressions = subparsers.add_parser("regressions", help="find benchmark regressions")
    _add_client_options(regressions)
    regressions.add_argument("--operator")
    regressions.add_argument("--threshold", type=float, default=0.2)

    pipeline = subparsers.add_parser("kernel-pipeline", help="run compile, correctness and benchmark stages")
    _add_client_options(pipeline)
    pipeline.add_argument("--kernel-id", required=True)
    pipeline.add_argument("--operator", required=True, choices=("matmul", "softmax", "rmsnorm", "silu"))
    pipeline.add_argument("--shape", required=True)
    pipeline.add_argument("--dtype", default="fp32", choices=("fp32", "fp16", "bf16"))
    pipeline.add_argument("--worker-id", action="append", default=[])
    pipeline.add_argument("--arch", action="append", default=[])
    pipeline.add_argument("--repeats", type=int, default=5)
    pipeline.add_argument("--warmup", type=int, default=5)
    pipeline.add_argument("--wait", action="store_true")

    autotune = subparsers.add_parser("kernel-autotune", help="search and select the best kernel configuration")
    _add_client_options(autotune)
    autotune.add_argument("--kernel-id", required=True)
    autotune.add_argument("--operator", required=True, choices=("matmul",))
    autotune.add_argument("--shape", required=True)
    autotune.add_argument("--dtype", default="fp16", choices=("fp16", "bf16"))
    autotune.add_argument("--worker-id", action="append", default=[])
    autotune.add_argument("--arch", action="append", default=[])
    autotune.add_argument("--candidate", action="append", default=[], help="JSON tuning config; repeat as needed")
    autotune.add_argument("--repeats", type=int, default=5)
    autotune.add_argument("--warmup", type=int, default=3)
    autotune.add_argument("--wait", action="store_true")

    tuning_runs = subparsers.add_parser("tuning-runs", help="list persisted Auto Tuning runs")
    _add_client_options(tuning_runs)
    tuning_runs.add_argument("--operator")
    tuning_runs.add_argument("--limit", type=int, default=100)

    compiler_plan = subparsers.add_parser("compiler-plan", help="explain the best Kernel and Worker path")
    _add_client_options(compiler_plan)
    _add_compiler_scheduler_options(compiler_plan, include_execution=False)

    compiler_run = subparsers.add_parser("compiler-run", help="plan and execute on the selected Kernel and Worker")
    _add_client_options(compiler_run)
    _add_compiler_scheduler_options(compiler_run, include_execution=True)

    decisions = subparsers.add_parser("schedule-decisions", help="list persisted compiler scheduling decisions")
    _add_client_options(decisions)
    decisions.add_argument("--limit", type=int, default=100)

    artifacts = subparsers.add_parser("artifacts", help="list content-addressed artifacts")
    _add_client_options(artifacts)
    artifacts.add_argument("--kind")
    artifacts.add_argument("--limit", type=int, default=100)

    artifact_put = subparsers.add_parser("artifact-put", help="upload one small artifact")
    _add_client_options(artifact_put)
    artifact_put.add_argument("path")
    artifact_put.add_argument("--kind", default="generic")
    artifact_put.add_argument("--media-type", default="application/octet-stream")
    artifact_put.add_argument("--name")
    return parser


def _add_compiler_scheduler_options(parser: argparse.ArgumentParser, *, include_execution: bool) -> None:
    parser.add_argument("--operator", required=True, choices=("matmul", "softmax", "rmsnorm", "silu"))
    parser.add_argument("--shape", required=True)
    parser.add_argument("--dtype", default="fp32", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--worker-id", action="append", default=[])
    parser.add_argument("--arch", action="append", default=[])
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--kernel-id", action="append", default=[])
    parser.add_argument("--compile-weight", type=float, default=0.05)
    parser.add_argument("--load-weight-ms", type=float, default=1.0)
    parser.add_argument("--unknown-latency-ms", type=float, default=1000.0)
    if include_execution:
        parser.add_argument("--repeats", type=int, default=5)
        parser.add_argument("--warmup", type=int, default=5)
        parser.add_argument("--wait", action="store_true")


def _submit(args: argparse.Namespace) -> int:
    argv = args.argv[1:] if args.argv and args.argv[0] == "--" else args.argv
    if not argv:
        raise ValueError("submit requires a command, for example: submit -- uname -a")
    payload: dict[str, Any] = {"argv": argv, "timeout_seconds": args.timeout}
    if args.cwd:
        payload["cwd"] = args.cwd
    if args.kind == "benchmark":
        payload["repeats"] = args.repeats
    requirements = {
        "architectures": args.arch,
        "accelerators": args.accelerator,
        "prefer_accelerators": args.prefer_accelerator,
        "worker_ids": args.worker_id,
        "labels": _labels(args.label),
        "min_memory_mb": args.min_memory_mb,
    }
    client = _client(args)
    task = client.request(
        "POST",
        "/api/v1/tasks",
        {
            "kind": args.kind,
            "payload": payload,
            "requirements": requirements,
            "priority": args.priority,
            "max_attempts": args.max_attempts,
        },
    )
    _print_json(task)
    if not args.wait:
        return 0
    while task["status"] not in {"succeeded", "failed", "cancelled"}:
        time.sleep(1)
        task = client.request("GET", f"/api/v1/tasks/{task['id']}")
    _print_json(task)
    return 0 if task["status"] == "succeeded" else 1


def _operator_submit(args: argparse.Namespace) -> int:
    try:
        shape = [int(item.strip()) for item in args.shape.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--shape must be comma-separated integers") from error
    if not shape:
        raise ValueError("--shape must not be empty")
    task = _client(args).request(
        "POST",
        "/api/v1/tasks",
        {
            "kind": "operator_benchmark",
            "payload": {
                "operator": {
                    "name": args.operator,
                    "shape": shape,
                    "dtype": args.dtype,
                    "backend": args.backend,
                },
                "repeats": args.repeats,
            },
            "requirements": {
                "architectures": args.arch,
                "worker_ids": args.worker_id,
                "accelerators": args.accelerator,
                "kernel_id": args.kernel_id,
            },
            "priority": args.priority,
        },
    )
    _print_json(task)
    if not args.wait:
        return 0
    while task["status"] not in {"succeeded", "failed", "cancelled"}:
        time.sleep(1)
        task = _client(args).request("GET", f"/api/v1/tasks/{task['id']}")
    _print_json(task)
    return 0 if task["status"] == "succeeded" else 1


def _register_kernel(args: argparse.Namespace) -> int:
    try:
        metadata = json.loads(args.metadata)
        compiler = json.loads(args.compiler)
    except json.JSONDecodeError as error:
        raise ValueError("--metadata and --compiler must be valid JSON") from error
    if not isinstance(metadata, dict) or not isinstance(compiler, dict):
        raise ValueError("--metadata and --compiler must be JSON objects")
    _print_json(
        _client(args).request(
            "POST",
            "/api/v1/kernels",
            {
                "id": args.id,
                "operator": args.operator,
                "backend": args.backend,
                "version": args.version,
                "architectures": args.arch or ["*"],
                "accelerators": args.accelerator,
                "dtypes": args.dtype or ["fp32"],
                "metadata": metadata,
                "compiler": compiler,
                "artifact_digest": args.artifact_digest,
            },
        )
    )
    return 0


def _pipeline_submit(args: argparse.Namespace) -> int:
    try:
        shape = [int(item.strip()) for item in args.shape.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--shape must be comma-separated integers") from error
    task = _client(args).request(
        "POST",
        "/api/v1/tasks",
        {
            "kind": "kernel_pipeline",
            "payload": {
                "operator": {"name": args.operator, "shape": shape, "dtype": args.dtype},
                "kernel_id": args.kernel_id,
                "repeats": args.repeats,
                "warmup": args.warmup,
            },
            "requirements": {
                "kernel_id": args.kernel_id,
                "worker_ids": args.worker_id,
                "architectures": args.arch,
            },
        },
    )
    _print_json(task)
    if not args.wait:
        return 0
    while task["status"] not in {"succeeded", "failed", "cancelled"}:
        time.sleep(1)
        task = _client(args).request("GET", f"/api/v1/tasks/{task['id']}")
    _print_json(task)
    return 0 if task["status"] == "succeeded" else 1


def _autotune_submit(args: argparse.Namespace) -> int:
    try:
        shape = [int(item.strip()) for item in args.shape.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--shape must be comma-separated integers") from error
    candidates = []
    for value in args.candidate:
        try:
            candidate = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("--candidate must be a valid JSON object") from error
        if not isinstance(candidate, dict):
            raise ValueError("--candidate must be a JSON object")
        candidates.append(candidate)
    task = _client(args).request(
        "POST",
        "/api/v1/tasks",
        {
            "kind": "kernel_autotune",
            "payload": {
                "operator": {"name": args.operator, "shape": shape, "dtype": args.dtype},
                "kernel_id": args.kernel_id,
                "candidates": candidates,
                "repeats": args.repeats,
                "warmup": args.warmup,
            },
            "requirements": {
                "kernel_id": args.kernel_id,
                "worker_ids": args.worker_id,
                "architectures": args.arch,
            },
        },
    )
    _print_json(task)
    if not args.wait:
        return 0
    while task["status"] not in {"succeeded", "failed", "cancelled"}:
        time.sleep(1)
        task = _client(args).request("GET", f"/api/v1/tasks/{task['id']}")
    _print_json(task)
    return 0 if task["status"] == "succeeded" else 1


def _compiler_scheduler_payload(args: argparse.Namespace) -> dict[str, Any]:
    try:
        shape = [int(item.strip()) for item in args.shape.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--shape must be comma-separated integers") from error
    if not shape:
        raise ValueError("--shape must not be empty")
    return {
        "operator": {"name": args.operator, "shape": shape, "dtype": args.dtype},
        "requirements": {
            "worker_ids": args.worker_id,
            "architectures": args.arch,
            "backends": args.backend,
            "kernel_ids": args.kernel_id,
        },
        "policy": {
            "compile_weight": args.compile_weight,
            "load_weight_ms": args.load_weight_ms,
            "unknown_latency_ms": args.unknown_latency_ms,
        },
    }


def _compiler_run(args: argparse.Namespace) -> int:
    request = _compiler_scheduler_payload(args)
    task = _client(args).request(
        "POST",
        "/api/v1/tasks",
        {
            "kind": "compiler_run",
            "payload": {
                "operator": request["operator"],
                "policy": request["policy"],
                "repeats": args.repeats,
                "warmup": args.warmup,
            },
            "requirements": request["requirements"],
        },
    )
    _print_json(task)
    if not args.wait:
        return 0
    while task["status"] not in {"succeeded", "failed", "cancelled"}:
        time.sleep(1)
        task = _client(args).request("GET", f"/api/v1/tasks/{task['id']}")
    _print_json(task)
    return 0 if task["status"] == "succeeded" else 1


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    component = "control" if args.command == "control" else "worker" if args.command == "worker" else "cli"
    configure_logging(component, __version__, args.log_dir, args.log_level)
    try:
        if args.command == "control":
            if not args.token:
                raise ValueError("control plane requires --token or EDGEFORGE_TOKEN")
            serve(args.bind, args.port, args.database, args.token, args.worker_timeout, args.artifact_dir)
            return
        if args.command == "worker":
            worker = Worker(
                client=_client(args),
                worker_id=args.worker_id,
                labels=_labels(args.label),
                work_root=Path(args.work_root),
                interval=max(0.2, args.interval),
                allowed_commands=set(args.allow_command),
                allow_any_command=args.allow_any_command,
            )
            worker.run()
            return
        if args.command == "workers":
            _print_json(_client(args).request("GET", "/api/v1/workers"))
            return
        if args.command == "tasks":
            _print_json(_client(args).request("GET", f"/api/v1/tasks?limit={args.limit}"))
            return
        if args.command == "task":
            _print_json(_client(args).request("GET", f"/api/v1/tasks/{args.task_id}"))
            return
        if args.command == "submit":
            raise SystemExit(_submit(args))
        if args.command == "operator-benchmark":
            raise SystemExit(_operator_submit(args))
        if args.command == "events":
            query = f"?limit={args.limit}"
            if args.version:
                query += f"&version={args.version}"
            _print_json(_client(args).request("GET", f"/api/v1/events{query}"))
            return
        if args.command == "benchmarks":
            query = f"?limit={args.limit}"
            if args.operator:
                query += f"&operator={args.operator}"
            _print_json(_client(args).request("GET", f"/api/v1/benchmarks{query}"))
            return
        if args.command == "releases":
            _print_json(_client(args).request("GET", "/api/v1/releases"))
            return
        if args.command == "release":
            try:
                metadata = json.loads(args.metadata)
            except json.JSONDecodeError as error:
                raise ValueError("--metadata must be valid JSON") from error
            if not isinstance(metadata, dict):
                raise ValueError("--metadata must be a JSON object")
            _print_json(
                _client(args).request(
                    "POST",
                    "/api/v1/releases",
                    {"version": args.version, "summary": args.summary, "status": args.status, "metadata": metadata},
                )
            )
            return
        if args.command == "kernels":
            query = f"?operator={args.operator}" if args.operator else ""
            _print_json(_client(args).request("GET", f"/api/v1/kernels{query}"))
            return
        if args.command == "kernel-register":
            raise SystemExit(_register_kernel(args))
        if args.command == "regressions":
            query = f"?threshold={args.threshold}"
            if args.operator:
                query += f"&operator={args.operator}"
            _print_json(_client(args).request("GET", f"/api/v1/regressions{query}"))
            return
        if args.command == "kernel-pipeline":
            raise SystemExit(_pipeline_submit(args))
        if args.command == "kernel-autotune":
            raise SystemExit(_autotune_submit(args))
        if args.command == "tuning-runs":
            query = f"?limit={args.limit}"
            if args.operator:
                query += f"&operator={args.operator}"
            _print_json(_client(args).request("GET", f"/api/v1/tuning-runs{query}"))
            return
        if args.command == "compiler-plan":
            _print_json(_client(args).request("POST", "/api/v1/plans", _compiler_scheduler_payload(args)))
            return
        if args.command == "compiler-run":
            raise SystemExit(_compiler_run(args))
        if args.command == "schedule-decisions":
            _print_json(_client(args).request("GET", f"/api/v1/schedule-decisions?limit={args.limit}"))
            return
        if args.command == "artifacts":
            query = f"?limit={args.limit}"
            if args.kind:
                query += f"&kind={args.kind}"
            _print_json(_client(args).request("GET", f"/api/v1/artifacts{query}"))
            return
        if args.command == "artifact-put":
            path = Path(args.path)
            content = path.read_bytes()
            _print_json(
                _client(args).request(
                    "POST",
                    "/api/v1/artifacts",
                    {
                        "name": args.name or path.name,
                        "kind": args.kind,
                        "media_type": args.media_type,
                        "content_base64": base64.b64encode(content).decode("ascii"),
                    },
                )
            )
            return
    except (APIError, OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
