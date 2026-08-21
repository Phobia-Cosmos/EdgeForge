"""SQLite-backed worker registry and task state machine."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from edgeforge import __version__
from edgeforge.kernel import KernelSpec, kernel_supports_operator, kernel_supports_worker
from edgeforge.scheduler import build_execution_plan, select_worker


TERMINAL_TASK_STATES = {"succeeded", "failed", "cancelled"}


class Store:
    def __init__(self, path: str | Path, worker_timeout: float = 30.0, version: str = __version__):
        self.path = str(path)
        self.worker_timeout = worker_timeout
        self.version = version
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS workers (
                    id TEXT PRIMARY KEY,
                    hostname TEXT NOT NULL,
                    capabilities TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    labels TEXT NOT NULL,
                    version TEXT NOT NULL,
                    registered_at REAL NOT NULL,
                    last_seen REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    requirements TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    assigned_worker_id TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    version TEXT NOT NULL DEFAULT '0.1.0',
                    runtime_version TEXT,
                    result TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    entity_type TEXT,
                    entity_id TEXT,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS releases (
                    version TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kernels (
                    id TEXT PRIMARY KEY,
                    operator TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    version TEXT NOT NULL,
                    architectures TEXT NOT NULL,
                    accelerators TEXT NOT NULL DEFAULT '[]',
                    dtypes TEXT NOT NULL,
                    shape_constraints TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    compiler TEXT NOT NULL DEFAULT '{}',
                    artifact_digest TEXT,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    registered_by_version TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS kernels_operator_idx
                ON kernels(operator, status, created_at DESC);

                CREATE TABLE IF NOT EXISTS benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    runtime_version TEXT,
                    worker_id TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    hardware_fingerprint TEXT,
                    operator TEXT NOT NULL,
                    kernel_id TEXT,
                    kernel_version TEXT,
                    artifact_digest TEXT,
                    compile_ms REAL,
                    shape TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    correctness INTEGER NOT NULL,
                    timings TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    digest TEXT PRIMARY KEY,
                    algorithm TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    created_by_version TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS artifacts_kind_idx
                ON artifacts(kind, created_at DESC);

                CREATE TABLE IF NOT EXISTS tuning_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL UNIQUE,
                    version TEXT NOT NULL,
                    runtime_version TEXT,
                    worker_id TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    hardware_fingerprint TEXT,
                    operator TEXT NOT NULL,
                    kernel_id TEXT,
                    kernel_version TEXT,
                    shape TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    search_space TEXT NOT NULL,
                    candidates TEXT NOT NULL,
                    best_config TEXT,
                    artifact_digest TEXT,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS tuning_runs_lookup_idx
                ON tuning_runs(operator, architecture, created_at DESC);

                CREATE TABLE IF NOT EXISTS schedule_decisions (
                    task_id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    operator TEXT NOT NULL,
                    policy TEXT NOT NULL,
                    candidates TEXT NOT NULL,
                    selected TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS schedule_decisions_created_idx
                ON schedule_decisions(created_at DESC);

                CREATE TABLE IF NOT EXISTS experiment_runs (
                    task_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    runtime_version TEXT,
                    worker_id TEXT NOT NULL,
                    workload TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    model TEXT NOT NULL,
                    protocol TEXT NOT NULL,
                    method TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    spec TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    artifact_digest TEXT,
                    source_digest TEXT,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS experiment_runs_lookup_idx
                ON experiment_runs(workload, method, created_at DESC);

                CREATE INDEX IF NOT EXISTS experiment_runs_id_idx
                ON experiment_runs(experiment_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS experiment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    step INTEGER,
                    unit TEXT NOT NULL,
                    context TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS experiment_metrics_lookup_idx
                ON experiment_metrics(experiment_id, name, step, created_at DESC);

                CREATE TABLE IF NOT EXISTS model_runs (
                    task_id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    runtime_version TEXT,
                    worker_id TEXT NOT NULL,
                    task_status TEXT NOT NULL DEFAULT 'succeeded',
                    model_name TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    dataset_manifest_digest TEXT,
                    transform_digest TEXT NOT NULL,
                    frontend TEXT NOT NULL,
                    compiler_backend TEXT NOT NULL,
                    compiler_identity TEXT NOT NULL,
                    target_architecture TEXT,
                    correctness INTEGER,
                    compile_ms REAL,
                    first_call_ms REAL,
                    steady_latency_ms REAL,
                    peak_memory_mb REAL,
                    output_digest TEXT,
                    artifact_digest TEXT,
                    manifest TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS model_runs_lookup_idx
                ON model_runs(model_name, dataset_name, created_at DESC);

                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    workload TEXT NOT NULL,
                    protocol TEXT NOT NULL,
                    comparison_group TEXT NOT NULL,
                    source_experiment_id TEXT NOT NULL,
                    checkpoint_digest TEXT,
                    descriptor TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    registered_by_version TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS models_lookup_idx
                ON models(workload, protocol, comparison_group, status, created_at DESC);

                CREATE UNIQUE INDEX IF NOT EXISTS models_production_idx
                ON models(workload, protocol, comparison_group)
                WHERE status = 'production';

                CREATE TABLE IF NOT EXISTS gate_policies (
                    id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    workload TEXT NOT NULL,
                    protocol TEXT NOT NULL,
                    comparison_group TEXT NOT NULL,
                    rules TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    created_by_version TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS gate_policies_lookup_idx
                ON gate_policies(workload, protocol, comparison_group, created_at DESC);

                CREATE TABLE IF NOT EXISTS gate_evaluations (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    policy_snapshot TEXT NOT NULL,
                    metric_snapshot TEXT NOT NULL,
                    rule_results TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    created_by_version TEXT NOT NULL,
                    FOREIGN KEY(model_id) REFERENCES models(id),
                    FOREIGN KEY(policy_id) REFERENCES gate_policies(id)
                );

                CREATE INDEX IF NOT EXISTS gate_evaluations_lookup_idx
                ON gate_evaluations(model_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS tasks_queue_idx
                ON tasks(status, priority DESC, created_at ASC);

                CREATE INDEX IF NOT EXISTS events_version_idx
                ON events(version, created_at DESC);

                CREATE INDEX IF NOT EXISTS benchmarks_lookup_idx
                ON benchmarks(operator, architecture, dtype, created_at DESC);
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(tasks)").fetchall()}
            if "version" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN version TEXT NOT NULL DEFAULT '0.1.0'")
            if "runtime_version" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN runtime_version TEXT")
            model_run_columns = {row["name"] for row in connection.execute("PRAGMA table_info(model_runs)").fetchall()}
            if "task_status" not in model_run_columns:
                connection.execute("ALTER TABLE model_runs ADD COLUMN task_status TEXT NOT NULL DEFAULT 'succeeded'")
            benchmark_columns = {row["name"] for row in connection.execute("PRAGMA table_info(benchmarks)").fetchall()}
            if "kernel_id" not in benchmark_columns:
                connection.execute("ALTER TABLE benchmarks ADD COLUMN kernel_id TEXT")
            if "kernel_version" not in benchmark_columns:
                connection.execute("ALTER TABLE benchmarks ADD COLUMN kernel_version TEXT")
            if "hardware_fingerprint" not in benchmark_columns:
                connection.execute("ALTER TABLE benchmarks ADD COLUMN hardware_fingerprint TEXT")
            if "artifact_digest" not in benchmark_columns:
                connection.execute("ALTER TABLE benchmarks ADD COLUMN artifact_digest TEXT")
            if "compile_ms" not in benchmark_columns:
                connection.execute("ALTER TABLE benchmarks ADD COLUMN compile_ms REAL")
            kernel_columns = {row["name"] for row in connection.execute("PRAGMA table_info(kernels)").fetchall()}
            if "compiler" not in kernel_columns:
                connection.execute("ALTER TABLE kernels ADD COLUMN compiler TEXT NOT NULL DEFAULT '{}'")
            if "artifact_digest" not in kernel_columns:
                connection.execute("ALTER TABLE kernels ADD COLUMN artifact_digest TEXT")
            if "accelerators" not in kernel_columns:
                connection.execute("ALTER TABLE kernels ADD COLUMN accelerators TEXT NOT NULL DEFAULT '[]'")
            connection.execute(
                "INSERT OR IGNORE INTO releases(version, created_at, status, summary, metadata) VALUES (?, ?, 'active', ?, '{}')",
                (self.version, time.time(), f"EdgeForge runtime {self.version}"),
            )

    @staticmethod
    def _decode_worker(row: sqlite3.Row, now: float, active_tasks: int = 0) -> dict[str, Any]:
        worker = dict(row)
        worker["capabilities"] = json.loads(worker["capabilities"])
        worker["metrics"] = json.loads(worker["metrics"])
        worker["labels"] = json.loads(worker["labels"])
        worker["status"] = "online" if now - worker["last_seen"] <= worker.pop("worker_timeout", float("inf")) else "offline"
        worker["active_tasks"] = active_tasks
        return worker

    @staticmethod
    def _decode_task(row: sqlite3.Row) -> dict[str, Any]:
        task = dict(row)
        task["payload"] = json.loads(task["payload"])
        task["requirements"] = json.loads(task["requirements"])
        task["result"] = json.loads(task["result"]) if task["result"] else None
        return task

    def register_worker(self, data: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        worker_id = str(data["id"])
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workers(id, hostname, capabilities, metrics, labels, version, registered_at, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    hostname = excluded.hostname,
                    capabilities = excluded.capabilities,
                    metrics = excluded.metrics,
                    labels = excluded.labels,
                    version = excluded.version,
                    last_seen = excluded.last_seen
                """,
                (
                    worker_id,
                    data.get("hostname") or worker_id,
                    json.dumps(data.get("capabilities") or {}, separators=(",", ":")),
                    json.dumps(data.get("metrics") or {}, separators=(",", ":")),
                    json.dumps(data.get("labels") or {}, separators=(",", ":")),
                    data.get("version") or "unknown",
                    now,
                    now,
                ),
            )
        worker = self.get_worker(worker_id)
        self.append_event("worker.registered", "control", "worker", worker_id, {"worker": worker})
        return worker

    def heartbeat(self, worker_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE workers SET metrics = ?, last_seen = ? WHERE id = ?",
                (json.dumps(metrics, separators=(",", ":")), now, worker_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(worker_id)
        return self.get_worker(worker_id)

    def get_worker(self, worker_id: str) -> dict[str, Any]:
        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT w.*,
                       ? AS worker_timeout,
                       (SELECT COUNT(*) FROM tasks t
                        WHERE t.assigned_worker_id = w.id AND t.status = 'running') AS active_tasks
                FROM workers w WHERE w.id = ?
                """,
                (self.worker_timeout, worker_id),
            ).fetchone()
        if row is None:
            raise KeyError(worker_id)
        return self._decode_worker(row, now, int(row["active_tasks"]))

    def list_workers(self) -> list[dict[str, Any]]:
        now = time.time()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT w.*,
                       ? AS worker_timeout,
                       (SELECT COUNT(*) FROM tasks t
                        WHERE t.assigned_worker_id = w.id AND t.status = 'running') AS active_tasks
                FROM workers w ORDER BY w.id
                """,
                (self.worker_timeout,),
            ).fetchall()
        return [self._decode_worker(row, now, int(row["active_tasks"])) for row in rows]

    def create_task(self, data: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        task_id = uuid.uuid4().hex
        kind = data.get("kind") or "command"
        if kind not in {"command", "benchmark", "operator_benchmark", "kernel_pipeline", "kernel_autotune", "compiler_run", "experiment_run", "model_pipeline"}:
            raise ValueError(f"unsupported task kind: {kind}")
        payload = data.get("payload") or {}
        requirements = dict(data.get("requirements") or {})
        schedule_decision = None
        if kind == "compiler_run":
            schedule_decision = self.plan_execution(
                payload.get("operator") or {},
                requirements,
                payload.get("policy") or {},
            )
            selected = schedule_decision["selected"]
            kernel = self.get_kernel(selected["kernel_id"])
            payload["operator"] = {**schedule_decision["operator"], "backend": kernel["backend"]}
            payload["kernel_id"] = kernel["id"]
            payload["kernel"] = kernel
            payload["schedule_decision"] = schedule_decision
            requirements["worker_ids"] = [selected["worker_id"]]
            requirements["kernel_id"] = kernel["id"]
        elif kind in {"operator_benchmark", "kernel_pipeline", "kernel_autotune"}:
            from edgeforge.operator import OperatorSpec

            OperatorSpec.from_payload(payload.get("operator") or {})
            requested_backend = str((payload.get("operator") or {}).get("backend") or "python-reference")
            if kind == "operator_benchmark" and requested_backend != "python-reference":
                raise ValueError(
                    "operator_benchmark only supports python-reference; use kernel_pipeline for a registered backend"
                )
            kernel_id = payload.get("kernel_id") or requirements.get("kernel_id")
            if kind == "kernel_pipeline" and not kernel_id:
                raise ValueError("kernel_pipeline requires kernel_id")
            if kind == "kernel_autotune" and not kernel_id:
                raise ValueError("kernel_autotune requires kernel_id")
            if kernel_id:
                kernel = self.get_kernel(str(kernel_id))
                if not kernel_supports_operator(kernel, payload.get("operator") or {}):
                    raise ValueError(f"kernel {kernel_id} does not support this operator or dtype")
                if kind == "operator_benchmark" and kernel.get("backend") != "python-reference":
                    raise ValueError(
                        "operator_benchmark only supports python-reference; use kernel_pipeline for a registered backend"
                    )
                if kind == "kernel_autotune":
                    from edgeforge.autotune import normalize_triton_matmul_candidates

                    if kernel.get("backend") != "triton" or payload["operator"].get("name") != "matmul":
                        raise ValueError("kernel_autotune currently requires a Triton MatMul kernel")
                    payload["candidates"] = normalize_triton_matmul_candidates(payload.get("candidates") or [])
                payload["operator"] = {**payload["operator"], "backend": kernel["backend"]}
                payload["kernel_id"] = str(kernel_id)
                payload["kernel"] = kernel
        elif kind == "experiment_run":
            from edgeforge.experiment import ExperimentSpec

            spec = ExperimentSpec.from_payload(payload.get("spec"))
            payload["spec"] = spec.to_dict()
        elif kind == "model_pipeline":
            from edgeforge.model_pipeline import normalize_model_manifest

            manifest = normalize_model_manifest(payload)
            payload = manifest
            target_arch = (manifest.get("target") or {}).get("architecture")
            if target_arch:
                requirements.setdefault("architectures", [target_arch])
            backend = str((manifest.get("compiler") or {}).get("backend") or "python-reference")
            requirements.setdefault("worker_backends", [backend])
        else:
            argv = payload.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
                raise ValueError("payload.argv must be a non-empty list of strings")
        max_attempts = min(10, max(1, int(data.get("max_attempts") or 1)))
        priority = min(100, max(-100, int(data.get("priority") or 0)))
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks(id, kind, payload, requirements, priority, status,
                                  attempts, max_attempts, created_at, version)
                VALUES (?, ?, ?, ?, ?, 'queued', 0, ?, ?, ?)
                """,
                (
                    task_id,
                    kind,
                    json.dumps(payload, separators=(",", ":")),
                    json.dumps(requirements, separators=(",", ":")),
                    priority,
                    max_attempts,
                    now,
                    self.version,
                ),
            )
        task = self.get_task(task_id)
        if schedule_decision:
            self._record_schedule_decision(task_id, schedule_decision)
        self.append_event("task.created", "control", "task", task_id, {"kind": kind, "requirements": requirements})
        return task

    def plan_execution(
        self,
        operator_payload: dict[str, Any],
        requirements: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from edgeforge.operator import OperatorSpec

        spec = OperatorSpec.from_payload(operator_payload)
        operator = {"name": spec.name, "shape": list(spec.shape), "dtype": spec.dtype, "attrs": spec.attrs}
        if requirements is not None and not isinstance(requirements, dict):
            raise ValueError("requirements must be an object")
        if policy is not None and not isinstance(policy, dict):
            raise ValueError("policy must be an object")
        return build_execution_plan(
            self.list_workers(),
            self.list_kernels(limit=1000),
            self.list_benchmarks(limit=1000),
            operator,
            requirements or {},
            policy or {},
        )

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return self._decode_task(row)

    def list_tasks(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = min(500, max(1, limit))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode_task(row) for row in rows]

    def _requeue_orphaned_tasks(self, connection: sqlite3.Connection, now: float) -> None:
        stale_before = now - self.worker_timeout
        orphaned = connection.execute(
            """
            SELECT t.id, t.attempts, t.max_attempts
            FROM tasks t LEFT JOIN workers w ON w.id = t.assigned_worker_id
            WHERE t.status = 'running' AND (w.id IS NULL OR w.last_seen < ?)
            """,
            (stale_before,),
        ).fetchall()
        for row in orphaned:
            if row["attempts"] < row["max_attempts"]:
                connection.execute(
                    """
                    UPDATE tasks SET status = 'queued', assigned_worker_id = NULL,
                                     started_at = NULL, error = 'worker lease expired'
                    WHERE id = ? AND status = 'running'
                    """,
                    (row["id"],),
                )
            else:
                connection.execute(
                    """
                    UPDATE tasks SET status = 'failed', finished_at = ?,
                                     error = 'worker lease expired'
                    WHERE id = ? AND status = 'running'
                    """,
                    (now, row["id"]),
                )

    def lease_task(self, worker_id: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            worker_row = connection.execute("SELECT id FROM workers WHERE id = ?", (worker_id,)).fetchone()
            if worker_row is None:
                raise KeyError(worker_id)
            self._requeue_orphaned_tasks(connection, now)

            active = connection.execute(
                "SELECT * FROM tasks WHERE assigned_worker_id = ? AND status = 'running' LIMIT 1",
                (worker_id,),
            ).fetchone()
            if active is not None:
                return None

            workers = self.list_workers()
            online_workers = [worker for worker in workers if worker["status"] == "online"]
            queued = connection.execute(
                "SELECT * FROM tasks WHERE status = 'queued' ORDER BY priority DESC, created_at ASC"
            ).fetchall()
            selected_row = None
            for row in queued:
                requirements = json.loads(row["requirements"])
                candidates = online_workers
                kernel_id = requirements.get("kernel_id") or (
                    json.loads(row["payload"]).get("kernel_id")
                    if row["kind"] in {"operator_benchmark", "kernel_pipeline", "kernel_autotune", "compiler_run"}
                    else None
                )
                if kernel_id:
                    kernel = self.get_kernel(str(kernel_id))
                    candidates = [worker for worker in candidates if kernel_supports_worker(kernel, worker)]
                selected = select_worker(candidates, requirements)
                if selected and selected["id"] == worker_id:
                    selected_row = row
                    break
            if selected_row is None:
                return None

            cursor = connection.execute(
                """
                UPDATE tasks SET status = 'running', assigned_worker_id = ?,
                                 attempts = attempts + 1, started_at = ?, error = NULL
                WHERE id = ? AND status = 'queued'
                """,
                (worker_id, now, selected_row["id"]),
            )
            if cursor.rowcount != 1:
                return None
            claimed = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (selected_row["id"],)
            ).fetchone()
        task = self._decode_task(claimed)
        self.append_event("task.leased", "scheduler", "task", task["id"], {"worker_id": worker_id})
        return task

    def complete_task(self, task_id: str, worker_id: str, data: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        status = data.get("status")
        if status not in {"succeeded", "failed"}:
            raise ValueError("completion status must be succeeded or failed")
        result = data.get("result") or {}
        error = data.get("error")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks SET status = ?, finished_at = ?, result = ?, error = ?, runtime_version = ?
                WHERE id = ? AND assigned_worker_id = ? AND status = 'running'
                """,
                (
                    status,
                    now,
                    json.dumps(result, separators=(",", ":")),
                    error,
                    data.get("runtime_version"),
                    task_id,
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("task is not running on this worker")
        task = self.get_task(task_id)
        artifact = (task.get("result") or {}).get("artifact") or {}
        kernel_id = (task.get("result") or {}).get("kernel_id")
        if artifact.get("digest") and kernel_id:
            self.bind_kernel_artifact(str(kernel_id), str(artifact["digest"]))
        if task["kind"] == "kernel_autotune" and task["result"]:
            self._record_tuning_run(task, worker_id)
            if task["result"].get("best_config"):
                self.update_kernel_tuning(kernel_id, task["result"]["best_config"], artifact.get("digest"))
        if task["kind"] in {"operator_benchmark", "kernel_pipeline", "kernel_autotune", "compiler_run"} and task["result"]:
            self._record_benchmark(task, worker_id)
        if task["kind"] in {"kernel_pipeline", "compiler_run"} and task["result"]:
            for stage in task["result"].get("pipeline") or []:
                self.append_event(
                    f"pipeline.{stage.get('name', 'unknown')}",
                    "worker",
                    "task",
                    task_id,
                    {"worker_id": worker_id, **stage},
                )
        if task["kind"] == "kernel_autotune" and task["result"]:
            for candidate in task["result"].get("candidates") or []:
                self.append_event(
                    "autotune.candidate",
                    "worker",
                    "task",
                    task_id,
                    {"worker_id": worker_id, **candidate},
                )
            self.append_event(
                "autotune.completed",
                "worker",
                "task",
                task_id,
                {"worker_id": worker_id, "best_config": task["result"].get("best_config")},
            )
        if task["kind"] == "experiment_run" and task["status"] == "succeeded" and task["result"]:
            self._record_experiment_run(task, worker_id)
            experiment_bundle = task["result"].get("experiment_bundle") or {}
            self.append_event(
                "experiment.completed",
                "worker",
                "experiment",
                str(task["result"].get("experiment_id") or task_id),
                {
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "workload": task["result"].get("workload"),
                    "artifact_digest": (task["result"].get("artifact") or {}).get("digest"),
                    "metric_count": len(task["result"].get("metrics") or experiment_bundle.get("metrics") or []),
                },
            )
        if task["kind"] == "model_pipeline" and task["result"]:
            for stage in task["result"].get("pipeline") or []:
                stage_name = str(stage.get("stage") or "unknown")
                self.append_event(
                    f"model_pipeline.{stage_name}",
                    "worker",
                    "task",
                    task_id,
                    {
                        "worker_id": worker_id,
                        "status": task["status"],
                        "stage": stage_name,
                        "exit_code": stage.get("exit_code"),
                        "elapsed_ms": stage.get("elapsed_ms"),
                        "parsed": stage.get("parsed"),
                        "validation_error": stage.get("validation_error"),
                    },
                )
            self._record_model_run(task, worker_id)
            self.append_event(
                "model_pipeline.completed",
                "worker",
                "task",
                task_id,
                {"worker_id": worker_id, "model": (task["result"].get("manifest") or {}).get("model", {}).get("name"), "artifact_digest": (task["result"].get("artifact") or {}).get("digest")},
            )
        self.append_event(
            "task.completed",
            "worker",
            "task",
            task_id,
            {"worker_id": worker_id, "status": status, "error": error, "runtime_version": data.get("runtime_version")},
        )
        return task

    def _record_model_run(self, task: dict[str, Any], worker_id: str) -> None:
        from edgeforge.model_pipeline import normalize_model_manifest

        result = task.get("result") or {}
        manifest = normalize_model_manifest(result.get("manifest") or task.get("payload") or {})
        benchmark = result.get("benchmark") or {}
        parsed = benchmark if isinstance(benchmark, dict) else {}
        correctness = result.get("correctness")
        if correctness is not None:
            correctness = 1 if bool(correctness) else 0
        summary = parsed.get("summary") if isinstance(parsed.get("summary"), dict) else parsed
        output_digest = None
        output = parsed.get("output_digest") or parsed.get("digest")
        if isinstance(output, str):
            output_digest = output
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO model_runs(
                    task_id, version, runtime_version, worker_id, task_status, model_name, dataset_name,
                    dataset_manifest_digest, transform_digest, frontend, compiler_backend,
                    compiler_identity, target_architecture, correctness, compile_ms,
                    first_call_ms, steady_latency_ms, peak_memory_mb, output_digest,
                    artifact_digest, manifest, summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task["id"], task["version"], task.get("runtime_version"), worker_id, task.get("status", "succeeded"),
                    manifest["model"]["name"], manifest["dataset"]["name"], manifest["dataset"].get("manifest_digest"),
                    manifest["transform_digest"], str(manifest["frontend"].get("name", "external")),
                    str(manifest["compiler"].get("backend", "unknown")), str(manifest["compiler"].get("identity", "unknown")),
                    (manifest.get("target") or {}).get("architecture"), correctness,
                    parsed.get("compile_ms", result.get("compile_ms")),
                    parsed.get("first_call_ms"), parsed.get("steady_latency_ms"), parsed.get("peak_memory_mb"), output_digest,
                    (result.get("artifact") or {}).get("digest"), json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(summary or {}, ensure_ascii=False, separators=(",", ":")), now,
                ),
            )

    def list_model_runs(self, model_name: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = min(1000, max(1, int(limit)))
        if model_name:
            query, params = "SELECT * FROM model_runs WHERE model_name = ? ORDER BY created_at DESC LIMIT ?", (model_name, limit)
        else:
            query, params = "SELECT * FROM model_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["manifest"] = json.loads(item["manifest"])
            item["summary"] = json.loads(item["summary"])
            if item.get("correctness") is not None:
                item["correctness"] = bool(item["correctness"])
            result.append(item)
        return result

    def model_regressions(
        self,
        model_name: str | None = None,
        dataset_name: str | None = None,
        backend: str | None = None,
        architecture: str | None = None,
        threshold: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Compare adjacent model runs without mixing backend or target paths."""
        threshold = min(10.0, max(0.0, float(threshold)))
        runs = self.list_model_runs(model_name, 1000)
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for run in runs:
            if dataset_name and run["dataset_name"] != dataset_name:
                continue
            if backend and run["compiler_backend"] != backend:
                continue
            if architecture and run.get("target_architecture") != architecture:
                continue
            manifest = run.get("manifest") or {}
            target = manifest.get("target") or {}
            key = (
                run["model_name"],
                run["dataset_name"],
                run["compiler_backend"],
                run.get("compiler_identity"),
                run.get("target_architecture"),
                target.get("device"),
                target.get("accelerator"),
                run.get("transform_digest"),
            )
            groups.setdefault(key, []).append(run)
        regressions: list[dict[str, Any]] = []
        for key, values in groups.items():
            values.sort(key=lambda value: (value["created_at"], value["task_id"]))
            if len(values) < 2:
                continue
            latest = values[-1]
            baseline = next(
                (value for value in reversed(values[:-1]) if value.get("task_status") == "succeeded" and value.get("correctness") is not False),
                None,
            )
            if baseline is None:
                continue
            common = {
                "model": key[0], "dataset": key[1], "backend": key[2], "compiler_identity": key[3],
                "architecture": key[4], "device": key[5], "accelerator": key[6],
                "transform_digest": key[7], "baseline": baseline, "latest": latest,
            }
            if latest.get("task_status") != "succeeded" or latest.get("correctness") is not True:
                regressions.append({**common, "kind": "correctness", "reason": "latest model run did not pass correctness"})
                continue
            for metric, label in (("steady_latency_ms", "steady_latency"), ("compile_ms", "compile_time")):
                previous_value = baseline.get(metric)
                latest_value = latest.get(metric)
                if isinstance(previous_value, (int, float)) and isinstance(latest_value, (int, float)) and previous_value > 0:
                    ratio = float(latest_value) / float(previous_value)
                    if ratio > 1.0 + threshold:
                        regressions.append({**common, "kind": label, "slowdown_ratio": round(ratio, 4)})
        return regressions

    def _record_experiment_run(self, task: dict[str, Any], worker_id: str) -> None:
        from edgeforge.experiment import ExperimentSpec

        spec = ExperimentSpec.from_payload((task.get("payload") or {}).get("spec"))
        result = task.get("result") or {}
        bundle = result.get("experiment_bundle") or {}
        if bundle.get("experiment_id") != spec.experiment_id:
            raise ValueError("experiment result id does not match submitted spec")
        metrics = result.get("metrics") or bundle.get("metrics") or []
        if not isinstance(metrics, list) or len(metrics) > 20_000:
            raise ValueError("experiment metrics must be an array with at most 20000 entries")
        summary = bundle.get("summary") or result.get("summary") or {}
        if not isinstance(summary, dict):
            raise ValueError("experiment summary must be an object")
        artifact_digest = (result.get("artifact") or {}).get("digest")
        source_digest = (bundle.get("source_result") or {}).get("digest")
        now = time.time()
        normalized_metrics = []
        for metric in metrics:
            if not isinstance(metric, dict):
                raise ValueError("each experiment metric must be an object")
            namespace = str(metric.get("namespace") or "")
            name = str(metric.get("name") or "")
            unit = str(metric.get("unit") or "scalar")
            context = metric.get("context") or {}
            value = metric.get("value")
            step = metric.get("step")
            if not namespace or not name or not isinstance(context, dict):
                raise ValueError("experiment metric namespace, name and context are required")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("experiment metric value must be numeric")
            if step is not None and (isinstance(step, bool) or not isinstance(step, int)):
                raise ValueError("experiment metric step must be an integer or null")
            normalized_metrics.append(
                (
                    task["id"],
                    spec.experiment_id,
                    namespace,
                    name,
                    float(value),
                    step,
                    unit,
                    json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                    now,
                )
            )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiment_runs(
                    task_id, experiment_id, version, runtime_version, worker_id,
                    workload, dataset, model, protocol, method, seed, spec,
                    summary, artifact_digest, source_digest, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task["id"],
                    spec.experiment_id,
                    task["version"],
                    task.get("runtime_version"),
                    worker_id,
                    spec.workload,
                    json.dumps(spec.dataset, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(spec.model, ensure_ascii=False, separators=(",", ":")),
                    spec.protocol,
                    spec.method,
                    spec.seed,
                    json.dumps(spec.to_dict(), ensure_ascii=False, separators=(",", ":")),
                    json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
                    artifact_digest,
                    source_digest,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO experiment_metrics(
                    task_id, experiment_id, namespace, name, value, step, unit, context, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                normalized_metrics,
            )

    def list_experiment_runs(
        self,
        workload: str | None = None,
        method: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = min(1000, max(1, int(limit)))
        clauses = []
        parameters: list[Any] = []
        if workload:
            clauses.append("workload = ?")
            parameters.append(workload)
        if method:
            clauses.append("method = ?")
            parameters.append(method)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM experiment_runs{where} ORDER BY created_at DESC LIMIT ?",
                (*parameters, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for field in ("dataset", "model", "spec", "summary"):
                item[field] = json.loads(item[field])
            result.append(item)
        return result

    def list_experiment_metrics(
        self,
        experiment_id: str | None = None,
        name: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        limit = min(20_000, max(1, int(limit)))
        clauses = []
        parameters: list[Any] = []
        if experiment_id:
            clauses.append("experiment_id = ?")
            parameters.append(experiment_id)
        if name:
            clauses.append("name = ?")
            parameters.append(name)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM experiment_metrics{where} ORDER BY created_at DESC, id DESC LIMIT ?",
                (*parameters, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["context"] = json.loads(item["context"])
            result.append(item)
        return result

    @staticmethod
    def _decode_model(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["descriptor"] = json.loads(item["descriptor"])
        return item

    @staticmethod
    def _decode_gate_policy(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["rules"] = json.loads(item["rules"])
        return item

    @staticmethod
    def _decode_gate_evaluation(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for field in ("policy_snapshot", "metric_snapshot", "rule_results"):
            item[field] = json.loads(item[field])
        return item

    def _get_experiment_run(self, experiment_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_runs WHERE experiment_id = ? ORDER BY created_at DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        item = dict(row)
        for field in ("dataset", "model", "spec", "summary"):
            item[field] = json.loads(item[field])
        return item

    @staticmethod
    def _experiment_comparison_group(experiment: dict[str, Any]) -> str:
        spec = experiment.get("spec") or {}
        metadata = spec.get("metadata") or {}
        return str(metadata.get("comparison_group") or spec.get("comparison_group") or "").strip()

    def register_model(self, data: dict[str, Any]) -> dict[str, Any]:
        from edgeforge.gate import normalize_model

        model = normalize_model(data)
        experiment = self._get_experiment_run(model["source_experiment_id"])
        experiment_group = self._experiment_comparison_group(experiment)
        expected = (experiment["workload"], experiment["protocol"], experiment_group)
        actual = (model["workload"], model["protocol"], model["comparison_group"])
        if not experiment_group:
            raise ValueError("source experiment has no comparison_group")
        if actual != expected:
            raise ValueError("model workload, protocol and comparison_group must match the source experiment")
        now = time.time()
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO models(
                        id, name, workload, protocol, comparison_group, source_experiment_id,
                        checkpoint_digest, descriptor, status, created_at, updated_at, registered_by_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?, ?, ?)
                    """,
                    (
                        model["id"],
                        model["name"],
                        model["workload"],
                        model["protocol"],
                        model["comparison_group"],
                        model["source_experiment_id"],
                        model["checkpoint_digest"],
                        json.dumps(model["descriptor"], ensure_ascii=False, separators=(",", ":")),
                        now,
                        now,
                        self.version,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise RuntimeError(f"model id already exists: {model['id']}") from error
        registered = self.get_model(model["id"])
        self.append_event("model.registered", "control", "model", model["id"], {"model": registered})
        return registered

    def get_model(self, model_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        if row is None:
            raise KeyError(model_id)
        return self._decode_model(row)

    def list_models(
        self,
        workload: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = min(1000, max(1, int(limit)))
        clauses = []
        parameters: list[Any] = []
        if workload:
            clauses.append("workload = ?")
            parameters.append(workload)
        if status:
            clauses.append("status = ?")
            parameters.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM models{where} ORDER BY created_at DESC LIMIT ?", (*parameters, limit)
            ).fetchall()
        return [self._decode_model(row) for row in rows]

    def create_gate_policy(self, data: dict[str, Any]) -> dict[str, Any]:
        from edgeforge.gate import normalize_policy

        policy = normalize_policy(data)
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO gate_policies(
                        id, version, workload, protocol, comparison_group, rules, created_at, created_by_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        policy["id"],
                        policy["version"],
                        policy["workload"],
                        policy["protocol"],
                        policy["comparison_group"],
                        json.dumps(policy["rules"], ensure_ascii=False, separators=(",", ":")),
                        time.time(),
                        self.version,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise RuntimeError(f"gate policy id already exists and policies are immutable: {policy['id']}") from error
        created = self.get_gate_policy(policy["id"])
        self.append_event("gate.policy.created", "control", "gate_policy", policy["id"], {"policy": created})
        return created

    def get_gate_policy(self, policy_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM gate_policies WHERE id = ?", (policy_id,)).fetchone()
        if row is None:
            raise KeyError(policy_id)
        return self._decode_gate_policy(row)

    def list_gate_policies(
        self,
        workload: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = min(1000, max(1, int(limit)))
        with self._connect() as connection:
            if workload:
                rows = connection.execute(
                    "SELECT * FROM gate_policies WHERE workload = ? ORDER BY created_at DESC LIMIT ?",
                    (workload, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM gate_policies ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._decode_gate_policy(row) for row in rows]

    @staticmethod
    def _summary_metric(summary: dict[str, Any], path: str) -> float:
        value: Any = summary
        for component in path.split("."):
            if not component or not isinstance(value, dict) or component not in value:
                raise ValueError(f"summary metric not found: summary.{path}")
            value = value[component]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"summary metric must be numeric: summary.{path}")
        return float(value)

    def _resolve_gate_metric(
        self,
        connection: sqlite3.Connection,
        experiment: dict[str, Any],
        rule: dict[str, Any],
    ) -> dict[str, Any]:
        metric = rule["metric"]
        if metric.startswith("summary."):
            value = self._summary_metric(experiment["summary"], metric.removeprefix("summary."))
            return {"metric": metric, "source": "summary", "step": None, "value": value, "unit": "scalar"}
        clauses = ["task_id = ?", "name = ?"]
        parameters: list[Any] = [experiment["task_id"], metric]
        if rule.get("namespace"):
            clauses.append("namespace = ?")
            parameters.append(rule["namespace"])
        if rule.get("step") is None:
            clauses.append("step IS NULL")
        else:
            clauses.append("step = ?")
            parameters.append(rule["step"])
        rows = connection.execute(
            f"SELECT * FROM experiment_metrics WHERE {' AND '.join(clauses)} ORDER BY id", parameters
        ).fetchall()
        if not rows:
            raise ValueError(f"experiment metric not found or step is ambiguous: {metric}")
        if len(rows) != 1:
            raise ValueError(f"gate metric must resolve to exactly one value: {metric}")
        row = dict(rows[0])
        return {
            "metric": metric,
            "source": "experiment_metrics",
            "namespace": row["namespace"],
            "step": row["step"],
            "value": float(row["value"]),
            "unit": row["unit"],
            "context": json.loads(row["context"]),
        }

    def evaluate_gate(self, model_id: str, policy_id: str, experiment_id: str | None = None) -> dict[str, Any]:
        from edgeforge.gate import compare

        model = self.get_model(model_id)
        policy = self.get_gate_policy(policy_id)
        bound_experiment_id = experiment_id or model["source_experiment_id"]
        if bound_experiment_id != model["source_experiment_id"]:
            raise ValueError("gate evaluation must use the model source experiment")
        experiment = self._get_experiment_run(bound_experiment_id)
        experiment_group = self._experiment_comparison_group(experiment)
        model_scope = (model["workload"], model["protocol"], model["comparison_group"])
        policy_scope = (policy["workload"], policy["protocol"], policy["comparison_group"])
        experiment_scope = (experiment["workload"], experiment["protocol"], experiment_group)
        if model_scope != policy_scope or model_scope != experiment_scope:
            raise ValueError("model, policy and experiment must share workload, protocol and comparison_group")
        if model["status"] != "candidate":
            raise RuntimeError("only candidate models can be evaluated")
        metric_snapshot = []
        rule_results = []
        with self._lock, self._connect() as connection:
            for index, rule in enumerate(policy["rules"]):
                snapshot = self._resolve_gate_metric(connection, experiment, rule)
                passed = compare(snapshot["value"], rule["operator"], rule["threshold"])
                metric_snapshot.append(snapshot)
                rule_results.append(
                    {
                        "index": index,
                        "metric": rule["metric"],
                        "operator": rule["operator"],
                        "threshold": rule["threshold"],
                        "actual": snapshot["value"],
                        "passed": passed,
                    }
                )
            status = "PASS" if all(result["passed"] for result in rule_results) else "FAIL"
            model_status = "accepted" if status == "PASS" else "rejected"
            evaluation_id = f"evaluation-{uuid.uuid4().hex}"
            policy_snapshot = dict(policy)
            policy_snapshot["experiment_task_id"] = experiment["task_id"]
            now = time.time()
            cursor = connection.execute(
                "UPDATE models SET status = ?, updated_at = ? WHERE id = ? AND status = 'candidate'",
                (model_status, now, model_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("model is no longer a candidate")
            connection.execute(
                """
                INSERT INTO gate_evaluations(
                    id, model_id, experiment_id, policy_id, policy_snapshot, metric_snapshot,
                    rule_results, status, created_at, created_by_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    model_id,
                    bound_experiment_id,
                    policy_id,
                    json.dumps(policy_snapshot, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(metric_snapshot, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(rule_results, ensure_ascii=False, separators=(",", ":")),
                    status,
                    now,
                    self.version,
                ),
            )
        evaluation = self.get_gate_evaluation(evaluation_id)
        self.append_event(
            "gate.evaluated",
            "control",
            "gate_evaluation",
            evaluation_id,
            {"model_id": model_id, "policy_id": policy_id, "status": status, "model_status": model_status},
        )
        return evaluation

    def get_gate_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM gate_evaluations WHERE id = ?", (evaluation_id,)
            ).fetchone()
        if row is None:
            raise KeyError(evaluation_id)
        return self._decode_gate_evaluation(row)

    def list_gate_evaluations(
        self,
        model_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = min(1000, max(1, int(limit)))
        with self._connect() as connection:
            if model_id:
                rows = connection.execute(
                    "SELECT * FROM gate_evaluations WHERE model_id = ? ORDER BY created_at DESC LIMIT ?",
                    (model_id, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM gate_evaluations ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._decode_gate_evaluation(row) for row in rows]

    @staticmethod
    def _transition_reason(reason: str) -> str:
        normalized = str(reason or "").strip()
        if not normalized:
            raise ValueError("a non-empty transition reason is required")
        return normalized

    def promote_model(self, model_id: str, reason: str) -> dict[str, Any]:
        reason = self._transition_reason(reason)
        model = self.get_model(model_id)
        if model["status"] != "accepted":
            raise RuntimeError("only accepted models can be promoted")
        now = time.time()
        with self._lock, self._connect() as connection:
            previous = connection.execute(
                """
                SELECT id FROM models
                WHERE workload = ? AND protocol = ? AND comparison_group = ? AND status = 'production'
                """,
                (model["workload"], model["protocol"], model["comparison_group"]),
            ).fetchone()
            previous_id = previous["id"] if previous else None
            if previous_id:
                connection.execute(
                    "UPDATE models SET status = 'rolled_back', updated_at = ? WHERE id = ?",
                    (now, previous_id),
                )
            cursor = connection.execute(
                "UPDATE models SET status = 'production', updated_at = ? WHERE id = ? AND status = 'accepted'",
                (now, model_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("model is no longer accepted")
        promoted = self.get_model(model_id)
        self.append_event(
            "model.promoted",
            "operator",
            "model",
            model_id,
            {"reason": reason, "previous_production_id": previous_id, "model": promoted},
        )
        return promoted

    def reject_model(self, model_id: str, reason: str) -> dict[str, Any]:
        reason = self._transition_reason(reason)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE models SET status = 'rejected', updated_at = ? WHERE id = ? AND status = 'candidate'",
                (time.time(), model_id),
            )
            if cursor.rowcount != 1:
                if connection.execute("SELECT 1 FROM models WHERE id = ?", (model_id,)).fetchone() is None:
                    raise KeyError(model_id)
                raise RuntimeError("only candidate models can be explicitly rejected")
        rejected = self.get_model(model_id)
        self.append_event("model.rejected", "operator", "model", model_id, {"reason": reason, "model": rejected})
        return rejected

    def rollback_model(self, model_id: str, target_model_id: str, reason: str) -> dict[str, Any]:
        reason = self._transition_reason(reason)
        if model_id == target_model_id:
            raise ValueError("rollback target must differ from the production model")
        current = self.get_model(model_id)
        target = self.get_model(target_model_id)
        if current["status"] != "production":
            raise RuntimeError("rollback source must be the production model")
        if target["status"] not in {"accepted", "rolled_back"}:
            raise RuntimeError("rollback target must have passed a gate")
        scope = ("workload", "protocol", "comparison_group")
        if any(current[field] != target[field] for field in scope):
            raise ValueError("rollback models must share workload, protocol and comparison_group")
        now = time.time()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE models SET status = 'rolled_back', updated_at = ? WHERE id = ? AND status = 'production'",
                (now, model_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("production model changed before rollback")
            cursor = connection.execute(
                """
                UPDATE models SET status = 'production', updated_at = ?
                WHERE id = ? AND status IN ('accepted', 'rolled_back')
                """,
                (now, target_model_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("rollback target changed before rollback")
        restored = self.get_model(target_model_id)
        self.append_event(
            "model.rolled_back",
            "operator",
            "model",
            model_id,
            {"reason": reason, "target_model_id": target_model_id, "restored_model": restored},
        )
        return {"rolled_back": self.get_model(model_id), "production": restored}

    def register_kernel(self, data: dict[str, Any]) -> dict[str, Any]:
        spec = KernelSpec.from_payload(data)
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO kernels(id, operator, backend, version, architectures, accelerators, dtypes,
                                    shape_constraints, metadata, compiler, artifact_digest,
                                    status, created_at, registered_by_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    operator = excluded.operator,
                    backend = excluded.backend,
                    version = excluded.version,
                    architectures = excluded.architectures,
                    accelerators = excluded.accelerators,
                    dtypes = excluded.dtypes,
                    shape_constraints = excluded.shape_constraints,
                    metadata = excluded.metadata,
                    compiler = excluded.compiler,
                    artifact_digest = COALESCE(excluded.artifact_digest, kernels.artifact_digest),
                    status = excluded.status,
                    registered_by_version = excluded.registered_by_version
                """,
                (
                    spec.id,
                    spec.operator,
                    spec.backend,
                    spec.version,
                    json.dumps(list(spec.architectures), separators=(",", ":")),
                    json.dumps(list(spec.accelerators), separators=(",", ":")),
                    json.dumps(list(spec.dtypes), separators=(",", ":")),
                    json.dumps(spec.shape_constraints, separators=(",", ":")),
                    json.dumps(spec.metadata, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(data.get("compiler") or {}, ensure_ascii=False, separators=(",", ":")),
                    data.get("artifact_digest"),
                    data.get("status", "active"),
                    now,
                    self.version,
                ),
            )
        kernel = self.get_kernel(spec.id)
        self.append_event("kernel.registered", "control", "kernel", spec.id, kernel)
        return kernel

    def get_kernel(self, kernel_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM kernels WHERE id = ?", (kernel_id,)).fetchone()
        if row is None:
            raise KeyError(kernel_id)
        item = dict(row)
        item["architectures"] = json.loads(item["architectures"])
        item["accelerators"] = json.loads(item["accelerators"])
        item["dtypes"] = json.loads(item["dtypes"])
        item["shape_constraints"] = json.loads(item["shape_constraints"])
        item["metadata"] = json.loads(item["metadata"])
        item["compiler"] = json.loads(item["compiler"])
        return item

    def list_kernels(self, operator: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = min(1000, max(1, int(limit)))
        with self._connect() as connection:
            if operator:
                rows = connection.execute(
                    "SELECT * FROM kernels WHERE operator IN (?, '*') ORDER BY created_at DESC, id LIMIT ?",
                    (operator, limit),
                ).fetchall()
            else:
                rows = connection.execute("SELECT * FROM kernels ORDER BY created_at DESC, id LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["architectures"] = json.loads(item["architectures"])
            item["accelerators"] = json.loads(item["accelerators"])
            item["dtypes"] = json.loads(item["dtypes"])
            item["shape_constraints"] = json.loads(item["shape_constraints"])
            item["metadata"] = json.loads(item["metadata"])
            item["compiler"] = json.loads(item["compiler"])
            result.append(item)
        return result

    def bind_kernel_artifact(self, kernel_id: str, digest: str) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE kernels SET artifact_digest = ? WHERE id = ?", (digest, kernel_id)
            )
            if cursor.rowcount != 1:
                raise KeyError(kernel_id)

    def update_kernel_tuning(self, kernel_id: str | None, config: dict[str, Any], digest: str | None = None) -> None:
        if not kernel_id:
            raise ValueError("autotune result requires kernel_id")
        kernel = self.get_kernel(kernel_id)
        metadata = dict(kernel.get("metadata") or {})
        metadata["tuning_config"] = config
        metadata["tuning_version"] = self.version
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE kernels SET metadata = ?, artifact_digest = COALESCE(?, artifact_digest) WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), digest, kernel_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(kernel_id)
        self.append_event("kernel.tuned", "control", "kernel", kernel_id, {"config": config, "artifact_digest": digest})

    def record_artifact(self, artifact: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(digest, algorithm, size_bytes, kind, media_type, name,
                                      storage_path, metadata, created_at, created_by_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(digest) DO UPDATE SET
                    kind = excluded.kind,
                    media_type = excluded.media_type,
                    name = excluded.name,
                    metadata = excluded.metadata
                """,
                (
                    artifact["digest"],
                    artifact.get("algorithm", "sha256"),
                    int(artifact["size_bytes"]),
                    artifact["kind"],
                    artifact["media_type"],
                    artifact["name"],
                    artifact["storage_path"],
                    json.dumps(artifact.get("metadata") or {}, ensure_ascii=False, separators=(",", ":")),
                    now,
                    self.version,
                ),
            )
        stored = self.get_artifact(artifact["digest"])
        self.append_event("artifact.stored", "control", "artifact", artifact["digest"], stored)
        return stored

    def get_artifact(self, digest: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM artifacts WHERE digest = ?", (digest,)).fetchone()
        if row is None:
            raise KeyError(digest)
        item = dict(row)
        item["metadata"] = json.loads(item["metadata"])
        return item

    def list_artifacts(self, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = min(1000, max(1, int(limit)))
        with self._connect() as connection:
            if kind:
                rows = connection.execute(
                    "SELECT * FROM artifacts WHERE kind = ? ORDER BY created_at DESC LIMIT ?",
                    (kind, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item["metadata"])
            result.append(item)
        return result

    def _record_tuning_run(self, task: dict[str, Any], worker_id: str) -> None:
        result = task["result"]
        operator = result.get("operator") or task["payload"].get("operator") or {}
        worker = self.get_worker(worker_id)
        capabilities = worker.get("capabilities") or {}
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO tuning_runs(
                    task_id, version, runtime_version, worker_id, architecture, hardware_fingerprint,
                    operator, kernel_id, kernel_version, shape, dtype, backend, search_space, candidates,
                    best_config, artifact_digest, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task["id"],
                    task["version"],
                    task.get("runtime_version"),
                    worker_id,
                    capabilities.get("architecture", "unknown"),
                    capabilities.get("hardware_fingerprint"),
                    operator.get("name", "unknown"),
                    result.get("kernel_id") or task["payload"].get("kernel_id"),
                    result.get("kernel_version"),
                    json.dumps(operator.get("shape") or []),
                    operator.get("dtype", "unknown"),
                    result.get("backend") or "unknown",
                    json.dumps(result.get("search_space") or [], separators=(",", ":")),
                    json.dumps(result.get("candidates") or [], separators=(",", ":")),
                    json.dumps(result.get("best_config"), separators=(",", ":")) if result.get("best_config") else None,
                    (result.get("artifact") or {}).get("digest"),
                    time.time(),
                ),
            )

    def list_tuning_runs(self, operator: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = min(1000, max(1, int(limit)))
        with self._connect() as connection:
            if operator:
                rows = connection.execute(
                    "SELECT * FROM tuning_runs WHERE operator = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                    (operator, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM tuning_runs ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
                ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["shape"] = json.loads(item["shape"])
            item["search_space"] = json.loads(item["search_space"])
            item["candidates"] = json.loads(item["candidates"])
            item["best_config"] = json.loads(item["best_config"]) if item["best_config"] else None
            result.append(item)
        return result

    def _record_schedule_decision(self, task_id: str, plan: dict[str, Any]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO schedule_decisions(task_id, version, operator, policy, candidates,
                                               selected, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    self.version,
                    json.dumps(plan["operator"], separators=(",", ":")),
                    json.dumps(plan["policy"], separators=(",", ":")),
                    json.dumps(plan["candidates"], separators=(",", ":")),
                    json.dumps(plan["selected"], separators=(",", ":")),
                    plan["reason"],
                    time.time(),
                ),
            )
        self.append_event("scheduler.decision", "scheduler", "task", task_id, plan)

    def list_schedule_decisions(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = min(1000, max(1, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM schedule_decisions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["operator"] = json.loads(item["operator"])
            item["policy"] = json.loads(item["policy"])
            item["candidates"] = json.loads(item["candidates"])
            item["selected"] = json.loads(item["selected"])
            result.append(item)
        return result

    def append_event(
        self,
        event_type: str,
        source: str,
        entity_type: str | None,
        entity_id: str | None,
        payload: dict[str, Any],
        version: str | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO events(version, event_type, source, entity_type, entity_id, created_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    version or self.version,
                    event_type,
                    source,
                    entity_type,
                    entity_id,
                    time.time(),
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                ),
            )

    def list_events(self, version: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = min(1000, max(1, int(limit)))
        with self._connect() as connection:
            if version:
                rows = connection.execute(
                    "SELECT * FROM events WHERE version = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                    (version, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM events ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
                ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result

    def _record_benchmark(self, task: dict[str, Any], worker_id: str) -> None:
        result = task["result"]
        operator = result.get("operator") or task["payload"].get("operator") or {}
        worker = self.get_worker(worker_id)
        capabilities = worker.get("capabilities") or {}
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO benchmarks(task_id, version, runtime_version, worker_id, architecture,
                                       hardware_fingerprint, operator, kernel_id, kernel_version,
                                       artifact_digest, compile_ms, shape, dtype, backend, correctness,
                                       timings, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task["id"],
                    task["version"],
                    task.get("runtime_version"),
                    worker_id,
                    capabilities.get("architecture", "unknown"),
                    capabilities.get("hardware_fingerprint"),
                    operator.get("name", "unknown"),
                    result.get("kernel_id") or operator.get("kernel_id") or task["payload"].get("kernel_id"),
                    result.get("kernel_version") or operator.get("kernel_version"),
                    (result.get("artifact") or {}).get("digest"),
                    result.get("compile_ms"),
                    json.dumps(operator.get("shape") or []),
                    operator.get("dtype", "unknown"),
                    result.get("backend") or operator.get("backend", "unknown"),
                    1 if result.get("correctness") else 0,
                    json.dumps(result.get("timings_ms") or []),
                    json.dumps(result.get("summary") or {}, separators=(",", ":")),
                    time.time(),
                ),
            )

    def list_benchmarks(self, operator: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = min(1000, max(1, int(limit)))
        with self._connect() as connection:
            if operator:
                rows = connection.execute(
                    "SELECT * FROM benchmarks WHERE operator = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                    (operator, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM benchmarks ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
                ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["shape"] = json.loads(item["shape"])
            item["timings"] = json.loads(item["timings"])
            item["summary"] = json.loads(item["summary"])
            item["correctness"] = bool(item["correctness"])
            result.append(item)
        return result

    def performance_regressions(self, operator: str | None = None, threshold: float = 0.2) -> list[dict[str, Any]]:
        threshold = min(10.0, max(0.0, float(threshold)))
        rows = self.list_benchmarks(operator, 1000)
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            key = (
                row["operator"],
                json.dumps(row["shape"], separators=(",", ":")),
                row["dtype"],
                row["backend"],
                row.get("kernel_id"),
                row["architecture"],
            )
            groups.setdefault(key, []).append(row)
        regressions = []
        for key, values in groups.items():
            values.sort(key=lambda value: (value["created_at"], value["id"]))
            if len(values) < 2:
                continue
            previous = values[-2]
            latest = values[-1]
            previous_median = float(previous["summary"].get("median_ms") or 0.0)
            latest_median = float(latest["summary"].get("median_ms") or 0.0)
            if previous_median > 0 and latest_median > previous_median * (1.0 + threshold):
                regressions.append(
                    {
                        "operator": key[0],
                        "shape": json.loads(key[1]),
                        "dtype": key[2],
                        "backend": key[3],
                        "kernel_id": key[4],
                        "architecture": key[5],
                        "previous": previous,
                        "latest": latest,
                        "slowdown_ratio": round(latest_median / previous_median, 4),
                    }
                )
        return regressions

    def record_release(self, version: str, summary: str, metadata: dict[str, Any], status: str = "active") -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO releases(version, created_at, status, summary, metadata) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(version) DO UPDATE SET
                    status = excluded.status,
                    summary = excluded.summary,
                    metadata = excluded.metadata
                """,
                (version, time.time(), status, summary, json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))),
            )
        release = self.get_release(version)
        self.append_event("release.recorded", "control", "release", version, release, version=version)
        return release

    def get_release(self, version: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM releases WHERE version = ?", (version,)).fetchone()
        if row is None:
            raise KeyError(version)
        item = dict(row)
        item["metadata"] = json.loads(item["metadata"])
        return item

    def list_releases(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM releases ORDER BY created_at DESC").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item["metadata"])
            result.append(item)
        return result
