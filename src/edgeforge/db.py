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
        if kind not in {"command", "benchmark", "operator_benchmark", "kernel_pipeline", "kernel_autotune", "compiler_run"}:
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
            kernel_id = payload.get("kernel_id") or requirements.get("kernel_id")
            if kind == "kernel_pipeline" and not kernel_id:
                raise ValueError("kernel_pipeline requires kernel_id")
            if kind == "kernel_autotune" and not kernel_id:
                raise ValueError("kernel_autotune requires kernel_id")
            if kernel_id:
                kernel = self.get_kernel(str(kernel_id))
                if not kernel_supports_operator(kernel, payload.get("operator") or {}):
                    raise ValueError(f"kernel {kernel_id} does not support this operator or dtype")
                if kind == "kernel_autotune":
                    from edgeforge.autotune import normalize_triton_matmul_candidates

                    if kernel.get("backend") != "triton" or payload["operator"].get("name") != "matmul":
                        raise ValueError("kernel_autotune currently requires a Triton MatMul kernel")
                    payload["candidates"] = normalize_triton_matmul_candidates(payload.get("candidates") or [])
                payload["operator"] = {**payload["operator"], "backend": kernel["backend"]}
                payload["kernel_id"] = str(kernel_id)
                payload["kernel"] = kernel
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
        self.append_event(
            "task.completed",
            "worker",
            "task",
            task_id,
            {"worker_id": worker_id, "status": status, "error": error, "runtime_version": data.get("runtime_version")},
        )
        return task

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
