import base64
import tempfile
import threading
import unittest
from pathlib import Path

from edgeforge import __version__
from edgeforge.api import ControlServer
from edgeforge.client import APIError, Client
from edgeforge.db import Store
from edgeforge.worker import Worker


class APITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        store = Store(Path(self.temporary.name) / "edgeforge.db")
        self.server = ControlServer(("127.0.0.1", 0), store, "test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.client = Client(f"http://{host}:{port}", "test-token")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def test_authentication_is_required(self):
        bad_client = Client(self.client.base_url, "wrong-token")
        with self.assertRaisesRegex(APIError, "HTTP 401"):
            bad_client.request("GET", "/api/v1/workers")

    def test_backend_capabilities_endpoint(self):
        response = self.client.request("GET", "/api/v1/backend-capabilities")
        names = {item["name"] for item in response["backends"]}
        self.assertIn("iree", names)
        self.assertIn("rknn", names)

    def test_full_task_protocol(self):
        registration = {
            "id": "worker-test",
            "hostname": "test",
            "capabilities": {
                "architecture": "riscv64",
                "accelerators": [],
                "cpu_count": 4,
                "memory_total_mb": 16000,
            },
            "metrics": {"load_1m": 0, "memory_available_mb": 12000},
            "labels": {"role": "test"},
            "version": "test",
        }
        registered = self.client.request("POST", "/api/v1/workers/register", registration)
        self.assertEqual(registered["worker"]["status"], "online")

        task = self.client.request(
            "POST",
            "/api/v1/tasks",
            {
                "kind": "command",
                "payload": {"argv": ["true"]},
                "requirements": {"architectures": ["riscv64"]},
            },
        )
        leased = self.client.request("POST", "/api/v1/workers/worker-test/lease", {})["task"]
        self.assertEqual(leased["id"], task["id"])

        completed = self.client.request(
            "POST",
            f"/api/v1/tasks/{task['id']}/complete",
            {
                "worker_id": "worker-test",
                "status": "succeeded",
                "result": {"exit_code": 0},
            },
        )
        self.assertEqual(completed["status"], "succeeded")

        events = self.client.request("GET", f"/api/v1/events?version={__version__}")["events"]
        self.assertTrue(any(event["event_type"] == "task.completed" for event in events))

    def test_model_pipeline_api_persists_run_and_artifact(self):
        registration = {
            "id": "model-api-worker",
            "hostname": "model-api-worker",
            "capabilities": {"architecture": "x86_64", "accelerators": [], "cpu_count": 2, "memory_total_mb": 4096},
            "metrics": {"load_1m": 0, "memory_available_mb": 2000},
            "labels": {},
            "version": __version__,
        }
        self.client.request("POST", "/api/v1/workers/register", registration)
        task = self.client.request(
            "POST",
            "/api/v1/model-pipelines",
            {
                "model": {"name": "api-tiny"},
                "dataset": {"name": "synthetic", "manifest_digest": "sha256:dataset"},
                "transforms": [{"name": "window", "version": "v1", "config": {"length": 8}}],
                "frontend": {"name": "external"},
                "compiler": {"backend": "python-reference", "identity": "test"},
                "target": {"architecture": "x86_64"},
                "requirements": {"worker_ids": ["model-api-worker"]},
            },
        )
        leased = self.client.request("POST", "/api/v1/workers/model-api-worker/lease", {})["task"]
        self.assertEqual(leased["id"], task["id"])
        worker = Worker(self.client, "model-api-worker", {}, Path(self.temporary.name) / "model-work", 1, {"true"}, False)
        result = worker.execute(leased)
        completed = self.client.request("POST", f"/api/v1/tasks/{task['id']}/complete", {"worker_id": "model-api-worker", "runtime_version": __version__, "status": "succeeded", "result": result})
        self.assertEqual(completed["status"], "succeeded")
        runs = self.client.request("GET", "/api/v1/model-runs?model=api-tiny")["model_runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["compiler_backend"], "python-reference")
        self.assertEqual(len(self.client.request("GET", "/api/v1/artifacts?kind=model-compiler-manifest")["artifacts"]), 1)

    def test_model_regressions_endpoint(self):
        response = self.client.request("GET", "/api/v1/model-regressions?model=missing&threshold=0.2")
        self.assertEqual(response["regressions"], [])

    def test_release_ledger(self):
        release = self.client.request(
            "POST",
            "/api/v1/releases",
            {"version": "0.2.0-test", "summary": "test release", "metadata": {"tests": 1}},
        )
        self.assertEqual(release["metadata"]["tests"], 1)
        versions = [item["version"] for item in self.client.request("GET", "/api/v1/releases")["releases"]]
        self.assertIn("0.2.0-test", versions)

    def test_kernel_registry_api(self):
        kernel = self.client.request(
            "POST",
            "/api/v1/kernels",
            {
                "id": "kernel-api-softmax-v1",
                "operator": "softmax",
                "backend": "python-reference",
                "version": "1",
                "architectures": ["aarch64"],
                "dtypes": ["fp32"],
            },
        )
        self.assertEqual(kernel["registered_by_version"], __version__)
        kernels = self.client.request("GET", "/api/v1/kernels?operator=softmax")["kernels"]
        self.assertEqual([item["id"] for item in kernels], ["kernel-api-softmax-v1"])

    def test_kernel_pipeline_uploads_and_binds_artifact(self):
        registration = {
            "id": "pipeline-worker",
            "hostname": "pipeline-worker",
            "capabilities": {
                "architecture": "x86_64",
                "accelerators": [],
                "cpu_count": 4,
                "memory_total_mb": 16000,
                "hardware_fingerprint": "pipeline-fingerprint",
            },
            "metrics": {"load_1m": 0, "memory_available_mb": 12000},
            "labels": {},
            "version": __version__,
        }
        self.client.request("POST", "/api/v1/workers/register", registration)
        self.client.request(
            "POST",
            "/api/v1/kernels",
            {
                "id": "kernel-pipeline-softmax-v1",
                "operator": "softmax",
                "backend": "python-reference",
                "version": "1",
                "architectures": ["x86_64"],
                "dtypes": ["fp32"],
            },
        )
        task = self.client.request(
            "POST",
            "/api/v1/tasks",
            {
                "kind": "kernel_pipeline",
                "payload": {
                    "operator": {"name": "softmax", "shape": [16], "dtype": "fp32"},
                    "kernel_id": "kernel-pipeline-softmax-v1",
                    "repeats": 2,
                },
                "requirements": {"worker_ids": ["pipeline-worker"], "kernel_id": "kernel-pipeline-softmax-v1"},
            },
        )
        leased = self.client.request("POST", "/api/v1/workers/pipeline-worker/lease", {})["task"]
        worker = Worker(
            self.client,
            "pipeline-worker",
            {},
            Path(self.temporary.name) / "work",
            1,
            {"true"},
            False,
        )
        result = worker.execute(leased)
        completed = self.client.request(
            "POST",
            f"/api/v1/tasks/{task['id']}/complete",
            {"worker_id": "pipeline-worker", "runtime_version": __version__, "status": "succeeded", "result": result},
        )
        artifact = completed["result"]["artifact"]
        self.assertEqual(len(artifact["digest"]), 64)
        self.assertNotIn("artifact_upload", completed["result"])
        artifacts = self.client.request("GET", "/api/v1/artifacts")["artifacts"]
        self.assertEqual([item["digest"] for item in artifacts], [artifact["digest"]])
        kernels = self.client.request("GET", "/api/v1/kernels?operator=softmax")["kernels"]
        self.assertEqual(kernels[0]["artifact_digest"], artifact["digest"])
        benchmarks = self.client.request("GET", "/api/v1/benchmarks?operator=softmax")["benchmarks"]
        self.assertEqual(benchmarks[0]["artifact_digest"], artifact["digest"])
        self.assertIsNotNone(benchmarks[0]["compile_ms"])
        event_types = {item["event_type"] for item in self.client.request("GET", f"/api/v1/events?version={__version__}")["events"]}
        self.assertTrue({"pipeline.compile", "pipeline.correctness", "pipeline.benchmark"}.issubset(event_types))

    def test_kernel_autotune_persists_candidates_and_updates_kernel(self):
        registration = {
            "id": "autotune-worker",
            "hostname": "autotune-worker",
            "capabilities": {
                "architecture": "x86_64",
                "accelerators": ["nvidia-gpu"],
                "cpu_count": 4,
                "memory_total_mb": 16000,
                "hardware_fingerprint": "autotune-fingerprint",
            },
            "metrics": {"load_1m": 0, "memory_available_mb": 12000},
            "labels": {},
            "version": __version__,
        }
        self.client.request("POST", "/api/v1/workers/register", registration)
        kernel_id = "kernel-triton-autotune-v1"
        self.client.request(
            "POST",
            "/api/v1/kernels",
            {
                "id": kernel_id,
                "operator": "matmul",
                "backend": "triton",
                "version": "1",
                "architectures": ["x86_64"],
                "accelerators": ["nvidia-gpu"],
                "dtypes": ["fp16"],
            },
        )
        task = self.client.request(
            "POST",
            "/api/v1/tasks",
            {
                "kind": "kernel_autotune",
                "payload": {
                    "operator": {"name": "matmul", "shape": [128, 128, 128], "dtype": "fp16"},
                    "kernel_id": kernel_id,
                },
                "requirements": {"worker_ids": ["autotune-worker"], "kernel_id": kernel_id},
            },
        )
        self.client.request("POST", "/api/v1/workers/autotune-worker/lease", {})
        best_config = {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 3}
        slower_config = {"block_m": 32, "block_n": 32, "block_k": 32, "num_warps": 4, "num_stages": 2}
        result = {
            "operator": {"name": "matmul", "shape": [128, 128, 128], "dtype": "fp16"},
            "backend": "triton",
            "kernel_id": kernel_id,
            "kernel_version": "1",
            "correctness": True,
            "timings_ms": [0.02, 0.021],
            "summary": {"runs": 2, "min_ms": 0.02, "median_ms": 0.0205, "p95_ms": 0.021},
            "compile_ms": 100.0,
            "best_config": best_config,
            "search_space": [slower_config, best_config],
            "candidates": [
                {"config": slower_config, "status": "succeeded", "correctness": True, "compile_ms": 80.0, "summary": {"median_ms": 0.03}},
                {"config": best_config, "status": "succeeded", "correctness": True, "compile_ms": 100.0, "summary": {"median_ms": 0.0205}},
            ],
            "exit_code": 0,
            "artifact_upload": {
                "name": "autotune.json",
                "kind": "autotune-manifest",
                "media_type": "application/json",
                "content_base64": base64.b64encode(b"autotune-manifest").decode("ascii"),
            },
        }
        completed = self.client.request(
            "POST",
            f"/api/v1/tasks/{task['id']}/complete",
            {"worker_id": "autotune-worker", "runtime_version": __version__, "status": "succeeded", "result": result},
        )
        digest = completed["result"]["artifact"]["digest"]
        tuning_runs = self.client.request("GET", "/api/v1/tuning-runs?operator=matmul")["tuning_runs"]
        self.assertEqual(tuning_runs[0]["best_config"], best_config)
        self.assertEqual(len(tuning_runs[0]["candidates"]), 2)
        self.assertEqual(tuning_runs[0]["artifact_digest"], digest)
        kernel = self.client.request("GET", "/api/v1/kernels?operator=matmul")["kernels"][0]
        self.assertEqual(kernel["metadata"]["tuning_config"], best_config)
        self.assertEqual(kernel["artifact_digest"], digest)
        pipeline_task = self.client.request(
            "POST",
            "/api/v1/tasks",
            {
                "kind": "kernel_pipeline",
                "payload": {
                    "operator": {"name": "matmul", "shape": [128, 128, 128], "dtype": "fp16"},
                    "kernel_id": kernel_id,
                },
                "requirements": {"worker_ids": ["autotune-worker"], "kernel_id": kernel_id},
            },
        )
        self.assertEqual(pipeline_task["payload"]["kernel"]["metadata"]["tuning_config"], best_config)
        self.assertEqual(pipeline_task["payload"]["operator"]["backend"], "triton")
        benchmarks = self.client.request("GET", "/api/v1/benchmarks?operator=matmul")["benchmarks"]
        self.assertEqual(benchmarks[0]["summary"]["median_ms"], 0.0205)
        event_types = {item["event_type"] for item in self.client.request("GET", f"/api/v1/events?version={__version__}")["events"]}
        self.assertTrue({"autotune.candidate", "autotune.completed", "kernel.tuned"}.issubset(event_types))

    def test_compiler_run_persists_explainable_decision(self):
        for worker_id, architecture, load in (
            ("planner-x86", "x86_64", 0.5),
            ("planner-arm", "aarch64", 0.1),
        ):
            self.client.request(
                "POST",
                "/api/v1/workers/register",
                {
                    "id": worker_id,
                    "hostname": worker_id,
                    "capabilities": {
                        "architecture": architecture,
                        "accelerators": [],
                        "cpu_count": 4,
                        "memory_total_mb": 16000,
                        "hardware_fingerprint": f"{worker_id}-fingerprint",
                    },
                    "metrics": {"load_1m": load, "memory_available_mb": 12000},
                    "labels": {},
                    "version": __version__,
                },
            )
        kernel_id = "kernel-planner-softmax-v1"
        self.client.request(
            "POST",
            "/api/v1/kernels",
            {
                "id": kernel_id,
                "operator": "softmax",
                "backend": "python-reference",
                "version": "1",
                "architectures": ["x86_64", "aarch64"],
                "dtypes": ["fp32"],
            },
        )
        plan_request = {
            "operator": {"name": "softmax", "shape": [16], "dtype": "fp32"},
            "requirements": {"kernel_ids": [kernel_id]},
            "policy": {"unknown_latency_ms": 10.0, "load_weight_ms": 10.0},
        }
        dry_run = self.client.request("POST", "/api/v1/plans", plan_request)
        self.assertEqual(dry_run["selected"]["worker_id"], "planner-arm")
        task = self.client.request(
            "POST",
            "/api/v1/tasks",
            {
                "kind": "compiler_run",
                "payload": {"operator": plan_request["operator"], "policy": plan_request["policy"], "repeats": 2},
                "requirements": plan_request["requirements"],
            },
        )
        self.assertEqual(task["requirements"]["worker_ids"], ["planner-arm"])
        self.assertEqual(task["payload"]["kernel_id"], kernel_id)
        self.assertEqual(task["payload"]["schedule_decision"]["selected"], dry_run["selected"])
        decisions = self.client.request("GET", "/api/v1/schedule-decisions")["schedule_decisions"]
        self.assertEqual(decisions[0]["task_id"], task["id"])
        self.assertEqual(decisions[0]["selected"]["worker_id"], "planner-arm")
        leased = self.client.request("POST", "/api/v1/workers/planner-arm/lease", {})["task"]
        worker = Worker(
            self.client,
            "planner-arm",
            {},
            Path(self.temporary.name) / "planner-work",
            1,
            {"true"},
            False,
        )
        result = worker.execute(leased)
        completed = self.client.request(
            "POST",
            f"/api/v1/tasks/{task['id']}/complete",
            {"worker_id": "planner-arm", "runtime_version": __version__, "status": "succeeded", "result": result},
        )
        self.assertTrue(completed["result"]["correctness"])
        benchmarks = self.client.request("GET", "/api/v1/benchmarks?operator=softmax")["benchmarks"]
        self.assertEqual(benchmarks[0]["worker_id"], "planner-arm")
        self.assertEqual(benchmarks[0]["kernel_id"], kernel_id)
        event_types = {item["event_type"] for item in self.client.request("GET", f"/api/v1/events?version={__version__}")["events"]}
        self.assertIn("scheduler.decision", event_types)


if __name__ == "__main__":
    unittest.main()
