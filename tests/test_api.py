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


if __name__ == "__main__":
    unittest.main()
