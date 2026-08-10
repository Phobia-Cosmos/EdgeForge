import tempfile
import threading
import unittest
from pathlib import Path

from edgeforge.api import ControlServer
from edgeforge.client import APIError, Client
from edgeforge.db import Store


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

        events = self.client.request("GET", "/api/v1/events?version=0.2.0")["events"]
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


if __name__ == "__main__":
    unittest.main()
