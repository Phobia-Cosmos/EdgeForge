import tempfile
import threading
import unittest
from pathlib import Path

from edgeforge import __version__
from edgeforge.api import ControlServer
from edgeforge.client import Client
from edgeforge.db import Store


class ExperimentAPITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        store = Store(Path(self.temporary.name) / "edgeforge.db")
        self.server = ControlServer(("127.0.0.1", 0), store, "test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.client = Client(f"http://{host}:{port}", "test-token")
        self.client.request(
            "POST",
            "/api/v1/workers/register",
            {
                "id": "experiment-worker",
                "hostname": "experiment-worker",
                "capabilities": {
                    "architecture": "x86_64",
                    "accelerators": ["nvidia-gpu"],
                    "cpu_count": 4,
                    "memory_total_mb": 16000,
                },
                "metrics": {"load_1m": 0, "memory_available_mb": 12000},
                "labels": {},
                "version": __version__,
            },
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def test_experiment_completion_persists_bundle_and_metrics(self):
        spec = {
            "schema_version": 1,
            "experiment_id": "api-raeeg-test",
            "workload": "raeeg",
            "dataset": {"name": "ISRUC"},
            "model": {"name": "BrainUICL"},
            "protocol": "eeg-cl-v1",
            "method": "si",
            "seed": 4321,
            "runner": {"mode": "import", "result_path": "metrics.json", "adapter": "raeeg-metrics-v1"},
        }
        task = self.client.request(
            "POST",
            "/api/v1/tasks",
            {"kind": "experiment_run", "payload": {"spec": spec}, "requirements": {"worker_ids": ["experiment-worker"]}},
        )
        self.client.request("POST", "/api/v1/workers/experiment-worker/lease", {})
        bundle = {
            "schema_version": 1,
            "experiment_id": "api-raeeg-test",
            "workload": "raeeg",
            "spec": spec,
            "metrics": [
                {
                    "namespace": "raeeg.research",
                    "name": "plasticity.acc_gain",
                    "value": 0.05,
                    "step": 1,
                    "unit": "ratio",
                    "context": {"subject": "64"},
                }
            ],
            "summary": {"final_old_acc": 0.71},
            "source_result": {"digest": "a" * 64},
        }
        completed = self.client.request(
            "POST",
            f"/api/v1/tasks/{task['id']}/complete",
            {
                "worker_id": "experiment-worker",
                "runtime_version": __version__,
                "status": "succeeded",
                "result": {"exit_code": 0, "experiment_id": "api-raeeg-test", "workload": "raeeg", "experiment_bundle": bundle},
            },
        )
        self.assertEqual(completed["status"], "succeeded")
        experiments = self.client.request("GET", "/api/v1/experiments?workload=raeeg")["experiments"]
        self.assertEqual(experiments[0]["method"], "si")
        self.assertEqual(experiments[0]["summary"]["final_old_acc"], 0.71)
        metrics = self.client.request("GET", "/api/v1/experiment-metrics?experiment_id=api-raeeg-test")["metrics"]
        self.assertEqual(metrics[0]["context"], {"subject": "64"})
        events = self.client.request("GET", f"/api/v1/events?version={__version__}")["events"]
        self.assertTrue(any(item["event_type"] == "experiment.completed" for item in events))


if __name__ == "__main__":
    unittest.main()
