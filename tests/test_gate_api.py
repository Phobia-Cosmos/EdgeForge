import tempfile
import threading
import unittest
from pathlib import Path

from edgeforge import __version__
from edgeforge.api import ControlServer
from edgeforge.client import Client
from edgeforge.db import Store


class CapabilityGateAPITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temporary.name) / "edgeforge.db")
        self.store.register_worker(
            {
                "id": "gate-api-worker",
                "hostname": "gate-api-worker",
                "capabilities": {
                    "architecture": "x86_64",
                    "accelerators": [],
                    "cpu_count": 4,
                    "memory_total_mb": 16000,
                },
                "metrics": {"load_1m": 0, "memory_available_mb": 12000},
                "labels": {},
                "version": __version__,
            }
        )
        self._record_experiment()
        self.server = ControlServer(("127.0.0.1", 0), self.store, "test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.client = Client(f"http://{host}:{port}", "test-token")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def _record_experiment(self):
        experiment_id = "api-gate-exp"
        spec = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "workload": "raeeg",
            "dataset": {"name": "ISRUC"},
            "model": {"name": "BrainUICL"},
            "protocol": "eeg-cl-v1-aligned",
            "method": "brainuicl",
            "seed": 4321,
            "runner": {"mode": "import", "result_path": "metrics.json", "adapter": "raeeg-metrics-v1"},
            "metadata": {"comparison_group": "aligned-full49"},
        }
        task = self.store.create_task({"kind": "experiment_run", "payload": {"spec": spec}})
        self.store.lease_task("gate-api-worker")
        self.store.complete_task(
            task["id"],
            "gate-api-worker",
            {
                "status": "succeeded",
                "runtime_version": __version__,
                "result": {
                    "experiment_id": experiment_id,
                    "workload": "raeeg",
                    "experiment_bundle": {
                        "experiment_id": experiment_id,
                        "summary": {"final_old_acc": 0.73},
                        "metrics": [],
                        "source_result": {"digest": "b" * 64},
                    },
                },
            },
        )

    def test_registry_gate_and_explicit_promotion_routes(self):
        model = self.client.request(
            "POST",
            "/api/v1/models",
            {
                "id": "api-brainuicl",
                "name": "BrainUICL",
                "workload": "raeeg",
                "protocol": "eeg-cl-v1-aligned",
                "comparison_group": "aligned-full49",
                "source_experiment_id": "api-gate-exp",
                "descriptor": {"source": "api-test"},
            },
        )
        self.assertEqual(model["status"], "candidate")
        policy = self.client.request(
            "POST",
            "/api/v1/gate-policies",
            {
                "id": "api-policy-v1",
                "version": "1",
                "workload": "raeeg",
                "protocol": "eeg-cl-v1-aligned",
                "comparison_group": "aligned-full49",
                "rules": [{"metric": "summary.final_old_acc", "operator": ">=", "threshold": 0.70}],
            },
        )
        evaluation = self.client.request(
            "POST",
            "/api/v1/gate-evaluations",
            {"model_id": model["id"], "policy_id": policy["id"]},
        )
        self.assertEqual(evaluation["status"], "PASS")
        accepted = self.client.request("GET", "/api/v1/models?status=accepted")["models"]
        self.assertEqual([item["id"] for item in accepted], ["api-brainuicl"])

        promoted = self.client.request(
            "POST",
            "/api/v1/models/api-brainuicl/promote",
            {"reason": "API operator approval"},
        )
        self.assertEqual(promoted["status"], "production")
        evaluations = self.client.request(
            "GET", "/api/v1/gate-evaluations?model_id=api-brainuicl"
        )["gate_evaluations"]
        self.assertEqual(evaluations[0]["policy_snapshot"]["id"], "api-policy-v1")


if __name__ == "__main__":
    unittest.main()
