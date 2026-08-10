import tempfile
import unittest
from pathlib import Path

from edgeforge import __version__
from edgeforge.db import Store


def worker(worker_id, architecture, load=0.0):
    return {
        "id": worker_id,
        "hostname": worker_id,
        "capabilities": {
            "architecture": architecture,
            "accelerators": [],
            "cpu_count": 4,
            "memory_total_mb": 16000,
        },
        "metrics": {"load_1m": load, "memory_available_mb": 12000},
        "labels": {},
        "version": "test",
    }


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temporary.name) / "edgeforge.db")
        self.store.register_worker(worker("x86", "x86_64"))
        self.store.register_worker(worker("arm", "aarch64"))

    def tearDown(self):
        self.temporary.cleanup()

    def test_task_runs_only_on_matching_architecture(self):
        task = self.store.create_task(
            {
                "kind": "command",
                "payload": {"argv": ["uname", "-m"]},
                "requirements": {"architectures": ["aarch64"]},
            }
        )
        self.assertIsNone(self.store.lease_task("x86"))
        leased = self.store.lease_task("arm")
        self.assertEqual(leased["id"], task["id"])
        self.assertEqual(leased["status"], "running")

    def test_worker_cannot_lease_second_task_while_busy(self):
        first = self.store.create_task({"payload": {"argv": ["true"]}})
        self.store.create_task({"payload": {"argv": ["true"]}})
        leased = self.store.lease_task("arm")
        if leased is None:
            leased = self.store.lease_task("x86")
            worker_id = "x86"
        else:
            worker_id = "arm"
        self.assertEqual(leased["id"], first["id"])
        self.assertIsNone(self.store.lease_task(worker_id))

    def test_only_assigned_worker_can_complete(self):
        task = self.store.create_task(
            {
                "payload": {"argv": ["true"]},
                "requirements": {"worker_ids": ["arm"]},
            }
        )
        self.store.lease_task("arm")
        with self.assertRaises(RuntimeError):
            self.store.complete_task(task["id"], "x86", {"status": "succeeded"})
        completed = self.store.complete_task(
            task["id"], "arm", {"status": "succeeded", "result": {"exit_code": 0}}
        )
        self.assertEqual(completed["status"], "succeeded")

    def test_invalid_task_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.create_task({"payload": {"argv": []}})
        with self.assertRaises(ValueError):
            self.store.create_task({"kind": "shell", "payload": {"argv": ["true"]}})

    def test_release_update_preserves_original_timestamp(self):
        first = self.store.record_release("test-version", "first", {"step": 1}, status="active")
        second = self.store.record_release("test-version", "updated", {"step": 2}, status="retired")
        self.assertEqual(first["created_at"], second["created_at"])
        self.assertEqual(second["status"], "retired")
        self.assertEqual(second["metadata"], {"step": 2})

    def test_operator_benchmark_is_persisted_separately(self):
        task = self.store.create_task(
            {
                "kind": "operator_benchmark",
                "payload": {"operator": {"name": "softmax", "shape": [8], "dtype": "fp32"}, "repeats": 2},
                "requirements": {"worker_ids": ["arm"]},
            }
        )
        leased = self.store.lease_task("arm")
        self.assertEqual(leased["id"], task["id"])
        result = {
            "operator": {"name": "softmax", "shape": [8], "dtype": "fp32", "backend": "python-reference"},
            "backend": "python-reference",
            "correctness": True,
            "timings_ms": [1.0, 1.2],
            "summary": {"runs": 2, "median_ms": 1.1},
        }
        self.store.complete_task(
            task["id"], "arm", {"status": "succeeded", "runtime_version": __version__, "result": result}
        )
        benchmarks = self.store.list_benchmarks("softmax")
        self.assertEqual(len(benchmarks), 1)
        self.assertEqual(benchmarks[0]["version"], __version__)
        self.assertTrue(benchmarks[0]["correctness"])
        events = self.store.list_events(version=__version__)
        self.assertTrue(any(event["event_type"] == "task.completed" for event in events))

    def test_kernel_registry_filters_architecture_and_detects_regression(self):
        kernel = self.store.register_kernel(
            {
                "id": "kernel-softmax-arm-v1",
                "operator": "softmax",
                "backend": "python-reference",
                "version": "1",
                "architectures": ["aarch64"],
                "dtypes": ["fp32"],
            }
        )
        first = self.store.create_task(
            {
                "kind": "operator_benchmark",
                "payload": {"operator": {"name": "softmax", "shape": [8], "dtype": "fp32"}, "kernel_id": kernel["id"]},
                "requirements": {"worker_ids": ["arm"]},
            }
        )
        self.assertIsNone(self.store.lease_task("x86"))
        self.assertIsNotNone(self.store.lease_task("arm"))
        self.store.complete_task(
            first["id"],
            "arm",
            {
                "status": "succeeded",
                "runtime_version": __version__,
                "result": {
                    "operator": {"name": "softmax", "shape": [8], "dtype": "fp32"},
                    "backend": "python-reference",
                    "kernel_id": kernel["id"],
                    "kernel_version": "1",
                    "correctness": True,
                    "timings_ms": [1.0],
                    "summary": {"median_ms": 1.0},
                },
            },
        )
        second = self.store.create_task(
            {
                "kind": "operator_benchmark",
                "payload": {"operator": {"name": "softmax", "shape": [8], "dtype": "fp32"}, "kernel_id": kernel["id"]},
                "requirements": {"worker_ids": ["arm"]},
            }
        )
        self.assertIsNotNone(self.store.lease_task("arm"))
        self.store.complete_task(
            second["id"],
            "arm",
            {
                "status": "succeeded",
                "runtime_version": __version__,
                "result": {
                    "operator": {"name": "softmax", "shape": [8], "dtype": "fp32"},
                    "backend": "python-reference",
                    "kernel_id": kernel["id"],
                    "kernel_version": "1",
                    "correctness": True,
                    "timings_ms": [2.5],
                    "summary": {"median_ms": 2.5},
                },
            },
        )
        regressions = self.store.performance_regressions("softmax", threshold=0.2)
        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0]["kernel_id"], kernel["id"])
        self.assertEqual(regressions[0]["latest"]["kernel_version"], "1")


if __name__ == "__main__":
    unittest.main()
