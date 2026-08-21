import tempfile
import unittest
import time
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
            "backends": ["python-reference", "torch-eager", "torch-compile"],
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

    def test_model_pipeline_persists_model_run(self):
        task = self.store.create_task(
            {
                "kind": "model_pipeline",
                "payload": {
                    "model": {"name": "tiny"},
                    "dataset": {"name": "synthetic", "manifest_digest": "sha256:data"},
                    "transforms": [{"name": "window", "version": "v1", "config": {"length": 8}}],
                    "frontend": {"name": "python"},
                    "compiler": {"backend": "python-reference", "identity": "test"},
                    "target": {"architecture": "aarch64"},
                },
                "requirements": {"worker_ids": ["arm"]},
            }
        )
        leased = self.store.lease_task("arm")
        self.assertEqual(leased["id"], task["id"])
        result = {
            "exit_code": 0,
            "manifest": leased["payload"],
            "correctness": True,
            "compile_ms": 2.5,
            "benchmark": {"steady_latency_ms": 3.1},
            "artifact": {"digest": "a" * 64},
        }
        self.store.complete_task(task["id"], "arm", {"status": "succeeded", "runtime_version": __version__, "result": result})
        runs = self.store.list_model_runs("tiny")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["transform_digest"], leased["payload"]["transform_digest"])
        self.assertEqual(runs[0]["compiler_backend"], "python-reference")

    def test_model_run_prefers_adapter_compile_time(self):
        task = self.store.create_task(
            {
                "kind": "model_pipeline",
                "payload": {
                    "model": {"name": "compile-time-model"},
                    "dataset": {"name": "synthetic"},
                    "compiler": {"backend": "python-reference", "identity": "test"},
                    "target": {"architecture": "aarch64"},
                },
                "requirements": {"worker_ids": ["arm"]},
            }
        )
        leased = self.store.lease_task("arm")
        self.store.complete_task(
            task["id"], "arm", {"status": "succeeded", "runtime_version": __version__, "result": {
                "manifest": leased["payload"], "correctness": True, "compile_ms": 99.0,
                "benchmark": {"compile_ms": 3.5, "steady_latency_ms": 1.0},
            }}
        )
        self.assertEqual(self.store.list_model_runs("compile-time-model")[0]["compile_ms"], 3.5)

    def test_model_regressions_compare_only_matching_backend_and_target(self):
        def complete(model, backend, target, latency, compile_ms, status="succeeded", correctness=True):
            worker_id = "x86" if target.get("architecture") == "x86_64" else "arm"
            task = self.store.create_task(
                {
                    "kind": "model_pipeline",
                    "payload": {
                        "model": {"name": model},
                        "dataset": {"name": "synthetic", "manifest_digest": "sha256:data"},
                        "compiler": {"backend": backend, "identity": "test"},
                        "target": target,
                    },
                    "requirements": {"worker_ids": [worker_id]},
                }
            )
            leased = self.store.lease_task(worker_id)
            result = {
                "exit_code": 0 if status == "succeeded" else 1,
                "manifest": leased["payload"],
                "correctness": correctness,
                "compile_ms": compile_ms,
                "benchmark": {"steady_latency_ms": latency},
            }
            self.store.complete_task(task["id"], worker_id, {"status": status, "runtime_version": __version__, "result": result})
            time.sleep(0.002)

        complete("regression-model", "torch-eager", {"architecture": "aarch64"}, 1.0, 2.0)
        complete("regression-model", "torch-eager", {"architecture": "aarch64"}, 2.0, 5.0)
        complete("regression-model", "torch-compile", {"architecture": "aarch64"}, 20.0, 50.0)
        complete("regression-model", "torch-eager", {"architecture": "x86_64"}, 20.0, 50.0)
        regressions = self.store.model_regressions("regression-model", threshold=0.2)
        self.assertEqual({item["kind"] for item in regressions}, {"steady_latency", "compile_time"})
        self.assertTrue(all(item["backend"] == "torch-eager" and item["architecture"] == "aarch64" for item in regressions))

    def test_failed_model_run_is_recorded_but_not_used_as_baseline(self):
        task = self.store.create_task(
            {
                "kind": "model_pipeline",
                "payload": {
                    "model": {"name": "failed-model"},
                    "dataset": {"name": "synthetic"},
                    "compiler": {"backend": "python-reference", "identity": "test"},
                    "target": {"architecture": "aarch64"},
                },
                "requirements": {"worker_ids": ["arm"]},
            }
        )
        self.store.lease_task("arm")
        self.store.complete_task(task["id"], "arm", {"status": "failed", "result": {"manifest": task["payload"], "correctness": False}})
        runs = self.store.list_model_runs("failed-model")
        self.assertEqual(runs[0]["task_status"], "failed")
        self.assertEqual(self.store.model_regressions("failed-model"), [])

    def test_model_pipeline_rejects_iree_without_explicit_device(self):
        with self.assertRaisesRegex(ValueError, "requires explicit target"):
            self.store.create_task(
                {
                    "kind": "model_pipeline",
                    "payload": {
                        "model": {"name": "tiny"},
                        "dataset": {"name": "synthetic"},
                        "compiler": {"backend": "iree"},
                    },
                }
            )

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

    def test_non_reference_backend_requires_kernel_pipeline(self):
        kernel = self.store.register_kernel(
            {
                "id": "kernel-iree-conv-store-v1",
                "operator": "conv_nchwc",
                "backend": "iree-ukernel",
                "version": "issue-24760-adapter-scaffold",
                "architectures": ["aarch64"],
                "dtypes": ["fp32"],
            }
        )
        with self.assertRaisesRegex(ValueError, "use kernel_pipeline"):
            self.store.create_task(
                {
                    "kind": "operator_benchmark",
                    "payload": {
                        "operator": {
                            "name": "conv_nchwc",
                            "shape": [1, 1, 3, 5, 1, 3, 3, 16, 16],
                            "dtype": "fp32",
                        },
                        "kernel_id": kernel["id"],
                    },
                    "requirements": {"worker_ids": ["arm"], "kernel_id": kernel["id"]},
                }
            )

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
