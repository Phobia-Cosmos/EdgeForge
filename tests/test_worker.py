import tempfile
import unittest
import json
from pathlib import Path

from edgeforge.client import Client
from edgeforge.worker import Worker


class WorkerExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.worker = Worker(
            client=Client("http://127.0.0.1:1", "test"),
            worker_id="test-worker",
            labels={},
            work_root=self.root,
            interval=1,
            allowed_commands={"python3", "true"},
            allow_any_command=False,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_allowed_command_runs_without_shell(self):
        result = self.worker.execute(
            {
                "kind": "command",
                "payload": {"argv": ["python3", "-c", "print('ok')"]},
            }
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"].strip(), "ok")

    def test_benchmark_collects_each_timing(self):
        result = self.worker.execute(
            {"kind": "benchmark", "payload": {"argv": ["true"], "repeats": 3}}
        )
        self.assertEqual(result["summary"]["runs"], 3)
        self.assertEqual(len(result["timings_ms"]), 3)

    def test_unlisted_command_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "not allowed"):
            self.worker.execute({"kind": "command", "payload": {"argv": ["uname"]}})

    def test_allowed_virtualenv_symlink_is_preserved(self):
        target = self.root / "python-target"
        target.write_text("placeholder", encoding="utf-8")
        launcher = self.root / "venv-python"
        launcher.symlink_to(target)
        self.worker.allowed_commands.add(str(launcher))
        self.assertEqual(self.worker._resolve_command(str(launcher)), str(launcher))

    def test_cwd_cannot_escape_work_root(self):
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            self.worker.execute(
                {"kind": "command", "payload": {"argv": ["true"], "cwd": "../outside"}}
            )

    def test_operator_benchmark_does_not_require_command_permission(self):
        result = self.worker.execute(
            {
                "kind": "operator_benchmark",
                "payload": {
                    "operator": {"name": "matmul", "shape": [2, 3, 4], "dtype": "fp32"},
                    "repeats": 2,
                },
            }
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["correctness"])
        self.assertEqual(result["summary"]["runs"], 2)

    def test_operator_result_preserves_kernel_identity(self):
        result = self.worker.execute(
            {
                "kind": "operator_benchmark",
                "payload": {
                    "operator": {"name": "softmax", "shape": [8], "dtype": "fp32"},
                    "kernel": {"id": "kernel-softmax-v1", "version": "1"},
                },
            }
        )
        self.assertEqual(result["kernel_id"], "kernel-softmax-v1")
        self.assertEqual(result["kernel_version"], "1")

    def test_experiment_import_builds_normalized_bundle(self):
        result_dir = self.root / "results"
        result_dir.mkdir()
        (result_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "summary": {"final_old_acc": 0.71},
                    "tasks": [
                        {
                            "task": 1,
                            "subject": 64,
                            "current_before": {"acc": 0.60},
                            "current_after": {"acc": 0.65},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        result = self.worker.execute(
            {
                "kind": "experiment_run",
                "payload": {
                    "spec": {
                        "schema_version": 1,
                        "experiment_id": "worker-import-test",
                        "workload": "raeeg",
                        "dataset": {"name": "ISRUC"},
                        "model": {"name": "BrainUICL"},
                        "protocol": "eeg-cl-v1",
                        "method": "finetune",
                        "seed": 4321,
                        "runner": {
                            "mode": "import",
                            "result_path": "results/metrics.json",
                            "adapter": "raeeg-metrics-v1",
                        },
                    }
                },
            }
        )
        bundle = result["experiment_bundle"]
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(bundle["experiment_id"], "worker-import-test")
        self.assertTrue(any(item["name"] == "plasticity.acc_gain" for item in bundle["metrics"]))
        self.assertEqual(result["artifact_upload"]["kind"], "experiment-bundle")

    def test_model_pipeline_runs_external_stages_and_records_digests(self):
        result = self.worker.execute(
            {
                "kind": "model_pipeline",
                "payload": {
                    "model": {"name": "tiny", "format": "pytorch"},
                    "dataset": {"name": "synthetic", "manifest_digest": "sha256:test"},
                    "transforms": [{"name": "normalize", "version": "v1", "config": {"mean": 0.0}}],
                    "frontend": {"name": "pytorch", "command": ["python3", "-c", "print('{}')"]},
                    "compiler": {"backend": "torch-compile", "identity": "test", "command": ["python3", "-c", "print('{}')"]},
                    "runtime": {"command": ["python3", "-c", "print('{}')"]},
                    "correctness": {"command": ["python3", "-c", "print('{\"correctness\": true}')"]},
                    "benchmark": {"command": ["python3", "-c", "print('{\"steady_latency_ms\": 1.2}')"], "repeats": 1},
                    "target": {"architecture": "x86_64"},
                },
            }
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["correctness"])
        self.assertEqual(len(result["pipeline"]), 6)
        self.assertEqual(result["manifest"]["compiler"]["backend"], "torch-compile")
        self.assertEqual(len(result["manifest"]["transform_digest"]), 64)
        self.assertEqual(result["artifact_upload"]["kind"], "model-compiler-manifest")

    def test_model_pipeline_rejects_unlisted_stage_command(self):
        with self.assertRaisesRegex(RuntimeError, "not allowed"):
            self.worker.execute(
                {
                    "kind": "model_pipeline",
                    "payload": {
                        "model": {"name": "tiny"},
                        "dataset": {"name": "synthetic"},
                        "frontend": {"command": ["uname", "-m"]},
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
