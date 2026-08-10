import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
