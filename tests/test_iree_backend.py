import sys
import tempfile
import unittest
from pathlib import Path

from edgeforge.compiler import run_kernel_pipeline
from edgeforge.operator import OperatorSpec


class IreeBackendTests(unittest.TestCase):
    def test_prebuilt_pipeline_runs_registered_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_iree_tool.py"
            script.write_text(
                "import sys\n"
                "if sys.argv[1] == 'test':\n"
                "    print('Summary: 3 tests run, 0 failed, 0 skipped.')\n"
                "else:\n"
                "    print('ARGS=' + ' '.join(sys.argv[2:]))\n"
                "    print('BM_conv_nchwc_f32f32f32_tile_16x16/real_time 12.5 us 12.0 us 1 items_per_second=2.0G/s')\n",
                encoding="utf-8",
            )
            kernel = {
                "id": "kernel-iree-conv-test-v1",
                "version": "issue-24760-dilated-conv",
                "backend": "iree-ukernel",
                "metadata": {
                    "trusted": True,
                    "test_command": [sys.executable, str(script), "test"],
                    "benchmark_command": [sys.executable, str(script), "benchmark"],
                    "workdir": str(root),
                    "iree_commit": "a82330a3",
                    "patch_sha256": "test-digest",
                },
            }
            spec = OperatorSpec.from_payload(
                {
                    "name": "conv_nchwc",
                    "shape": [1, 1, 3, 5, 1, 3, 3, 16, 16],
                    "dtype": "fp32",
                    "attrs": {"dilation_h": 2, "dilation_w": 3},
                }
            )
            result = run_kernel_pipeline(
                {"operator": spec.to_dict(), "kernel": kernel, "repeats": 2}
            )
            self.assertEqual([stage["name"] for stage in result["pipeline"]], ["compile", "correctness", "benchmark"])
            self.assertTrue(all(stage["status"] == "succeeded" for stage in result["pipeline"]))
            self.assertTrue(result["correctness"])
            self.assertEqual(result["summary"]["runs"], 2)
            self.assertEqual(result["throughput_items_per_second"], [2_000_000_000.0, 2_000_000_000.0])
            self.assertIn("content_base64", result["artifact_upload"])
            self.assertTrue(
                any(
                    "--dilation_h=2" in output and "--dilation_w=3" in output
                    for output in result["raw_outputs"]
                )
            )

    def test_iree_backend_enforces_worker_allow_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool = root / "iree-tool"
            tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            tool.chmod(0o755)
            spec = OperatorSpec.from_payload(
                {"name": "conv_nchwc", "shape": [1, 1, 1, 1, 1, 1, 1, 16, 16]}
            )
            kernel = {
                "id": "kernel-iree-allow-list-v1",
                "backend": "iree-ukernel",
                "metadata": {
                    "trusted": True,
                    "test_command": [str(tool)],
                    "benchmark_command": [str(tool)],
                    "workdir": str(root),
                },
            }
            with self.assertRaisesRegex(RuntimeError, "not allowed"):
                run_kernel_pipeline(
                    {
                        "operator": spec.to_dict(),
                        "kernel": kernel,
                        "_worker_work_root": str(root),
                        "_allowed_commands": [],
                    }
                )

    def test_iree_backend_requires_trusted_kernel_metadata(self):
        with self.assertRaisesRegex(ValueError, "trusted=true"):
            run_kernel_pipeline(
                {
                    "operator": {"name": "conv_nchwc", "shape": [1, 1, 1, 1, 1, 1, 1, 16, 16]},
                    "kernel": {"id": "untrusted", "backend": "iree-ukernel", "metadata": {}},
                }
            )


if __name__ == "__main__":
    unittest.main()
