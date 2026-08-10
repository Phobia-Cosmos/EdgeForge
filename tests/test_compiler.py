import unittest

from edgeforge.compiler import run_kernel_pipeline


class CompilerPipelineTests(unittest.TestCase):
    def test_reference_pipeline_has_three_successful_stages(self):
        result = run_kernel_pipeline(
            {
                "operator": {"name": "softmax", "shape": [16], "dtype": "fp32"},
                "kernel": {"id": "kernel-reference-softmax-v1", "version": "1", "backend": "python-reference"},
                "repeats": 2,
            }
        )
        self.assertEqual([stage["name"] for stage in result["pipeline"]], ["compile", "correctness", "benchmark"])
        self.assertTrue(all(stage["status"] == "succeeded" for stage in result["pipeline"]))
        self.assertEqual(result["kernel_id"], "kernel-reference-softmax-v1")
        self.assertIn("content_base64", result["artifact_upload"])


if __name__ == "__main__":
    unittest.main()

