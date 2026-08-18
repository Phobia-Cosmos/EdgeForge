import unittest

from edgeforge.operator import OperatorSpec, benchmark_operator


class OperatorTests(unittest.TestCase):
    def test_supported_operators_have_correct_reference_results(self):
        specs = (
            {"name": "matmul", "shape": [2, 3, 4]},
            {"name": "softmax", "shape": [16]},
            {"name": "rmsnorm", "shape": [16]},
            {"name": "silu", "shape": [16]},
            {
                "name": "conv_nchwc",
                "shape": [1, 1, 3, 5, 1, 3, 3, 16, 16],
                "attrs": {"dilation_h": 2, "dilation_w": 3},
            },
        )
        for payload in specs:
            result = benchmark_operator(OperatorSpec.from_payload(payload), repeats=2)
            self.assertTrue(result["correctness"], payload["name"])
            self.assertEqual(len(result["timings_ms"]), 2)
            self.assertEqual(len(result["checksum"]), 64)

    def test_shape_and_dtype_are_validated(self):
        with self.assertRaises(ValueError):
            OperatorSpec.from_payload({"name": "matmul", "shape": [2, 3]})
        with self.assertRaises(ValueError):
            OperatorSpec.from_payload({"name": "softmax", "shape": [4], "dtype": "int4"})

    def test_small_sample_p95_uses_the_upper_observation(self):
        result = benchmark_operator(OperatorSpec.from_payload({"name": "softmax", "shape": [8]}), repeats=2)
        self.assertEqual(result["summary"]["p95_ms"], max(result["timings_ms"]))


if __name__ == "__main__":
    unittest.main()
