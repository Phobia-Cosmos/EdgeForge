import unittest

from edgeforge.autotune import (
    default_triton_matmul_candidates,
    normalize_triton_matmul_candidates,
    select_best_candidate,
)


class AutoTuneTests(unittest.TestCase):
    def test_default_search_space_is_valid_and_unique(self):
        candidates = normalize_triton_matmul_candidates([])
        self.assertEqual(candidates, default_triton_matmul_candidates())
        self.assertEqual(len({tuple(item.items()) for item in candidates}), len(candidates))

    def test_invalid_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_triton_matmul_candidates([{"block_m": 7}])
        with self.assertRaises(ValueError):
            normalize_triton_matmul_candidates([{"num_warps": 3}])

    def test_best_candidate_requires_correctness_and_lowest_median(self):
        candidates = [
            {"status": "failed", "correctness": False, "compile_ms": 1.0, "summary": {"median_ms": 0.1}},
            {"status": "succeeded", "correctness": True, "compile_ms": 3.0, "summary": {"median_ms": 0.4, "p95_ms": 0.5}, "config": {"block_m": 32}},
            {"status": "succeeded", "correctness": True, "compile_ms": 4.0, "summary": {"median_ms": 0.2, "p95_ms": 0.3}, "config": {"block_m": 64}},
        ]
        self.assertEqual(select_best_candidate(candidates)["config"], {"block_m": 64})


if __name__ == "__main__":
    unittest.main()
