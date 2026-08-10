import unittest

from edgeforge.scheduler import select_worker, worker_score


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.x86 = {
            "id": "worker-4070s",
            "capabilities": {
                "architecture": "x86_64",
                "accelerators": ["nvidia-gpu"],
                "cpu_count": 16,
                "memory_total_mb": 32000,
            },
            "metrics": {"load_1m": 4.0, "memory_available_mb": 16000},
            "labels": {"role": "gpu"},
            "active_tasks": 0,
        }
        self.arm = {
            "id": "worker-orangepi",
            "capabilities": {
                "architecture": "aarch64",
                "accelerators": ["rk3588-npu"],
                "cpu_count": 8,
                "memory_total_mb": 16000,
            },
            "metrics": {"load_1m": 1.0, "memory_available_mb": 12000},
            "labels": {"role": "edge-npu"},
            "active_tasks": 0,
        }

    def test_architecture_is_a_hard_constraint(self):
        selected = select_worker([self.x86, self.arm], {"architectures": ["aarch64"]})
        self.assertEqual(selected["id"], "worker-orangepi")

    def test_accelerator_is_a_hard_constraint(self):
        selected = select_worker([self.x86, self.arm], {"accelerators": ["nvidia-gpu"]})
        self.assertEqual(selected["id"], "worker-4070s")

    def test_preference_changes_score_without_filtering(self):
        requirements = {"prefer_accelerators": ["nvidia-gpu"]}
        self.assertGreater(worker_score(self.x86, requirements), worker_score(self.arm, requirements))

    def test_labels_and_memory_filter(self):
        selected = select_worker(
            [self.x86, self.arm],
            {"labels": {"role": "edge-npu"}, "min_memory_mb": 10000},
        )
        self.assertEqual(selected["id"], "worker-orangepi")


if __name__ == "__main__":
    unittest.main()

