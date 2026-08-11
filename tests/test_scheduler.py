import unittest

from edgeforge.scheduler import build_execution_plan, select_worker, worker_score


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
            "status": "online",
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
            "status": "online",
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

    def test_compiler_plan_prefers_measured_lower_latency(self):
        kernel = {
            "id": "kernel-reference-matmul-v1",
            "operator": "matmul",
            "backend": "python-reference",
            "architectures": ["x86_64", "aarch64"],
            "accelerators": [],
            "dtypes": ["fp32"],
            "status": "active",
        }
        operator = {"name": "matmul", "shape": [32, 32, 32], "dtype": "fp32"}
        benchmarks = [
            {
                "worker_id": "worker-4070s",
                "architecture": "x86_64",
                "operator": "matmul",
                "shape": [32, 32, 32],
                "dtype": "fp32",
                "kernel_id": kernel["id"],
                "correctness": True,
                "compile_ms": 0.1,
                "summary": {"median_ms": 5.0},
            },
            {
                "worker_id": "worker-orangepi",
                "architecture": "aarch64",
                "operator": "matmul",
                "shape": [32, 32, 32],
                "dtype": "fp32",
                "kernel_id": kernel["id"],
                "correctness": True,
                "compile_ms": 0.1,
                "summary": {"median_ms": 2.0},
            },
        ]
        plan = build_execution_plan(
            [self.x86, self.arm],
            [kernel],
            benchmarks,
            operator,
            policy={"load_weight_ms": 0.0},
        )
        self.assertEqual(plan["selected"]["worker_id"], "worker-orangepi")
        self.assertEqual(plan["selected"]["estimate_source"], "exact-worker")

    def test_compiler_plan_uses_load_for_unseen_paths(self):
        kernel = {
            "id": "kernel-reference-softmax-v1",
            "operator": "softmax",
            "backend": "python-reference",
            "architectures": ["x86_64", "aarch64"],
            "accelerators": [],
            "dtypes": ["fp32"],
            "status": "active",
        }
        plan = build_execution_plan(
            [self.x86, self.arm],
            [kernel],
            [],
            {"name": "softmax", "shape": [1024], "dtype": "fp32"},
            policy={"unknown_latency_ms": 100.0, "load_weight_ms": 10.0},
        )
        self.assertEqual(plan["selected"]["worker_id"], "worker-orangepi")
        self.assertEqual(plan["selected"]["estimate_source"], "unseen")

    def test_compiler_plan_respects_backend_filter(self):
        kernels = [
            {
                "id": "kernel-reference-matmul-v1",
                "operator": "matmul",
                "backend": "python-reference",
                "architectures": ["*"],
                "accelerators": [],
                "dtypes": ["fp16"],
                "status": "active",
            },
            {
                "id": "kernel-triton-matmul-v1",
                "operator": "matmul",
                "backend": "triton",
                "architectures": ["x86_64"],
                "accelerators": ["nvidia-gpu"],
                "dtypes": ["fp16"],
                "status": "active",
            },
        ]
        plan = build_execution_plan(
            [self.x86, self.arm],
            kernels,
            [],
            {"name": "matmul", "shape": [64, 64, 64], "dtype": "fp16"},
            requirements={"backends": ["triton"]},
        )
        self.assertEqual(plan["selected"]["kernel_id"], "kernel-triton-matmul-v1")
        self.assertEqual(len(plan["candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
