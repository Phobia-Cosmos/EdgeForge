import unittest

from edgeforge.deployment_preflight import evaluate_preflight


class DeploymentPreflightTests(unittest.TestCase):
    def test_blocks_missing_torch_on_arm_target(self):
        result = evaluate_preflight(
            {"model": {"name": "brainuicl"}, "compiler": {"backend": "torch-eager"}, "target": {"architecture": "aarch64"}},
            {"name": "orangepi", "status": "online", "summary": {"architecture": "aarch64"}, "runtime_capabilities": {"torch_python": False}},
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("torch_python", result["missing_capabilities"])
        self.assertFalse(result["execution_performed"])

    def test_reference_target_can_pass_without_framework(self):
        result = evaluate_preflight(
            {"model": {"name": "reference"}, "compiler": {"backend": "python-reference"}, "target": {"architecture": "riscv64"}},
            {"name": "p550", "status": "online", "summary": {"architecture": "riscv64"}, "runtime_capabilities": {}},
        )
        self.assertEqual(result["status"], "PASS")

    def test_rknn_requires_recorded_board_api_smoke_without_python_toolkit(self):
        result = evaluate_preflight(
            {"model": {"name": "mobilenet"}, "compiler": {"backend": "rknn"}, "target": {"architecture": "aarch64"}},
            {
                "name": "orangepi",
                "status": "online",
                "summary": {"architecture": "aarch64"},
                "runtime_capabilities": {
                    "rknn_python": False,
                    "rknn_runtime_files": True,
                    "rk3588_npu_drm": True,
                    "rk3588_npu_device": False,
                },
            },
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertNotIn("rknn_python", result["required_capabilities"])

        validation = {
            "status": "partial",
            "npu": {
                "status": "pass",
                "init": {"code": 0},
                "sdk": {"code": 0},
                "io_query": {"code": 0},
                "inputs_set": {"code": 0},
                "destroy": {"code": 0},
                "deterministic_zero_input": True,
            },
        }
        passed = evaluate_preflight(
            {"model": {"name": "mobilenet"}, "compiler": {"backend": "rknn"}, "target": {"architecture": "aarch64"}},
            {
                "name": "orangepi",
                "status": "online",
                "summary": {"architecture": "aarch64"},
                "runtime_capabilities": {"rknn_runtime_files": True, "rk3588_npu_drm": True},
            },
            validation,
        )
        self.assertEqual(passed["status"], "PASS")

        mismatched = dict(validation)
        mismatched["name"] = "other-board"
        blocked = evaluate_preflight(
            {"model": {"name": "mobilenet"}, "compiler": {"backend": "rknn"}, "target": {"architecture": "aarch64"}},
            {
                "name": "orangepi",
                "status": "online",
                "summary": {"architecture": "aarch64"},
                "runtime_capabilities": {"rknn_runtime_files": True, "rk3588_npu_drm": True},
            },
            mismatched,
        )
        self.assertEqual(blocked["status"], "BLOCKED")

    def test_rknn_blocks_when_runtime_exists_but_no_npu_evidence(self):
        result = evaluate_preflight(
            {"model": {"name": "mobilenet"}, "compiler": {"backend": "rknn"}, "target": {"architecture": "aarch64"}},
            {
                "name": "orangepi",
                "status": "online",
                "summary": {"architecture": "aarch64"},
                "runtime_capabilities": {"rknn_runtime_files": True},
            },
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(result["missing_alternative_capabilities"])


if __name__ == "__main__":
    unittest.main()
