import unittest

from edgeforge.backend_registry import backend_capabilities, validate_backend_target


class BackendRegistryTests(unittest.TestCase):
    def test_registry_lists_explicit_target_contracts(self):
        names = {item["name"] for item in backend_capabilities()}
        self.assertTrue({"python-reference", "onnx-runtime", "iree", "rknn"}.issubset(names))

    def test_iree_requires_explicit_architecture_and_device(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_backend_target("iree", {})
        with self.assertRaisesRegex(ValueError, "device"):
            validate_backend_target("iree", {"architecture": "aarch64"})
        spec = validate_backend_target("iree", {"architecture": "aarch64", "device": "cpu"})
        self.assertEqual(spec["name"], "iree")

    def test_rknn_target_must_be_arm_npu(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_backend_target("rknn", {"architecture": "x86_64"})
        with self.assertRaisesRegex(ValueError, "accelerator"):
            validate_backend_target("rknn", {"architecture": "aarch64"})
        with self.assertRaisesRegex(ValueError, "accelerator"):
            validate_backend_target("rknn", {"architecture": "aarch64", "accelerator": "drm"})
        validate_backend_target("rknn", {"architecture": "aarch64", "accelerator": "rk3588-npu"})

    def test_custom_backend_requires_target(self):
        with self.assertRaisesRegex(ValueError, "target.architecture"):
            validate_backend_target("my-runtime", {})
        spec = validate_backend_target("my-runtime", {"architecture": "riscv64"})
        self.assertEqual(spec["kind"], "custom")
