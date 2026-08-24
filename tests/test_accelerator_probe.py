import unittest

from edgeforge.accelerator_probe import (
    SCHEMA_VERSION,
    _RknnTensorAttr,
    _vulkan_version,
    run_accelerator_probe,
)


class AcceleratorProbeTests(unittest.TestCase):
    def test_schema_and_manual_contract_are_explicit(self):
        result = run_accelerator_probe(
            name="fixture",
            skip_opencl=True,
            skip_vulkan=True,
            skip_rknn=True,
        )
        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertEqual(result["toolchain_contract"]["board_api"], "RKNN C API")
        self.assertFalse(result["toolchain_contract"]["camera_used"])
        self.assertEqual(result["npu"]["status"], "skipped")

    def test_rknn_tensor_layout_matches_public_header_shape(self):
        self.assertGreaterEqual(_RknnTensorAttr.dims.offset, 8)
        self.assertGreater(_RknnTensorAttr.name.offset, _RknnTensorAttr.dims.offset)

    def test_vulkan_version_encoding(self):
        self.assertEqual(_vulkan_version((1 << 22) | (2 << 12) | 3), "1.2.3")


if __name__ == "__main__":
    unittest.main()

