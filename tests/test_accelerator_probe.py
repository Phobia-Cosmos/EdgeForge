import unittest
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

from edgeforge.accelerator_probe import (
    SCHEMA_VERSION,
    _RknnTensorAttr,
    _temporary_environment,
    _vulkan_manifest_evidence,
    _vulkan_version,
    probe_vulkan,
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

    def test_vulkan_manifest_evidence_is_digest_bound_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "libcandidate.so"
            library.write_bytes(b"candidate")
            manifest = root / "candidate.json"
            manifest.write_text(
                json.dumps(
                    {
                        "file_format_version": "1.0.0",
                        "ICD": {"library_path": "libcandidate.so", "api_version": "1.3.276"},
                    }
                ),
                encoding="utf-8",
            )
            evidence = _vulkan_manifest_evidence(manifest)
            self.assertEqual(evidence["status"], "candidate")
            self.assertTrue(evidence["library_exists"])
            self.assertEqual(evidence["library_path"], str(library))
            self.assertRegex(evidence["sha256"], r"^[0-9a-f]{64}$")

    def test_vulkan_environment_override_is_restored(self):
        with mock.patch.dict(os.environ, {"VK_ICD_FILENAMES": "original"}, clear=False):
            with _temporary_environment(VK_ICD_FILENAMES="temporary"):
                self.assertEqual(os.environ["VK_ICD_FILENAMES"], "temporary")
            self.assertEqual(os.environ["VK_ICD_FILENAMES"], "original")

    def test_vulkan_rejects_malformed_explicit_manifest_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "broken.json"
            manifest.write_text("{}\\nnot-json", encoding="utf-8")
            result = probe_vulkan(icd_manifest=manifest)
            self.assertEqual(result["status"], "blocked")
            self.assertIn("manifest", result["reason"])


if __name__ == "__main__":
    unittest.main()
