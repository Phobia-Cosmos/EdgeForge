import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "audit-eeg-condition-quality.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_audit_eeg_condition_quality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuditEEGConditionQualityTests(unittest.TestCase):
    def test_identical_derived_condition_passes_quality_gate(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean"
            derived = root / "derived"
            for base in (clean, derived):
                data = base / "source" / "1" / "data"
                label = base / "source" / "1" / "label"
                data.mkdir(parents=True)
                label.mkdir(parents=True)
                values = np.random.default_rng(0).normal(size=(2, 8, 2048)).astype(np.float32)
                np.save(data / "0.npy", values, allow_pickle=False)
                np.save(label / "0.npy", np.asarray([0, 1], dtype=np.int64), allow_pickle=False)
            # The audit expects all three groups, so add empty subject roots.
            for base in (clean, derived):
                for group in ("target", "retention"):
                    (base / group).mkdir(parents=True)
            result = module.audit(clean, derived, root / "out")
            self.assertTrue(result["labels_byte_identical"])
            self.assertTrue(result["preferred_waveform_gate_pass"])
            self.assertAlmostEqual(result["correlation_min"], 1.0, places=6)
            self.assertEqual(result["changed_data_files"], 0)
            self.assertIn("source", result["groups"])


if __name__ == "__main__":
    unittest.main()
