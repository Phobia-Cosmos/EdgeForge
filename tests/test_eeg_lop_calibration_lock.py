import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "create-eeg-lop-calibration-lock.py"
SPEC = importlib.util.spec_from_file_location("edgeforge_test_calibration_lock", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
lock_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lock_module)


class CalibrationLockTests(unittest.TestCase):
    def _write(self, path: Path, value: dict) -> Path:
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_lock_binds_inputs_and_freezes_formal_design(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write(root / "config.json", {
                "experiment_version": "test",
                "input_scaling": {"input_scale": 100000.0},
                "phases": {
                    "calibration": {
                        "architectures": ["tcn"], "seeds": [1], "orders": ["random-a"],
                        "target_limit": 1, "source_epochs": 2, "budgets": [0, 5],
                        "batch_size": 4, "source_lr": 0.002, "adapt_lr": 0.001,
                        "input_scale": 100000.0, "freeze_batch_norm": True,
                    },
                    "confirmatory": {
                        "architectures": ["tcn"], "seeds": [1, 2], "orders": ["random-a", "random-b"],
                        "target_limit": None, "source_epochs": 3, "budgets": [0, 5],
                        "batch_size": 4, "source_lr": 0.002, "adapt_lr": 0.001,
                        "input_scale": 100000.0, "freeze_batch_norm": True,
                    },
                },
            })
            plan = self._write(root / "plan.json", {"plan_digest": "p"})
            manifest = self._write(root / "manifest.json", {"phase": "calibration", "commands": [{"status": "succeeded"}]})
            analysis = self._write(root / "analysis.json", {
                "phase": "calibration", "status": "candidate-evidence-ready", "candidate_evidence_ready": True,
                "audit": {
                    "completeness_passed": True, "design_consistency_passed": True,
                    "learning_adequacy_passed": True, "sample_size_passed": True,
                    "all_expected_cells_valid": True, "valid_cell_count": 2, "expected_cell_count": 2,
                },
                "acceptance": {"fresh_learning_aggregation": "per-seed-median"},
            })
            lock = lock_module.build_lock(config_path=config, plan_path=plan, manifest_path=manifest, analysis_path=analysis)
            config_digest = lock_module._sha256(config)

        self.assertEqual(lock["status"], "locked")
        self.assertFalse(lock["scientific_conclusion_allowed"])
        self.assertEqual(lock["formal_design_locked_for_next_phase"]["seeds"], [1, 2])
        self.assertEqual(len(lock["lock_digest"]), 64)
        self.assertEqual(lock["config"]["sha256"], config_digest)

    def test_lock_rejects_blocked_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write(root / "config.json", {"phases": {"calibration": {}, "confirmatory": {}}})
            plan = self._write(root / "plan.json", {})
            manifest = self._write(root / "manifest.json", {"phase": "calibration", "commands": [{"status": "succeeded"}]})
            analysis = self._write(root / "analysis.json", {"phase": "calibration", "status": "blocked", "candidate_evidence_ready": False})
            with self.assertRaisesRegex(ValueError, "candidate-evidence-ready"):
                lock_module.build_lock(config_path=config, plan_path=plan, manifest_path=manifest, analysis_path=analysis)


if __name__ == "__main__":
    unittest.main()
