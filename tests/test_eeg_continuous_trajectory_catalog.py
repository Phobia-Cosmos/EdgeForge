import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, module_name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


catalog_module = _load_script("build-eeg-continuous-trajectory-catalog.py", "edgeforge_test_continuous_catalog")
audit_module = _load_script("audit-eeg-continuous-lop.py", "edgeforge_test_continuous_audit")


def _stage(index: int, subject: int, gap: float) -> dict:
    final = {
        "step": 50,
        "retention_accuracy": 0.7 - index * 0.01,
        "retention_loss": 0.8 + index * 0.01,
        "retention_macro_f1": 0.6 - index * 0.01,
    }
    return {
        "stage": index,
        "subject": subject,
        "gaps": [
            {"step": 0, "fresh_gap": gap / 2},
            {"step": 25, "fresh_gap": gap * 0.8},
            {"step": 50, "fresh_gap": gap},
        ],
        "warm": {"final": final},
        "fresh": {"final": {**final, "retention_accuracy": final["retention_accuracy"] - 0.02}},
    }


def _write_summary(root: Path) -> Path:
    subjects = [2, 11]
    runs = []
    for offset, seed in enumerate((4321, 4322, 4323)):
        run = {
            "architecture": "tcn",
            "seed": seed,
            "parameters": 5429,
            "stages": [
                _stage(0, subjects[0], 0.2 + offset * 0.01),
                _stage(1, subjects[1], 0.3 + offset * 0.01),
            ],
        }
        run_dir = root / "tcn" / f"seed{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
        runs.append(run)
    summary = {
        "metadata": {
            "dataset": "ISRUC",
            "target_subject_order": subjects,
            "budgets": [0, 25, 50],
            "adapt_lr": 0.002,
        },
        "runs": runs,
        "scientific_conclusion_allowed": False,
    }
    path = root / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


class ContinuousTrajectoryCatalogTests(unittest.TestCase):
    def test_catalog_is_gate_ready_and_keeps_retention_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = _write_summary(root)
            output = root / "derived" / "trajectory-catalog.json"
            catalog = catalog_module.build_catalog(summary, output, budget=50)
            self.assertEqual(len(catalog["experiments"]), 3)
            self.assertEqual({item["learning_rate"] for item in catalog["experiments"]}, {0.002})
            first_result = json.loads((output.parent / catalog["experiments"][0]["runner"]["result_path"]).read_text())
            outcomes = [item for item in first_result["metrics"] if item["name"] == "task.plasticity.fresh_gap"]
            retention = [item for item in first_result["metrics"] if item["context"]["metric_role"] == "retention"]
            self.assertEqual([item["context"]["target_subject"] for item in outcomes], [2, 11])
            self.assertEqual(len(retention), 12)
            self.assertTrue(all(item["context"]["metric_role"] == "outcome" for item in outcomes))

            audited = audit_module.audit(summary, root / "audit", budget=50, bootstrap_repeats=100)
            report = audited["reports"]["tcn"]
            self.assertEqual(report["status"], "candidate")
            self.assertEqual(report["seed_count"], 3)
            self.assertEqual(report["stage_count"], 2)
            self.assertTrue(report["design_consistent"])
            self.assertTrue(report["direction_supported"])
            gate = json.loads((root / "audit" / "gate" / "tcn.json").read_text())
            self.assertTrue(gate["retention_separate"])
            self.assertEqual(len(gate["retention_inventory"]), 36)

    def test_selected_budget_is_read_from_each_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = _write_summary(root)
            output = root / "budget25" / "trajectory-catalog.json"
            catalog_module.build_catalog(summary, output, budget=25)
            result = json.loads((output.parent / "trajectories" / "tcn" / "seed4321.json").read_text())
            values = [item["value"] for item in result["metrics"] if item["name"] == "task.plasticity.fresh_gap"]
            self.assertEqual(len(values), 2)
            self.assertAlmostEqual(values[0], 0.16)
            self.assertAlmostEqual(values[1], 0.24)
            self.assertEqual(result["spec"]["probe_budget"], 25)


if __name__ == "__main__":
    unittest.main()
