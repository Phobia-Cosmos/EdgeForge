import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize-eeg-architectures.py"
    spec = importlib.util.spec_from_file_location("architecture_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, architecture: str, offset: float = 0.0) -> None:
    metadata = {
        "dataset": "synthetic",
        "source_subjects": [1],
        "target_subject_order": [2],
        "retention_subjects": [3],
        "target_split": "fixed",
        "input_shape": [2, 4],
        "classes": 2,
        "source_epochs": 1,
        "source_lr": 0.1,
        "budgets": [0, 1],
        "adapt_lr": 0.01,
        "batch_size": 2,
        "fresh_mode": "fresh",
        "warm_mode": "warm",
        "adaptation_strategy": "plain",
        "replay_ratio": 0.0,
        "l2_sp_lambda": 0.0,
    }
    run = {
        "architecture": architecture,
        "seed": 1,
        "stages": [{
            "subject": 2,
            "warm": {"curve": [
                {"step": 0, "accuracy": 0.5, "retention_accuracy": 0.6},
                {"step": 1, "accuracy": 0.6, "retention_accuracy": 0.5 + offset},
            ]},
            "fresh": {"curve": [
                {"step": 0, "accuracy": 0.5, "retention_accuracy": 0.6},
                {"step": 1, "accuracy": 0.7 + offset, "retention_accuracy": 0.5},
            ]},
            "gaps": [{"step": 0, "fresh_gap": 0.0}, {"step": 1, "fresh_gap": 0.1 + offset}],
        }],
    }
    path.write_text(json.dumps({"metadata": metadata, "runs": [run]}), encoding="utf-8")


class ArchitectureSummaryTests(unittest.TestCase):
    def test_pairs_architecture_cells_and_keeps_scientific_gate_closed(self):
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "tcn.json", "tcn")
            _write(root / "brainuicl.json", "brainuicl", 0.05)
            result = module.compare(
                {"tcn": root / "tcn.json", "brainuicl": root / "brainuicl.json"},
                root / "out",
                baseline="tcn",
                budgets=(1,),
            )
            self.assertAlmostEqual(result["stats"]["brainuicl"]["1"]["fresh_gap_mean"], 0.15)
            self.assertAlmostEqual(result["paired_deltas_vs_baseline"]["brainuicl"]["1"]["fresh_gap_mean_delta"], 0.05)
            self.assertFalse(result["scientific_conclusion_allowed"])
            self.assertTrue((root / "out" / "REPORT.md").is_file())

    def test_rejects_unmatched_protocols(self):
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "tcn.json", "tcn")
            _write(root / "brainuicl.json", "brainuicl")
            payload = json.loads((root / "brainuicl.json").read_text())
            payload["metadata"]["target_subject_order"] = [99]
            (root / "brainuicl.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "do not share one protocol"):
                module.compare(
                    {"tcn": root / "tcn.json", "brainuicl": root / "brainuicl.json"},
                    root / "out",
                    baseline="tcn",
                    budgets=(1,),
                )


if __name__ == "__main__":
    unittest.main()
