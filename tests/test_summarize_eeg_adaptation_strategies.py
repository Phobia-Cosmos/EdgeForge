import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize-eeg-adaptation-strategies.py"
    spec = importlib.util.spec_from_file_location("strategy_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, strategy: str, offset: float = 0.0) -> None:
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
        "adaptation_strategy": strategy,
    }
    run = {
        "architecture": "tcn",
        "seed": 1,
        "adaptation_strategy": strategy,
        "stages": [{
            "subject": 2,
            "warm": {"curve": [
                {"step": 0, "accuracy": 0.5, "retention_accuracy": 0.6},
                {"step": 1, "accuracy": 0.6, "retention_accuracy": 0.5},
            ]},
            "fresh": {"curve": [
                {"step": 0, "accuracy": 0.5, "retention_accuracy": 0.6},
                {"step": 1, "accuracy": 0.7 + offset, "retention_accuracy": 0.5},
            ]},
            "gaps": [{"step": 0, "fresh_gap": 0.0}, {"step": 1, "fresh_gap": 0.1 + offset}],
        }],
    }
    path.write_text(json.dumps({"metadata": metadata, "adaptation_strategy": strategy, "runs": [run]}), encoding="utf-8")


class StrategySummaryTests(unittest.TestCase):
    def test_pairs_strategy_cells_and_keeps_scientific_gate_closed(self):
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "plain.json", "plain")
            _write(root / "replay.json", "source_replay", 0.05)
            result = module.compare(
                {"plain": root / "plain.json", "source_replay": root / "replay.json"},
                root / "out",
                architecture="tcn",
                budgets=(1,),
            )
            self.assertAlmostEqual(result["stats"]["source_replay"]["1"]["fresh_gap_mean"], 0.15)
            self.assertAlmostEqual(result["paired_deltas_vs_plain"]["source_replay"]["1"]["fresh_gap_mean_delta"], 0.05)
            self.assertFalse(result["scientific_conclusion_allowed"])
            self.assertTrue((root / "out" / "REPORT.md").is_file())


if __name__ == "__main__":
    unittest.main()
