import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "summarize-eeg-condition-experiment.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_summarize_eeg_condition_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(path: Path, offset: float) -> None:
    runs = []
    for seed in (1, 2, 3):
        runs.append({
            "architecture": "tcn",
            "seed": seed,
            "source": {"accuracy": 0.4},
            "stages": [{"gaps": [{"step": 0, "fresh_gap": 0.0}, {"step": 10, "fresh_gap": 0.1 + offset}]}],
        })
    path.write_text(json.dumps({"runs": runs}), encoding="utf-8")


class SummarizeEEGConditionExperimentTests(unittest.TestCase):
    def test_paired_budget_delta_is_computed(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            condition = root / "condition.json"
            _summary(baseline, 0.0)
            _summary(condition, 0.05)
            result = module.compare(baseline, {"condition": condition}, root / "out")
            self.assertAlmostEqual(result["stats"]["condition"]["budget_metrics"]["10"]["mean"], 0.15)
            self.assertAlmostEqual(result["paired_deltas_vs_raw"]["condition"]["10"]["mean_delta"], 0.05)
            self.assertTrue((root / "out" / "REPORT.md").is_file())


if __name__ == "__main__":
    unittest.main()
