import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "induce-eeg-lop-by-metrics.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_induce_eeg_lop_by_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetricDrivenPlanTests(unittest.TestCase):
    def test_plan_ranks_consistent_condition(self):
        module = _load_module()
        def summary(gaps):
            return {"runs": [{"architecture": "tcn", "seed": 1, "stages": [{"subject": index, "gaps": [{"step": 25, "fresh_gap": value}]} for index, value in enumerate(gaps)]}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            good = root / "good.json"
            base.write_text(json.dumps(summary([-0.1, 0.0, -0.2])))
            good.write_text(json.dumps(summary([0.1, 0.2, 0.3])))
            result = module.plan(base, {"target_crosstalk10": good}, root / "out", budget=25)
            self.assertEqual(result["next_conditions"], ["target_crosstalk10"])
            self.assertTrue((root / "out" / "PLAN.md").is_file())


if __name__ == "__main__":
    unittest.main()
