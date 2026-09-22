import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "summarize-eeg-checkpoint-diagnostics.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_summarize_eeg_checkpoint_diagnostics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnostic(rank: float) -> dict:
    return {
        "status": "computed",
        "representations": {
            "embedding": {"effective_rank": rank, "effective_rank_normalized": rank / 10.0},
            "block1": {"effective_rank": rank + 1.0},
            "classifier_input": {"near_zero_fraction": 0.1},
        },
        "gradient": {"norm_l2": rank, "nonzero_fraction": 0.9},
        "parameter_norm": {"global_relative_update": rank / 100.0},
    }


class SummarizeEEGCheckpointDiagnosticsTests(unittest.TestCase):
    def test_summary_joins_diagnostics_with_fresh_gap(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.json"
            runs = []
            for seed, rank in ((1, 2.0), (2, 3.0), (3, 4.0)):
                curve = [
                    {"step": 0, "diagnostics": _diagnostic(rank)},
                    {"step": 50, "diagnostics": _diagnostic(rank + 1.0)},
                ]
                runs.append(
                    {
                        "architecture": "tcn",
                        "seed": seed,
                        "stages": [
                            {
                                "stage": 0,
                                "subject": 2,
                                "gaps": [{"step": 0, "fresh_gap": 0.0}, {"step": 50, "fresh_gap": 0.2}],
                                "warm": {"curve": curve},
                                "fresh": {"curve": curve},
                            }
                        ],
                    }
                )
            summary.write_text(json.dumps({"runs": runs}), encoding="utf-8")
            result = module.summarize(summary, root / "out")
            self.assertEqual(result["row_count"], 12)
            self.assertEqual(result["aggregate"]["warm:50"]["rows"], 3)
            self.assertAlmostEqual(result["aggregate"]["warm:0"]["metrics"]["embedding_effective_rank"]["mean"], 3.0)
            self.assertTrue((root / "out" / "REPORT.md").is_file())
            self.assertFalse(result["scientific_conclusion_allowed"])


if __name__ == "__main__":
    unittest.main()
