import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "plot-eeg-lop-dose-curve.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_plot_eeg_lop_dose_curve", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(path: Path, offset: float) -> None:
    runs = []
    for seed in (1, 2, 3):
        runs.append(
            {
                "architecture": "tcn",
                "seed": seed,
                "stages": [
                    {
                        "gaps": [
                            {"step": 0, "fresh_gap": offset},
                            {"step": 50, "fresh_gap": 0.1 + offset},
                        ]
                    }
                ],
            }
        )
    path.write_text(json.dumps({"runs": runs}), encoding="utf-8")


class PlotEEGLoPDoseCurveTests(unittest.TestCase):
    def test_build_curve_assigns_monotonic_noise_doses_and_writes_plot(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            snr20 = root / "snr20.json"
            snr10 = root / "snr10.json"
            _summary(baseline, 0.0)
            _summary(snr20, 0.02)
            _summary(snr10, 0.04)
            output = root / "out"
            result = module.build_curve(
                baseline,
                {"target_snr20_noise": snr20, "target_snr10_noise": snr10},
                output,
            )
            doses = {
                row["condition"]: row["dose"]
                for row in result["rows"]
                if int(row["budget"]) == 50
            }
            self.assertEqual(doses["raw"], 0.0)
            self.assertLess(doses["target_snr20_noise"], doses["target_snr10_noise"])
            self.assertAlmostEqual(result["rows"][0]["ci95"][0], result["rows"][0]["ci95"][1])
            self.assertTrue((output / "dose-curve.json").is_file())
            self.assertTrue((output / "eeg-lop-dose-curves.png").is_file())
            self.assertFalse(result["scientific_conclusion_allowed"])


if __name__ == "__main__":
    unittest.main()
