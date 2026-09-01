import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eeg_lop_diagnostics.py"
    spec = importlib.util.spec_from_file_location("edgeforge_eeg_lop_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EEGLoPScriptConfigTests(unittest.TestCase):
    def test_json_manifest_translates_model_and_probe_defaults(self):
        module = _load_script_module()
        payload = {
            "model": {"name": "brainuicl", "in_channels": 8, "input_length": 3000, "num_classes": 5},
            "diagnostics": {
                "max_observations": 17,
                "objective": {"name": "output_mean", "label_source": "unlabeled"},
                "hessian": {"mode": "power", "iterations": 3},
                "fresh_warm": {"budgets": [0, 2, 5]},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            defaults, model = module._manifest_cli_defaults(path)
        self.assertEqual(defaults["architectures"], "brainuicl")
        self.assertEqual(defaults["channels"], 8)
        self.assertEqual(defaults["max_observations"], 17)
        self.assertEqual(defaults["objective"], "output_mean")
        self.assertEqual(defaults["label_source"], "none")
        self.assertEqual(defaults["hessian_mode"], "power")
        self.assertEqual(defaults["probe_steps"], "0,2,5")
        self.assertEqual(model["name"], "brainuicl")

    def test_cli_result_contains_edgeforge_metric_envelope(self):
        module = _load_script_module()
        import tempfile
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            command = [
                sys.executable,
                str(Path(module.__file__)),
                "--data",
                "synthetic-eeg",
                "--architectures",
                "eegnet",
                "--tasks",
                "2",
                "--train-samples",
                "4",
                "--eval-samples",
                "4",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--channels",
                "2",
                "--length",
                "16",
                "--classes",
                "2",
                "--probe-steps",
                "0,1",
                "--objective",
                "output_mean",
                "--label-source",
                "none",
                "--output-dir",
                str(output),
            ]
            environment = dict(__import__("os").environ)
            environment["PYTHONPATH"] = str(Path(module.__file__).resolve().parents[1] / "src")
            completed = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
            self.assertIn("scientific_conclusion_allowed", completed.stdout)
            payload = json.loads((output / "synthetic-eeg-lop-diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["envelope"]["schema"], "edgeforge-bundle-v1")
        self.assertEqual(payload["config"]["objective"], "output_mean")
        self.assertEqual(payload["config"]["label_source"], "none")
        self.assertGreater(len(payload["metrics"]), 0)
        self.assertTrue(all("metric_role" in item["context"] for item in payload["metrics"]))


if __name__ == "__main__":
    unittest.main()
