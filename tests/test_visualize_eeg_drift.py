import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "visualize-eeg-drift.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_visualize_eeg_drift", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EEGDriftVisualizationTests(unittest.TestCase):
    def test_visualize_uses_explicit_subject_groups(self):
        module = _load_module()
        loaded = []

        def profile(_root, group, subject):
            loaded.append((group, subject))
            return {"group": group, "subject": subject, "values": np.ones((1, 8, 1200)), "labels": np.zeros(1)}

        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(module, "_profile", side_effect=profile), \
             mock.patch.object(module, "_waveform_plot"), \
             mock.patch.object(module, "_normalized_overlay_plot"), \
             mock.patch.object(module, "_spectra_plot"), \
             mock.patch.object(module, "_rms_trajectory_plot"), \
             mock.patch.object(module, "_pca_plot", return_value={"rows": 3}), \
             mock.patch.object(module, "_label_plot"), \
             mock.patch.object(module, "_quantify_drift", return_value={}), \
             mock.patch.object(module, "_subject_distance_plot", return_value={"matrix": []}):
            output = Path(directory) / "output"
            module.visualize(
                Path(directory),
                output,
                [61, 62],
                source_subjects=[1, 60],
                retention_subjects=[81],
            )

            self.assertEqual(loaded, [("source", 1), ("source", 60), ("target", 61), ("target", 62), ("retention", 81)])
            summary = json.loads((output / "visualization-summary.json").read_text())
            self.assertEqual(summary["groups"], {"source": [1, 60], "target": [61, 62], "retention": [81]})

    def test_mean_epoch_psd_is_finite_and_channel_averaged(self):
        module = _load_module()
        values = np.zeros((3, 2, 2048), dtype=np.float32)
        time = np.arange(2048, dtype=np.float32) / module.FS_HZ
        values[:, 0] = np.sin(2 * np.pi * 2 * time)
        values[:, 1] = 0.5 * np.sin(2 * np.pi * 10 * time)
        frequencies, power = module._mean_epoch_psd(values)
        self.assertEqual(frequencies.shape, power.shape)
        self.assertTrue(np.all(np.isfinite(power)))
        self.assertGreater(float(power[(frequencies >= 1.5) & (frequencies < 2.5)].max()), float(power[(frequencies >= 10.0) & (frequencies < 11.0)].max()))

    def test_quantitative_drift_reports_scale_and_spectral_distance(self):
        module = _load_module()
        rng = np.random.default_rng(0)
        profiles = [
            {"group": "target", "subject": 2, "values": rng.normal(size=(4, 8, 2048)).astype(np.float32), "labels": np.zeros(4, dtype=np.int64)},
            {"group": "target", "subject": 11, "values": (10 * rng.normal(size=(4, 8, 2048))).astype(np.float32), "labels": np.zeros(4, dtype=np.int64)},
        ]
        result = module._quantify_drift(profiles, [2, 11])
        self.assertGreater(result["target_rms_max_min_ratio"], 5.0)
        self.assertEqual(set(result["target_spectral_js_to_median"]), {"2", "11"})
        self.assertTrue(all(value >= 0.0 for value in result["target_spectral_js_to_median"].values()))

    def test_rms_trajectory_plot_is_written(self):
        module = _load_module()
        profiles = [{"group": "target", "subject": 2, "values": np.ones((4, 8, 1200), dtype=np.float32), "labels": np.zeros(4, dtype=np.int64)}]
        with tempfile.TemporaryDirectory() as directory:
            module._rms_trajectory_plot(profiles, Path(directory), [2])
            self.assertTrue((Path(directory) / "eeg-drift-rms-trajectory.png").is_file())

    def test_subject_distance_and_overlay_plots_are_written(self):
        module = _load_module()
        rng = np.random.default_rng(1)
        profiles = [
            {"group": "target", "subject": 2, "values": rng.normal(size=(4, 8, 1200)).astype(np.float32), "labels": np.zeros(4, dtype=np.int64)},
            {"group": "target", "subject": 11, "values": (2 * rng.normal(size=(4, 8, 1200))).astype(np.float32), "labels": np.zeros(4, dtype=np.int64)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            module._normalized_overlay_plot(profiles, output, [2, 11])
            distance = module._subject_distance_plot(profiles, output, [2, 11])
            self.assertEqual(len(distance["matrix"]), 2)
            self.assertTrue((output / "eeg-drift-normalized-overlay.png").is_file())
            self.assertTrue((output / "eeg-drift-subject-distance.png").is_file())


if __name__ == "__main__":
    unittest.main()
