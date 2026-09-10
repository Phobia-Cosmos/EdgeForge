import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "prepare-eeg-lop-conditions.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_prepare_eeg_lop_conditions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrepareEEGLoPConditionsTests(unittest.TestCase):
    def _dataset(self, root: Path) -> None:
        for group, subject, scale in (("source", 1, 1.0), ("source", 2, 2.0), ("target", 3, 4.0), ("retention", 4, 1.5)):
            data_dir = root / group / str(subject) / "data"
            label_dir = root / group / str(subject) / "label"
            data_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            np.save(data_dir / "0.npy", (scale * np.ones((2, 8, 3000), dtype=np.float32)), allow_pickle=False)
            np.save(label_dir / "0.npy", np.asarray([0, 1], dtype=np.int64), allow_pickle=False)

    def test_rms_equalized_preserves_labels_and_equalizes_subject_scale(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            self._dataset(root)
            result = module.prepare(root, output, "rms_equalized")
            self.assertEqual(result["files"], 4)
            values = [np.load(output / group / str(subject) / "data" / "0.npy", allow_pickle=False) for group, subject in (("source", 1), ("source", 2), ("target", 3), ("retention", 4))]
            self.assertAlmostEqual(float(np.std(values[0])), 0.0, places=6)
            self.assertAlmostEqual(float(np.sqrt(np.mean(values[0] ** 2))), float(np.sqrt(np.mean(values[1] ** 2))), places=5)
            np.testing.assert_array_equal(np.load(output / "target" / "3" / "label" / "0.npy", allow_pickle=False), [0, 1])

    def test_target_noise_leaves_source_unchanged_and_is_correlated(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            self._dataset(root)
            varying = np.linspace(-1.0, 1.0, 2 * 8 * 3000, dtype=np.float32).reshape(2, 8, 3000)
            np.save(root / "target" / "3" / "data" / "0.npy", 4.0 * varying, allow_pickle=False)
            module.prepare(root, output, "target_snr20_noise", seed=7)
            source_original = np.load(root / "source" / "1" / "data" / "0.npy", allow_pickle=False)
            source_derived = np.load(output / "source" / "1" / "data" / "0.npy", allow_pickle=False)
            target_original = np.load(root / "target" / "3" / "data" / "0.npy", allow_pickle=False)
            target_derived = np.load(output / "target" / "3" / "data" / "0.npy", allow_pickle=False)
            np.testing.assert_array_equal(source_original, source_derived)
            self.assertGreater(float(np.corrcoef(target_original.reshape(-1), target_derived.reshape(-1))[0, 1]), 0.98)
            self.assertFalse(np.array_equal(target_original, target_derived))

    def test_gain_drift_is_smooth_and_preserves_labels(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            self._dataset(root)
            varying = np.linspace(-1.0, 1.0, 2 * 8 * 3000, dtype=np.float32).reshape(2, 8, 3000)
            np.save(root / "target" / "3" / "data" / "0.npy", 4.0 * varying, allow_pickle=False)
            module.prepare(root, output, "target_gain_drift10", seed=7)
            original = np.load(root / "target" / "3" / "data" / "0.npy", allow_pickle=False)
            derived = np.load(output / "target" / "3" / "data" / "0.npy", allow_pickle=False)
            ratio = derived / original
            self.assertGreater(float(ratio.min()), 0.89)
            self.assertLess(float(ratio.max()), 1.11)
            self.assertGreater(float(np.corrcoef(original.reshape(-1), derived.reshape(-1))[0, 1]), 0.99)
            np.testing.assert_array_equal(np.load(output / "target" / "3" / "label" / "0.npy", allow_pickle=False), [0, 1])

    def test_channel_crosstalk_preserves_axis_and_records_matrix(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            self._dataset(root)
            varying = np.zeros((2, 8, 3000), dtype=np.float32)
            varying[:, 0] = 1.0
            np.save(root / "target" / "3" / "data" / "0.npy", varying, allow_pickle=False)
            module.prepare(root, output, "target_crosstalk5", seed=7)
            original = np.load(root / "target" / "3" / "data" / "0.npy", allow_pickle=False)
            derived = np.load(output / "target" / "3" / "data" / "0.npy", allow_pickle=False)
            self.assertEqual(derived.shape, original.shape)
            self.assertAlmostEqual(float(derived[0, 0, 0]), 0.95, places=5)
            self.assertAlmostEqual(float(derived[0, 1, 0]), 0.025, places=5)
            self.assertFalse(np.array_equal(original, derived))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["perturbation"]["type"], "adjacent_channel_cross_talk")
            self.assertEqual(len(manifest["perturbation"]["matrix"]), 8)

    def test_baseline_drift_is_low_frequency_and_label_preserving(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            self._dataset(root)
            varying = np.linspace(-1.0, 1.0, 2 * 8 * 3000, dtype=np.float32).reshape(2, 8, 3000)
            np.save(root / "target" / "3" / "data" / "0.npy", varying, allow_pickle=False)
            module.prepare(root, output, "target_baseline_drift5", seed=7)
            original = np.load(root / "target" / "3" / "data" / "0.npy", allow_pickle=False)
            derived = np.load(output / "target" / "3" / "data" / "0.npy", allow_pickle=False)
            self.assertEqual(derived.shape, original.shape)
            self.assertGreater(float(np.corrcoef(original.reshape(-1), derived.reshape(-1))[0, 1]), 0.99)
            difference = derived - original
            frequencies = np.fft.rfftfreq(difference.shape[-1], d=1.0 / module.FS_HZ)
            spectrum = np.abs(np.fft.rfft(difference[0, 0]))
            peak = float(frequencies[int(np.argmax(spectrum[1:]) + 1)])
            self.assertAlmostEqual(peak, module.BASELINE_DRIFT_FREQUENCY_HZ, delta=0.05)
            np.testing.assert_array_equal(np.load(output / "target" / "3" / "label" / "0.npy", allow_pickle=False), [0, 1])

    def test_target_channel_polarity_inverts_only_selected_channel(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            self._dataset(root)
            varying = np.ones((2, 8, 3000), dtype=np.float32)
            np.save(root / "target" / "3" / "data" / "0.npy", varying, allow_pickle=False)
            module.prepare(root, output, "target_channel_polarity", channel_index=2)
            derived = np.load(output / "target" / "3" / "data" / "0.npy", allow_pickle=False)
            np.testing.assert_array_equal(derived[:, 2], -1.0)
            np.testing.assert_array_equal(derived[:, 0], 1.0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["perturbation"]["type"], "channel_polarity_inversion")
            self.assertEqual(manifest["perturbation"]["channel_index"], 2)
            target_record = next(item for item in manifest["records"] if item["group"] == "target")
            self.assertAlmostEqual(target_record["selected_channel_correlation"], -1.0)
            self.assertAlmostEqual(target_record["selected_channel_rms_ratio"], 1.0)
            self.assertEqual(target_record["selected_channel_magnitude_max_abs_error"], 0.0)
            self.assertEqual(target_record["other_channels_max_abs_error"], 0.0)
            np.testing.assert_array_equal(np.load(output / "target" / "3" / "label" / "0.npy", allow_pickle=False), [0, 1])


if __name__ == "__main__":
    unittest.main()
