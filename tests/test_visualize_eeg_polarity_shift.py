import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "visualize-eeg-polarity-shift.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_visualize_polarity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisualizeEEGPolarityShiftTests(unittest.TestCase):
    def test_visualization_reports_sign_flip_and_power_invariance(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_root = root / "clean"
            shift_root = root / "shift"
            rng = np.random.default_rng(7)
            clean = rng.standard_normal((2, 8, 3000), dtype=np.float32)
            shifted = clean.copy()
            shifted[:, 0] *= -1.0
            for dataset_root, values in ((clean_root, clean), (shift_root, shifted)):
                data_dir = dataset_root / "target" / "2" / "data"
                data_dir.mkdir(parents=True)
                np.save(data_dir / "0.npy", values, allow_pickle=False)
            output = root / "figures"
            summary = module.visualize(clean_root, shift_root, output, 2, 0)
            self.assertAlmostEqual(summary["selected_channel_correlation"], -1.0)
            self.assertAlmostEqual(summary["selected_channel_rms_ratio"], 1.0)
            self.assertLess(summary["selected_channel_psd_relative_error"], 1e-7)
            self.assertTrue((output / "eeg-polarity-waveform-and-psd.png").is_file())
            self.assertTrue((output / "eeg-polarity-channel-correlation.png").is_file())


if __name__ == "__main__":
    unittest.main()
