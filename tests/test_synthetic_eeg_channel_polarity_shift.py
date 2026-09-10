import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "generate-synthetic-eeg-channel-polarity-shift.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_channel_polarity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticEEGChannelPolarityShiftTests(unittest.TestCase):
    def test_waveform_is_inverted_but_labels_keep_original_polarity(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            target = source / "target" / "4"
            (target / "data").mkdir(parents=True)
            (target / "label").mkdir(parents=True)
            values = np.zeros((2, 8, 3000), dtype=np.float32)
            values[0, 0, :1500] = 2.0
            values[1, 0, :1500] = -2.0
            np.save(target / "data" / "000.npy", values)
            np.save(target / "label" / "000.npy", np.array([0, 1], dtype=np.int64))
            (source / "manifest.json").write_text('{"dataset":"base"}\n', encoding="utf-8")
            output = Path(directory) / "output"
            result = module.generate(source, output)
            transformed = np.load(output / "target" / "4" / "data" / "000.npy")
            labels = np.load(output / "target" / "4" / "label" / "000.npy")
            self.assertEqual(result["transformed_target_files"], 1)
            self.assertEqual(transformed[0, 0, 0], -2.0)
            self.assertEqual(transformed[1, 0, 0], 2.0)
            self.assertEqual(labels.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
