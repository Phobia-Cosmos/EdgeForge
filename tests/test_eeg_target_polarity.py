import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run-eeg-architecture-continuous-lop.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_target_polarity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EEGTargetPolarityTests(unittest.TestCase):
    def test_read_target_stage_flips_only_selected_channel(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "target" / "2" / "data"
            label_dir = root / "target" / "2" / "label"
            data_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            values = np.ones((20, 8, 3000), dtype=np.float32)
            np.save(data_dir / "0.npy", values, allow_pickle=False)
            np.save(label_dir / "0.npy", np.arange(20, dtype=np.int64) % 5, allow_pickle=False)
            train, train_labels, evaluation, eval_labels, _ = module.read_target_stage(root, 2, channel_polarity=3)
            self.assertEqual(tuple(train.shape), (10, 8, 3000))
            self.assertEqual(tuple(evaluation.shape), (10, 8, 3000))
            self.assertTrue(np.all(train[:, 3].numpy() == -1.0))
            self.assertTrue(np.all(train[:, 0].numpy() == 1.0))
            np.testing.assert_array_equal(train_labels.numpy(), np.arange(10) % 5)
            np.testing.assert_array_equal(eval_labels.numpy(), np.arange(10, 20) % 5)


if __name__ == "__main__":
    unittest.main()
