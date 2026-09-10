import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "generate-synthetic-eeg-label-reversal.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_label_reversal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticEEGLabelReversalTests(unittest.TestCase):
    def test_target_labels_are_recomputed_from_opposite_feature_polarity(self):
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
            labels = np.load(output / "target" / "4" / "label" / "000.npy")
            self.assertEqual(result["transformed_target_files"], 1)
            self.assertEqual(labels.tolist(), [0, 1])
            self.assertEqual(json_load(output / "manifest.json")["target_shift"], "all target labels reverse the source feature-0 polarity")


def json_load(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
