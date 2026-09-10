import importlib.util
import tempfile
import unittest
from pathlib import Path

try:
    import numpy
    import numpy as np
except ImportError:
    numpy = None
    np = None

if numpy is not None:
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "brainuicl-unlabeled-diagnostics.py"
    SPEC = importlib.util.spec_from_file_location("edgeforge_brainuicl_unlabeled", SCRIPT)
    assert SPEC is not None and SPEC.loader is not None
    diagnostics = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(diagnostics)
else:
    diagnostics = None


@unittest.skipUnless(numpy is not None, "NumPy is required for unlabeled diagnostics")
class BrainUICLUnlabeledDiagnosticsTests(unittest.TestCase):
    def test_sample_data_files_discovers_signals_without_label_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            data_dir = root / "sub-001" / "data"
            data_dir.mkdir(parents=True)
            signal = data_dir / "0.npy"
            np.save(signal, np.zeros((20, 8, 3000), dtype=np.float32), allow_pickle=False)
            self.assertEqual(diagnostics.sample_data_files(root, 1, 1), [signal])

    def test_data_loader_does_not_require_readable_label_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "0.npy"
            # The loader validates only the signal contract.  A tiny constant
            # array keeps this test deterministic while avoiding label reads.
            np.save(data, np.zeros((20, 8, 3000), dtype=np.float32), allow_pickle=False)
            batches = diagnostics.load_data_batches([(data, root / "missing-label.npy")], 1, "ISRUC")
            self.assertEqual(len(batches), 1)
            self.assertEqual(tuple(batches[0].shape), (1, 20, 8, 3000))

    def test_flatten_preserves_protocol_context_and_step(self):
        rows = []
        diagnostics._flatten({"confidence": 0.75, "labels_loaded": False}, "task.unlabeled", stage=4, context={"measurement_protocol": "unlabeled-pseudo-consistency-v1"}, output=rows)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 4)
        self.assertEqual(rows[0]["context"]["measurement_protocol"], "unlabeled-pseudo-consistency-v1")
        self.assertEqual(rows[0]["name"], "task.unlabeled.confidence")


if __name__ == "__main__":
    unittest.main()
