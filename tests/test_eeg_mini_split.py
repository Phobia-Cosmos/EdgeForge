import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


class EEGMiniSplitTests(unittest.TestCase):
    def test_create_eeg_mini_split_preserves_pairs_and_selects_diversity(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            for subject in (1, 2, 5):
                (source / str(subject) / "data").mkdir(parents=True)
                (source / str(subject) / "label").mkdir(parents=True)
                for index in (0, 1):
                    np.save(source / str(subject) / "data" / f"{index}.npy", np.full((20, 8, 3000), index, dtype=np.float32))
                np.save(source / str(subject) / "label" / "0.npy", np.zeros(20, dtype=np.int64))
                np.save(source / str(subject) / "label" / "1.npy", np.arange(20, dtype=np.int64) % 5)
            output = root / "out"
            script = Path(__file__).parents[1] / "scripts" / "create-eeg-mini-split.py"
            subprocess.run([sys.executable, str(script), "--source-root", str(source), "--output-root", str(output), "--source-subjects", "1", "--target-subject", "2", "--retention-subject", "5", "--selection-strategy", "label-diversity", "--public-release"], check=True)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["records"]), 3)
            self.assertTrue(manifest["public_release"])
            self.assertIsNone(manifest["source_root"])
            self.assertTrue(all(item["file"] == "1" for item in manifest["records"]))
            self.assertTrue(all(item["source_data_sha256"] == item["output_data_sha256"] for item in manifest["records"]))
            self.assertTrue((output / "source/1/data/1.npy").is_file())


if __name__ == "__main__":
    unittest.main()
