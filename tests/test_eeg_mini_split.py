import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_create_eeg_mini_split_preserves_pairs(tmp_path):
    source = tmp_path / "source"
    for subject in (1, 2, 5):
        (source / str(subject) / "data").mkdir(parents=True)
        (source / str(subject) / "label").mkdir(parents=True)
        np.save(source / str(subject) / "data" / "0.npy", np.zeros((20, 8, 3000), dtype=np.float32))
        np.save(source / str(subject) / "label" / "0.npy", np.arange(20, dtype=np.int64) % 5)
    output = tmp_path / "out"
    script = Path(__file__).parents[1] / "scripts" / "create-eeg-mini-split.py"
    subprocess.run([sys.executable, str(script), "--source-root", str(source), "--output-root", str(output), "--source-subjects", "1", "--target-subject", "2", "--retention-subject", "5"], check=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["records"]) == 3
    assert all(item["source_data_sha256"] == item["output_data_sha256"] for item in manifest["records"])
    assert (output / "source/1/data/0.npy").is_file()
