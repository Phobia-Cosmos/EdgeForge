import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "visualize-eeg-sequences.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_visualize_eeg_sequences", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EEGSequenceVisualizationTests(unittest.TestCase):
    def test_metrics_identify_characteristic_epochs_and_transitions(self):
        module = _load_module()
        values = np.ones((20, 8, 3000), dtype=np.float32)
        values[3] *= 0.1
        values[17] *= 5.0
        labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
        metrics = module._sequence_metrics(values, labels)
        self.assertEqual(metrics["lowest_rms_epoch"], 3)
        self.assertEqual(metrics["highest_rms_epoch"], 17)
        self.assertEqual(metrics["first_label_transition_epoch"], 10)
        self.assertEqual(metrics["label_transition_count"], 1)
        self.assertEqual(metrics["largest_adjacent_rms_jump_epoch"], 17)
        selected = module._characteristic_epochs(metrics)
        selected_indices = [item["epoch"] for item in selected]
        self.assertIn(3, selected_indices)
        self.assertIn(17, selected_indices)
        self.assertIn(10, selected_indices)

    def test_selection_keeps_complementary_sequences(self):
        module = _load_module()
        entries = [
            {"sequence": "1", "metrics": {"label_transition_count": 8, "epoch_rms_cv": 0.1, "sequence_rms": 1.0}},
            {"sequence": "2", "metrics": {"label_transition_count": 2, "epoch_rms_cv": 0.9, "sequence_rms": 2.0}},
            {"sequence": "3", "metrics": {"label_transition_count": 3, "epoch_rms_cv": 0.2, "sequence_rms": 1.5}},
        ]
        chosen = module._choose_sequences(entries, 2)
        self.assertEqual([item["sequence"] for item in chosen], ["1", "2"])

    def test_plots_and_explanation_are_written(self):
        module = _load_module()
        rng = np.random.default_rng(0)
        values = rng.normal(size=(20, 8, 3000)).astype(np.float32)
        labels = np.arange(20, dtype=np.int64) % 5
        metrics = module._sequence_metrics(values, labels)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            module._waveform_stack(values, labels, metrics, output / "stack.png", 2, "45", 0)
            module._all_channel_heatmap(values, labels, output / "channels.png", 2, "45")
            module._characteristic_epoch_panel(values, labels, metrics, output / "characteristic.png", 2, "45")
            self.assertTrue((output / "stack.png").is_file())
            self.assertTrue((output / "channels.png").is_file())
            self.assertTrue((output / "characteristic.png").is_file())


if __name__ == "__main__":
    unittest.main()
