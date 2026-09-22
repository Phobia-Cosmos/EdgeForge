import importlib.util
import unittest

import torch


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run-eeg-architecture-continuous-lop.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_eeg_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EEGInputNormalizationTests(unittest.TestCase):
    def test_epoch_rms_removes_global_scale(self):
        module = _load_module()
        values = torch.randn(4, 8, 128)
        normalized = module.normalize_inputs(values, "epoch_rms")
        scaled = module.normalize_inputs(values * 13.0, "epoch_rms")
        self.assertTrue(torch.allclose(normalized, scaled, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(torch.sqrt(torch.mean(normalized.square(), dim=(1, 2))), torch.ones(4), atol=1e-5))

    def test_channel_zscore_normalizes_each_channel(self):
        module = _load_module()
        values = torch.randn(4, 8, 128) * 4.0 + 7.0
        normalized = module.normalize_inputs(values, "channel_zscore")
        self.assertTrue(torch.allclose(normalized.mean(dim=-1), torch.zeros(4, 8), atol=1e-5))
        self.assertTrue(torch.allclose(normalized.std(dim=-1, unbiased=False), torch.ones(4, 8), atol=1e-5))

    def test_none_returns_original_tensor(self):
        module = _load_module()
        values = torch.randn(2, 8, 64)
        self.assertIs(module.normalize_inputs(values, "none"), values)


if __name__ == "__main__":
    unittest.main()
