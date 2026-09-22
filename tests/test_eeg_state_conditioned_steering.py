import importlib.util
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - dependency-free test collection
    torch = None


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run-eeg-state-conditioned-steering.py"
    spec = importlib.util.spec_from_file_location("edgeforge_state_steering", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None, "PyTorch is required for steering helpers")
class StateConditionedSteeringTests(unittest.TestCase):
    def test_band_limit_and_rms_bound(self):
        module = _load_module()
        base = torch.randn(2, 3, 128)
        delta = module.random_delta(base, seed=7, fraction=0.05, sample_rate=100.0)
        ratio = module.rms(delta, dims=(-1,)) / module.rms(base, dims=(-1,))
        self.assertTrue(torch.allclose(ratio, torch.full_like(ratio, 0.05), atol=1e-5, rtol=1e-4))
        frequencies = torch.fft.rfftfreq(128, d=0.01)
        spectrum = torch.fft.rfft(delta, dim=-1).abs().mean(dim=(0, 1))
        self.assertLessEqual(float(spectrum[frequencies < 0.5].max()), 1e-5)
        self.assertLessEqual(float(spectrum[frequencies > 40.0].max()), 1e-5)

    def test_unit_active_fraction_exposes_eegnet_filters(self):
        module = _load_module()
        from edgeforge.eeg_models import EEGModelConfig, build_eeg_decoder

        model = build_eeg_decoder(EEGModelConfig(name="eegnet", in_channels=8, input_length=128, num_classes=5, width=4, feature_dim=8, kernel_size=7))
        values = torch.randn(3, 8, 128)
        coverage = module.unit_active_fraction(model, values)
        self.assertEqual(tuple(coverage.shape), (4,))
        self.assertTrue(torch.all((coverage >= 0.0) & (coverage <= 1.0)))


if __name__ == "__main__":
    unittest.main()
