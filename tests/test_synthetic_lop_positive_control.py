import importlib.util
import unittest

import torch


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run-synthetic-lop-positive-control.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_synthetic_lop", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticLoPPositiveControlTests(unittest.TestCase):
    def test_gradient_scale_changes_backward_path_not_forward_values(self):
        module = _load_module()
        torch.manual_seed(7)
        model = module.LatentEEGMLP()
        values = torch.randn(4, 8, 3000)
        baseline = model(values)
        model.set_encoder_gradient_scale(0.0)
        lesioned = model(values)
        self.assertTrue(torch.allclose(baseline, lesioned))

        model.zero_grad(set_to_none=True)
        model.set_encoder_gradient_scale(0.0)
        model(values).sum().backward()
        frozen_grad = model.encoder[0].weight.grad
        self.assertIsNotNone(frozen_grad)
        self.assertTrue(torch.allclose(frozen_grad, torch.zeros_like(frozen_grad)))

        model.zero_grad(set_to_none=True)
        model.set_encoder_gradient_scale(0.5)
        model(values).sum().backward()
        half_grad = model.encoder[0].weight.grad
        self.assertIsNotNone(half_grad)

        model.zero_grad(set_to_none=True)
        model.set_encoder_gradient_scale(1.0)
        model(values).sum().backward()
        full_grad = model.encoder[0].weight.grad
        self.assertIsNotNone(full_grad)
        self.assertTrue(torch.allclose(half_grad, full_grad * 0.5, atol=1e-6, rtol=1e-5))

    def test_scale_is_stored_as_float_buffer(self):
        module = _load_module()
        model = module.LatentEEGMLP()
        model.set_encoder_gradient_scale(-0.5)
        self.assertAlmostEqual(float(model.encoder_gradient_scale), -0.5)


if __name__ == "__main__":
    unittest.main()
