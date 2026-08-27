import importlib.util
import unittest

try:
    import torch
    from torch import nn
except ImportError:  # Keep the core test suite usable without PyTorch.
    torch = None
    nn = None

from edgeforge import lop_metrics


@unittest.skipUnless(torch is not None, "PyTorch is required for numerical diagnostics")
class LopMetricsTests(unittest.TestCase):
    def test_spectrum_respects_eeg_token_and_conv_feature_axes(self):
        tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        token_summary = lop_metrics.spectral_summary(tokens, feature_axis=-1)
        self.assertEqual(token_summary["observation_count"], 3)
        self.assertEqual(token_summary["feature_dim"], 2)
        self.assertLessEqual(token_summary["effective_rank_normalized"], 1.0)

        conv = tokens.transpose(1, 2)  # [B, C, L]
        conv_summary = lop_metrics.spectral_summary(conv, feature_axis=1)
        self.assertEqual(conv_summary["observation_count"], 3)
        self.assertEqual(conv_summary["feature_dim"], 2)

    def test_activation_std_uses_mean_not_mean_absolute(self):
        values = torch.tensor([-1.0, 0.0, 1.0])
        summary = lop_metrics.activation_summary(values, kind="ReLU")
        self.assertAlmostEqual(summary["mean"], 0.0, places=7)
        self.assertAlmostEqual(summary["std"], (2.0 / 3.0) ** 0.5, places=7)
        self.assertAlmostEqual(summary["dead_fraction"], 1.0 / 3.0, places=7)

    def test_cka_is_invariant_to_feature_rotation(self):
        left = torch.randn(12, 4, generator=torch.Generator().manual_seed(3))
        rotation, _ = torch.linalg.qr(torch.randn(4, 4, generator=torch.Generator().manual_seed(4)))
        right = left @ rotation
        score = lop_metrics.linear_cka(left, right)
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, 1.0, places=5)
        residual = lop_metrics.procrustes_residual(left, right)
        self.assertIsNotNone(residual)
        self.assertLess(residual, 1e-5)

    def test_attention_records_normalization_axis_and_head_diversity(self):
        attention = torch.full((2, 2, 3, 3), 1.0 / 3.0)
        summary = lop_metrics.attention_summary(attention, normalization_axis=-1)
        self.assertEqual(summary["normalization_axis"], 3)
        self.assertEqual(summary["normalization_length"], 3)
        self.assertAlmostEqual(summary["entropy_normalized_mean"], 1.0, places=6)
        self.assertIn("head_diversity", summary)

    def test_sampled_jacobian_has_ntk_summary(self):
        model = nn.Linear(2, 1, bias=False)
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        jacobian = lop_metrics.sampled_parameter_jacobian(model, inputs, max_samples=3)
        self.assertEqual(tuple(jacobian.shape), (3, 2))
        summary = lop_metrics.jacobian_summary(jacobian)
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["parameter_dim"], 2)
        self.assertIn("ntk_trace", summary)

    def test_local_linearity_is_deterministic_for_same_seed(self):
        model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
        inputs = torch.tensor([[0.2, -0.1], [0.5, 0.3]])
        first = lop_metrics.local_linearity_summary(model, inputs, epsilons=(1e-3,), directions=2, seed=19)
        second = lop_metrics.local_linearity_summary(model, inputs, epsilons=(1e-3,), directions=2, seed=19)
        self.assertEqual(first, second)

    def test_fixed_budget_probe_emits_fresh_gap(self):
        torch.manual_seed(12)
        warm = nn.Linear(2, 2)
        fresh = nn.Linear(2, 2)
        train = [(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))]
        evaluation = train
        result = lop_metrics.fixed_budget_probe(warm, fresh, train, evaluation, steps=(0, 1), lr=0.01, seed=21)
        self.assertIn("fresh_gap_final", result["outcome"])
        self.assertEqual(result["steps"], [0, 1])


if __name__ == "__main__":
    unittest.main()
