import unittest
import hashlib
import tempfile
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:  # Keep the dependency-free test suite usable.
    torch = None
    nn = None

from edgeforge.lop_diagnostics import (
    LoPDiagnostics,
    MetricConfig,
    diagnose_model,
    hessian_vector_product,
    hessian_top_eigenvalue_summary,
    hutchinson_trace_summary,
    empirical_fisher_summary,
    objective_loss,
    calibration_manifest_digest,
    calibration_manifest_provenance,
)


@unittest.skipUnless(torch is not None, "PyTorch is required for curvature diagnostics")
class LopDiagnosticsTests(unittest.TestCase):
    def test_calibration_manifest_provenance_hashes_exact_bytes(self):
        payload = b'{"subject": 1, "split": "calibration"}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(calibration_manifest_digest(path), expected)
            record = calibration_manifest_provenance(path, declared_digest=f"sha256:{expected}")
        self.assertEqual(record["status"], "present")
        self.assertEqual(record["sha256"], expected)
        self.assertTrue(record["matches_declared"])

    def test_missing_calibration_manifest_is_explicit_not_an_exception(self):
        record = calibration_manifest_provenance("/definitely/missing/manifest.json")
        self.assertEqual(record["status"], "missing")
        self.assertIsNone(record["sha256"])

    def test_inline_manifest_bytes_are_not_serialized_raw(self):
        config = MetricConfig(calibration_manifest=b"secret-free-manifest")
        encoded = config.to_dict()["calibration_manifest"]
        self.assertIsInstance(encoded, dict)
        self.assertNotIn("secret-free-manifest", str(encoded))
        self.assertEqual(encoded["bytes"], len(b"secret-free-manifest"))

    def test_hvp_matches_scalar_quadratic(self):
        # For L(w) = 0.5 * 4 * w^2 the Hessian is exactly 4.
        model = nn.Linear(1, 1, bias=False)
        x = torch.ones(1, 1)

        def loss_fn(logits, _labels):
            return 2.0 * logits.square().mean()

        hvp = hessian_vector_product(model, (x, torch.zeros_like(x)), torch.tensor([1.0]), loss_fn=loss_fn)
        self.assertAlmostEqual(float(hvp.item()), 4.0, places=5)

    def test_nested_manifest_config_is_normalized(self):
        config = MetricConfig.from_mapping(
            {
                "objective": {"name": "cross_entropy", "label_source": "ground_truth"},
                "hessian": {"mode": "power", "iterations": 3},
                "batchnorm": {"freeze_running_stats": False},
            }
        )
        self.assertEqual(config.objective, "cross_entropy")
        self.assertEqual(config.label_source, "true")
        self.assertEqual(config.hessian_mode, "power")
        self.assertEqual(config.hessian_iterations, 3)
        self.assertFalse(config.freeze_batch_norm)

    def test_power_and_hutchinson_estimates_are_data_dependent(self):
        model = nn.Linear(1, 1, bias=False)
        x = torch.ones(8, 1)

        def loss_fn(logits, _labels):
            return 2.0 * logits.square().mean()

        batches = (x, torch.zeros_like(x))
        eigen = hessian_top_eigenvalue_summary(model, batches, loss_fn=loss_fn, iterations=12, tolerance=1e-8)
        trace = hutchinson_trace_summary(model, batches, loss_fn=loss_fn, probes=4)
        self.assertAlmostEqual(eigen["eigenvalue"], 4.0, places=4)
        self.assertAlmostEqual(trace["trace"], 4.0, places=4)
        self.assertTrue(eigen["data_dependent"])
        self.assertTrue(trace["data_dependent"])

    def test_empirical_fisher_uses_per_sample_gradients(self):
        model = nn.Linear(2, 2)
        x = torch.randn(3, 2)
        y = torch.tensor([0, 1, 0])
        summary = empirical_fisher_summary(model, (x, y), max_samples=3)
        self.assertEqual(summary["sample_count"], 3)
        self.assertTrue(summary["data_dependent"])
        self.assertEqual(summary["method"], "per_sample_gradient_square_diagonal")

    def test_public_objective_supports_pseudo_and_unlabeled_paths(self):
        model = nn.Linear(2, 2)
        logits = model(torch.ones(3, 2))
        pseudo = objective_loss(
            model,
            (torch.ones(3, 2), None),
            logits,
            None,
            objective="cross_entropy",
            label_source="pseudo",
        )
        unlabeled = objective_loss(
            model,
            (torch.ones(3, 2), None),
            logits,
            None,
            objective="output_mean",
            label_source="none",
        )
        self.assertTrue(torch.isfinite(pseudo))
        self.assertTrue(torch.isfinite(unlabeled))

    def test_diagnose_model_consumes_forward_bundle_representations(self):
        class BundleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2, 3)
                self.classifier = nn.Linear(3, 2)

            def representation_specs(self):
                return {"embedding": {"feature_axis": -1, "kind": "linear"}}

            def forward_bundle(self, values):
                embedding = self.layer(values)
                return type("Bundle", (), {"logits": self.classifier(embedding), "representations": {"embedding": embedding}})()

            def forward(self, values):
                return self.forward_bundle(values).logits

        model = BundleModel()
        x = torch.randn(4, 2)
        y = torch.tensor([0, 1, 0, 1])
        report = diagnose_model(
            model,
            (x, y),
            config=MetricConfig(jacobian_samples=2, hessian_probes=1, hessian_iterations=1),
        )
        self.assertIn("embedding", report["representations"])
        self.assertEqual(report["representations"]["embedding"]["feature_axis"], -1)
        self.assertEqual(report["hessian"]["status"], "computed")

    def test_configured_runner_is_reusable(self):
        model = nn.Linear(2, 2)
        x = torch.randn(3, 2)
        y = torch.tensor([0, 1, 0])
        runner = LoPDiagnostics(MetricConfig(include_representations=False, hessian_probes=1, hessian_iterations=1))
        report = runner(model, (x, y))
        self.assertEqual(report["status"], "computed")
        self.assertEqual(report["data"]["calibration_manifest"]["status"], "not-provided")

    def test_list_form_multi_input_batch_is_not_split_into_two_batches(self):
        class TwoInput(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, inputs):
                left, right = inputs
                return self.linear(torch.cat((left, right), dim=-1))

        model = TwoInput()
        left = torch.randn(3, 1)
        right = torch.randn(3, 1)
        labels = torch.tensor([0, 1, 0])
        # A list is a common result of lightweight collators.  The pair must
        # remain one multi-input batch rather than becoming two iterations.
        inputs, recovered = __import__("edgeforge.lop_diagnostics", fromlist=["split_batch"]).split_batch([left, right])
        self.assertIsNone(recovered)
        self.assertEqual(len(inputs), 2)
        report = diagnose_model(
            model,
            ([left, right], labels),
            config=MetricConfig(include_representations=False, hessian_mode="none", jacobian_samples=1),
        )
        self.assertEqual(report["status"], "computed")

    def test_report_records_attention_and_restores_batch_norm_mode(self):
        from edgeforge.eeg_models import EEGModelConfig, build_eeg_decoder

        model = build_eeg_decoder(
            EEGModelConfig(
                name="brainuicl",
                in_channels=3,
                input_length=16,
                num_classes=2,
                feature_dim=8,
                heads=2,
                layers=1,
                options={"d_model": 8},
            )
        )
        model.train()
        batch = (torch.randn(3, 3, 16), torch.tensor([0, 1, 0]))
        report = diagnose_model(
            model,
            batch,
            config=MetricConfig(jacobian_samples=1, hessian_probes=1, hessian_iterations=1),
        )
        self.assertTrue(model.training)
        self.assertEqual(report["batch_norm"]["freeze_running_stats"], True)
        self.assertEqual(report["attention"]["normalization_axis"], 1)
        self.assertEqual(report["attention"]["status"], "computed")

    def test_sequence_conv_specs_shift_channel_axis_after_restoration(self):
        """A ``[B,T,C,S]`` bundle must measure channels, not sequence slots."""

        from edgeforge.eeg_models import EEGModelConfig, build_eeg_decoder

        model = build_eeg_decoder(
            EEGModelConfig(
                name="eegnet",
                in_channels=3,
                input_length=16,
                num_classes=2,
                feature_dim=8,
                width=6,
                kernel_size=5,
                dropout=0.0,
            )
        )
        inputs = torch.randn(2, 3, 3, 16)
        labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
        report = diagnose_model(
            model,
            (inputs, labels),
            config=MetricConfig(
                max_batches=1,
                include_jacobian=False,
                include_gradients=False,
                include_hessian=False,
            ),
        )
        temporal = report["representations"]["temporal"]
        self.assertEqual(temporal["feature_axis"], 2)
        self.assertTrue(temporal["sequence_axis_adjusted"])
        self.assertEqual(temporal["spectrum"]["feature_dim"], 6)

    def test_explicit_feature_axis_override_is_authoritative_for_sequences(self):
        from edgeforge.eeg_models import EEGModelConfig, build_eeg_decoder

        model = build_eeg_decoder(
            EEGModelConfig(
                name="tcn",
                in_channels=2,
                input_length=12,
                num_classes=2,
                feature_dim=6,
                width=4,
                kernel_size=3,
                dropout=0.0,
            )
        )
        inputs = torch.randn(2, 2, 2, 12)
        labels = torch.tensor([[0, 1], [1, 0]])
        report = diagnose_model(
            model,
            (inputs, labels),
            config=MetricConfig(
                max_batches=1,
                include_jacobian=False,
                include_gradients=False,
                include_hessian=False,
                feature_axes={"block1": 1},
            ),
        )
        block = report["representations"]["block1"]
        self.assertEqual(block["feature_axis"], 1)
        self.assertFalse(block["sequence_axis_adjusted"])
        self.assertEqual(block["axis_source"], "config")


if __name__ == "__main__":
    unittest.main()
