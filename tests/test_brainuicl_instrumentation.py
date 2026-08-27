import importlib.util
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # Keep the dependency-free EdgeForge test suite usable.
    torch = None


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "brainuicl-instrumentation.py"
SPEC = importlib.util.spec_from_file_location("edgeforge_brainuicl_instrumentation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
instrumentation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(instrumentation)


@unittest.skipUnless(torch is not None, "PyTorch is required for instrumentation diagnostics")
class BrainUICLInstrumentationTests(unittest.TestCase):
    def test_auxiliary_checkpoint_files_are_digested_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "regularizer_state.pt"
            state.write_bytes(b"opaque-state")
            (root / "feature_encoder_parameter_4321.pkl").write_bytes(b"model")
            rows = instrumentation._auxiliary_checkpoint_files(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["name"], "regularizer_state.pt")
            self.assertEqual(rows[0]["size_bytes"], len(b"opaque-state"))

    def test_optimizer_state_provenance_is_explicit_and_does_not_load_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "regularizer_state.pt").write_bytes(b"regularizer")
            (root / "optimizer_state.pt").write_bytes(b"opaque-optimizer")
            provenance = instrumentation._optimizer_state_provenance(root)
            self.assertEqual(provenance["status"], "available")
            self.assertFalse(provenance["loaded"])
            self.assertEqual(provenance["files"][0]["name"], "optimizer_state.pt")
            self.assertFalse(provenance["files"][0]["loaded"])

    def test_optimizer_state_provenance_marks_parameter_only_checkpoint_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "feature_encoder_parameter_4321.pkl").write_bytes(b"model")
            provenance = instrumentation._optimizer_state_provenance(root)
            self.assertEqual(provenance["status"], "unavailable")
            self.assertIn("no serialized optimizer", provenance["reason"])

    def test_parameter_update_summary_reports_zero_for_same_checkpoint(self):
        first = [torch.nn.Linear(3, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)]
        second = [torch.nn.Linear(3, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)]
        for left, right in zip(first, second):
            right.load_state_dict(left.state_dict())
        summary = instrumentation.parameter_update_summary(first, second)
        self.assertEqual(summary["status"], "computed")
        self.assertAlmostEqual(summary["global_delta_l2"], 0.0, places=7)
        self.assertAlmostEqual(summary["global_relative_update"], 0.0, places=7)

    def test_parameter_update_summary_exposes_block_and_parameter_movement(self):
        first = [torch.nn.Linear(3, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)]
        second = [torch.nn.Linear(3, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)]
        for left, right in zip(first, second):
            right.load_state_dict(left.state_dict())
        with torch.no_grad():
            second[1].weight.add_(1.0)
        summary = instrumentation.parameter_update_summary(second, first)
        self.assertGreater(summary["global_delta_l2"], 0.0)
        self.assertGreater(summary["by_block"]["feature_encoder"]["delta_l2"], 0.0)
        self.assertEqual(summary["top_parameters"][0]["name"], "feature_encoder.weight")
        self.assertLessEqual(summary["top_parameters"][0]["cosine_to_reference"], 1.0)

    def test_shared_lop_metrics_uses_explicit_token_feature_axis(self):
        representations = {
            "fusion": torch.randn(2, 20, 8),
            "transformer_1": torch.randn(2, 20, 8),
            "classifier_input": torch.randn(2, 20, 4),
        }
        blocks = [torch.nn.Linear(3, 2), torch.nn.LayerNorm(2), torch.nn.Linear(2, 1)]
        result = instrumentation.shared_lop_metrics(
            representations,
            blocks=blocks,
            max_observations=16,
        )
        self.assertEqual(result["protocol"], "edgeforge-lop-metrics-v1")
        self.assertEqual(result["representation_layout"], "[batch, sequence, feature]")
        self.assertEqual(result["layers"]["transformer_1"]["feature_axis"], -1)
        self.assertEqual(result["layers"]["transformer_1"]["spectrum"]["max_observations"], 16)
        self.assertEqual(result["parameters"]["feature_extractor"]["status"], "computed")


if __name__ == "__main__":
    unittest.main()
