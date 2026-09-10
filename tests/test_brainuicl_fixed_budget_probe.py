import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "brainuicl-fixed-budget-probe.py"
SPEC = importlib.util.spec_from_file_location("edgeforge_brainuicl_fixed_budget_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class BrainUICLFixedBudgetProbeTests(unittest.TestCase):
    def test_curve_summary_uses_explicit_steps_and_retention_drop_sign(self):
        rows = [
            {"step": 0, "loss": 2.0, "acc": 0.4, "mf1": 0.3, "retention_loss": 1.0, "retention_acc": 0.8, "retention_mf1": 0.7},
            {"step": 5, "loss": 1.0, "acc": 0.6, "mf1": 0.5, "retention_loss": 1.2, "retention_acc": 0.7, "retention_mf1": 0.6},
        ]
        summary = probe._summarize_curve(rows, ("retention_loss", "retention_acc", "retention_mf1"))
        self.assertEqual(summary["retention_acc_initial"], 0.8)
        self.assertEqual(summary["retention_acc_final"], 0.7)
        self.assertAlmostEqual(summary["retention_acc_gain"], -0.1)
        self.assertAlmostEqual(summary["retention_loss_aulc"], 1.1)

    def test_parse_steps_requires_zero_and_positive_budget(self):
        self.assertEqual(probe.parse_steps("10,0,5,5"), [0, 5, 10])
        with self.assertRaises(ValueError):
            probe.parse_steps("0")

    def test_configure_reproducibility_records_deterministic_contract(self):
        result = probe.configure_reproducibility(4321)
        self.assertEqual(result["seed"], 4321)
        self.assertEqual(result["cublas_workspace_config"], ":4096:8")
        self.assertTrue(result["deterministic_algorithms"])


if __name__ == "__main__":
    unittest.main()
