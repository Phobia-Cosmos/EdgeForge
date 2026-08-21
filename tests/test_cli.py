import unittest

from edgeforge.cli import build_parser


class CLITests(unittest.TestCase):
    def test_capability_gate_cli_preserves_explicit_operator_actions(self):
        parser = build_parser()
        register = parser.parse_args(
            [
                "model-register",
                "--name",
                "BrainUICL",
                "--workload",
                "raeeg",
                "--protocol",
                "eeg-cl-v1-aligned",
                "--comparison-group",
                "aligned-full49",
                "--source-experiment-id",
                "brainuicl-exp",
            ]
        )
        self.assertEqual(register.comparison_group, "aligned-full49")
        promote = parser.parse_args(["model-promote", "brainuicl", "--reason", "approved"])
        self.assertEqual(promote.model_id, "brainuicl")
        rollback = parser.parse_args(
            ["model-rollback", "si", "--target-model-id", "brainuicl", "--reason", "regression"]
        )
        self.assertEqual(rollback.target_model_id, "brainuicl")

    def test_experiment_cli_accepts_hardware_constraints(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "experiment-run",
                "--spec",
                "spec.json",
                "--worker-id",
                "worker-4070s",
                "--accelerator",
                "nvidia-gpu",
            ]
        )
        self.assertEqual(args.worker_id, ["worker-4070s"])
        self.assertEqual(args.accelerator, ["nvidia-gpu"])

    def test_kernel_register_uses_only_explicit_dtypes(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "kernel-register",
                "--operator",
                "matmul",
                "--backend",
                "triton",
                "--version",
                "1",
                "--dtype",
                "fp16",
            ]
        )
        self.assertEqual(args.dtype, ["fp16"])

    def test_kernel_register_can_apply_fp32_default_after_parsing(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "kernel-register",
                "--operator",
                "softmax",
                "--backend",
                "python-reference",
                "--version",
                "1",
            ]
        )
        self.assertEqual(args.dtype or ["fp32"], ["fp32"])

    def test_target_probe_accepts_output_manifest_path(self):
        parser = build_parser()
        args = parser.parse_args(["target-probe", "--output", "probe.json"])
        self.assertEqual(args.output, "probe.json")

    def test_model_regressions_accepts_backend_filters(self):
        parser = build_parser()
        args = parser.parse_args(["model-regressions", "--backend", "torch-compile", "--threshold", "0.3"])
        self.assertEqual(args.backend, "torch-compile")
        self.assertEqual(args.threshold, 0.3)

    def test_lop_analyze_defaults_to_lagged_relation(self):
        parser = build_parser()
        args = parser.parse_args(["lop-analyze", "--experiment-id", "exp-1"])
        self.assertEqual(args.lag, 1)
        self.assertEqual(args.minimum_seeds, 3)

    def test_lop_audit_accepts_local_catalog_and_summary(self):
        parser = build_parser()
        args = parser.parse_args(["lop-audit", "--catalog", "catalog.json", "--method", "ewc", "--output", "audit.json", "--summary"])
        self.assertEqual(args.catalog, "catalog.json")
        self.assertEqual(args.method, ["ewc"])
        self.assertEqual(args.output, "audit.json")
        self.assertTrue(args.summary)


if __name__ == "__main__":
    unittest.main()
