import unittest

from edgeforge.lop_envelope import diagnostic_to_metrics, edgeforge_bundle_from_diagnostic


class LoPEnvelopeTests(unittest.TestCase):
    def test_generic_runs_emit_roles_and_lagged_probe_step(self):
        result = {
            "protocol": "generic",
            "config": {"data": "synthetic-eeg"},
            "runs": [
                {
                    "architecture": "transformer",
                    "stages": [
                        {
                            "stage": 0,
                            "metrics": {
                                "layers": {
                                    "encoder.0": {
                                        "spectrum": {"effective_rank": 3.5, "stable_rank": 2.0},
                                        "activation": {"near_zero_fraction": 0.1},
                                    }
                                },
                                "jacobian": {"ntk_trace": 4.0},
                            },
                        }
                    ],
                    "probes": [
                        {
                            "after_stage": 0,
                            "next_task": 1,
                            "outcome": {"fresh_gap_final": 0.2, "warm_acc_gain": 0.1},
                            "curves": {
                                "warm": [{"step": 0, "accuracy": 0.3}],
                                "fresh": [{"step": 0, "accuracy": 0.4}],
                            },
                        }
                    ],
                }
            ],
        }
        metrics = diagnostic_to_metrics(result)
        by_name = {(item["name"], item["step"]): item for item in metrics}
        predictor = by_name[("task.spectra.encoder.0.effective_rank", 0)]
        alias = by_name[("task.spectra.transformer_1.effective_rank", 0)]
        self.assertEqual(predictor["context"]["metric_role"], "predictor")
        self.assertEqual(alias["context"]["metric_role"], "predictor")
        self.assertEqual(by_name[("plasticity.fresh_gap_final", 1)]["context"]["metric_role"], "outcome")
        self.assertEqual(by_name[("task.jacobian.ntk_trace", 0)]["context"]["metric_role"], "diagnostic")
        self.assertEqual(by_name[("task.probe.curves.warm.accuracy", 0)]["context"]["source_stage"], 0)

    def test_brainuicl_task_shape_marks_retention_separately(self):
        result = {
            "protocol": "brainuicl",
            "config": {"dataset": "ISRUC", "method": "finetune"},
            "tasks": [
                {
                    "stage": 10,
                    "subject": 2,
                    "spectra": {"transformer": {"effective_rank": 5.0}},
                    "plasticity": {"outcome": {"fresh_gap_final": 0.1}},
                    "retention": {"mean_accuracy": 0.8},
                }
            ],
        }
        metrics = diagnostic_to_metrics(result)
        self.assertEqual(
            next(item for item in metrics if item["name"] == "task.spectra.transformer.effective_rank")["context"]["metric_role"],
            "predictor",
        )
        self.assertEqual(
            next(item for item in metrics if item["name"] == "task.retention.mean_accuracy")["context"]["metric_role"],
            "retention",
        )

    def test_bundle_wrapper_is_directly_importable(self):
        bundle = edgeforge_bundle_from_diagnostic(
            {"config": {"data": "synthetic-eeg"}, "tasks": []},
            experiment_id="eeg-smoke",
        )
        self.assertEqual(bundle["schema_version"], 1)
        self.assertEqual(bundle["experiment_id"], "eeg-smoke")
        self.assertIsInstance(bundle["metrics"], list)
        self.assertFalse(bundle["scientific_conclusion_allowed"])

    def test_empty_placeholder_metrics_does_not_hide_trajectory(self):
        metrics = diagnostic_to_metrics(
            {
                "metrics": [],
                "tasks": [
                    {
                        "stage": 0,
                        "spectra": {"transformer": {"effective_rank": 2.0}},
                    }
                ],
            }
        )
        self.assertTrue(any(item["name"] == "task.spectra.transformer.effective_rank" for item in metrics))


if __name__ == "__main__":
    unittest.main()
