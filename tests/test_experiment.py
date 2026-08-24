import unittest

from edgeforge.experiment import ExperimentSpec, normalize_raeeg_metrics


def valid_spec():
    return {
        "schema_version": 1,
        "experiment_id": "isruc-finetune-seed4321",
        "workload": "raeeg",
        "dataset": {"name": "ISRUC", "manifest_digest": "abc"},
        "model": {"name": "BrainUICL"},
        "protocol": "eeg-cl-v1",
        "method": "finetune",
        "seed": 4321,
        "runner": {
            "mode": "import",
            "result_path": "results/metrics.json",
            "adapter": "raeeg-metrics-v1",
        },
    }


class ExperimentContractTests(unittest.TestCase):
    def test_spec_requires_explicit_research_identity(self):
        spec = ExperimentSpec.from_payload(valid_spec())
        self.assertEqual(spec.dataset["name"], "ISRUC")
        broken = valid_spec()
        broken["seed"] = "4321"
        with self.assertRaisesRegex(ValueError, "seed"):
            ExperimentSpec.from_payload(broken)

    def test_raeeg_normalizer_preserves_summary_and_task_plasticity(self):
        metrics, summary = normalize_raeeg_metrics(
            {
                "summary": {"final_old_acc": 0.71, "bwt_acc": -0.01},
                "tasks": [
                    {
                        "task": 1,
                        "subject": 64,
                        "current_before": {"acc": 0.60, "mf1": 0.50},
                        "current_after": {"acc": 0.65, "mf1": 0.52},
                        "spectra": {"transformer_1": {"effective_rank": 12.5}},
                    }
                ],
            }
        )
        self.assertEqual(summary["final_old_acc"], 0.71)
        by_name = {(item["name"], item["step"]): item for item in metrics}
        self.assertAlmostEqual(by_name[("plasticity.acc_gain", 1)]["value"], 0.05)
        self.assertEqual(by_name[("task.spectra.transformer_1.effective_rank", 1)]["value"], 12.5)
        self.assertEqual(by_name[("plasticity.acc_gain", 1)]["context"], {"subject": "64"})

    def test_normalizer_preserves_probe_curve_steps_and_legacy_transformer_alias(self):
        metrics, _summary = normalize_raeeg_metrics(
            {
                "tasks": [
                    {
                        "stage": 10,
                        "subject": 1,
                        "split": "fresh-test",
                        "spectra": {"transformer": {"effective_rank": 6.0}},
                        "plasticity": {"curve": [
                            {"step": 0, "acc": 0.20, "loss": 1.5},
                            {"step": 4, "acc": 0.40, "loss": 1.0},
                        ]},
                    }
                ]
            }
        )
        by_key = {(item["name"], item["step"]): item for item in metrics}
        self.assertEqual(by_key[("task.spectra.transformer_1.effective_rank", 10)]["value"], 6.0)
        self.assertEqual(by_key[("task.plasticity.curve.acc", 0)]["value"], 0.20)
        self.assertEqual(by_key[("task.plasticity.curve.acc", 4)]["value"], 0.40)
        self.assertNotIn("checkpoint_stage", by_key[("task.spectra.transformer_1.effective_rank", 10)]["context"])

    def test_normalizer_marks_retention_metrics_as_separate_role(self):
        metrics, _summary = normalize_raeeg_metrics({
            "tasks": [{
                "stage": 2,
                "forgetting": {"checkpoint_acc_drop": 0.12},
                "probe": {"retention": {"checkpoint_mf1_drop": 0.08}},
            }]
        })
        roles = {
            item["name"]: item.get("context", {}).get("metric_role")
            for item in metrics
            if "forgetting" in item["name"] or "probe.retention" in item["name"]
        }
        self.assertEqual(roles["task.forgetting.checkpoint_acc_drop"], "retention")
        self.assertEqual(roles["task.probe.retention.checkpoint_mf1_drop"], "retention")


if __name__ == "__main__":
    unittest.main()
