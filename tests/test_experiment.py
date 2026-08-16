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


if __name__ == "__main__":
    unittest.main()
