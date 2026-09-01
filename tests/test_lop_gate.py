import unittest

from edgeforge.lop_analysis import DEFAULT_LOP_OUTCOME, evaluate_lop_gate


def _trajectory(seed: int, gaps=(0.2, 0.3, 0.4), *, stages=(0, 10, 25), retention=False):
    experiment_id = f"gate-seed-{seed}"
    experiment = {
        "experiment_id": experiment_id,
        "workload": "raeeg-lop",
        "dataset": {"name": "ISRUC"},
        "model": {"name": "BrainUICL", "d_model": 64},
        "protocol": "fixed-budget-v1",
        "method": "finetune",
        "seed": seed,
        "metadata": {
            "comparison_group": "aligned",
            "subject": "2",
            "split": "subject-eval",
            "task_order": [0, 1, 2],
            "probe_budget": 50,
            "optimizer": "sgd",
            "lr": 0.01,
        },
    }
    metrics = []
    for stage, gap in zip(stages, gaps):
        context = {
            "dataset": "ISRUC",
            "subject": "2",
            "split": "subject-eval",
            "probe_budget": 50,
            "measurement_protocol": "fixed-budget-v1",
            "metric_role": "outcome",
        }
        metrics.append({"name": DEFAULT_LOP_OUTCOME, "value": gap, "step": stage, "context": context})
        if retention:
            metrics.append({"name": "task.forgetting.checkpoint_acc_drop", "value": -0.1, "step": stage, "context": {**context, "metric_role": "retention"}})
    return experiment, metrics


class LopGateTests(unittest.TestCase):
    def test_positive_three_seed_trajectory_is_candidate(self):
        experiments, metrics = zip(*[_trajectory(seed) for seed in (1, 2, 3)])
        result = evaluate_lop_gate(list(experiments), {item["experiment_id"]: values for item, values in zip(experiments, metrics)}, bootstrap_repeats=100)
        self.assertEqual(result["status"], "candidate")
        self.assertTrue(result["direction_supported"])
        self.assertEqual(result["positive_stage_count"], 3)
        self.assertFalse(result["scientific_conclusion_allowed"])

    def test_two_seed_evidence_is_insufficient(self):
        experiments, metrics = zip(*[_trajectory(seed) for seed in (1, 2)])
        result = evaluate_lop_gate(list(experiments), {item["experiment_id"]: values for item, values in zip(experiments, metrics)}, bootstrap_repeats=100)
        self.assertEqual(result["status"], "insufficient-seeds")

    def test_stage_grid_mismatch_is_blocked(self):
        first, first_metrics = _trajectory(1)
        second, second_metrics = _trajectory(2, stages=(0, 10, 49))
        third, third_metrics = _trajectory(3)
        result = evaluate_lop_gate(
            [first, second, third],
            {first["experiment_id"]: first_metrics, second["experiment_id"]: second_metrics, third["experiment_id"]: third_metrics},
            bootstrap_repeats=100,
        )
        self.assertEqual(result["status"], "blocked-incomparable-stages")
        self.assertFalse(result["stages_consistent"])

    def test_mixed_direction_is_blocked_and_retention_is_separate(self):
        first, first_metrics = _trajectory(1)
        second, second_metrics = _trajectory(2, gaps=(0.2, -0.1, 0.3), retention=True)
        third, third_metrics = _trajectory(3)
        result = evaluate_lop_gate(
            [first, second, third],
            {first["experiment_id"]: first_metrics, second["experiment_id"]: second_metrics, third["experiment_id"]: third_metrics},
            bootstrap_repeats=100,
        )
        self.assertEqual(result["status"], "blocked-inconsistent-direction")
        self.assertTrue(result["retention_separate"])
        self.assertGreater(len(result["retention_inventory"]), 0)

    def test_retention_metric_cannot_be_requested_as_primary_outcome(self):
        experiments, metrics = zip(*[_trajectory(seed, retention=True) for seed in (1, 2, 3)])
        result = evaluate_lop_gate(
            list(experiments),
            {item["experiment_id"]: values for item, values in zip(experiments, metrics)},
            outcome="task.forgetting.checkpoint_acc_drop",
            bootstrap_repeats=100,
        )
        self.assertEqual(result["status"], "blocked-invalid-outcome")
        self.assertEqual(result["outcome_role"], "invalid")


if __name__ == "__main__":
    unittest.main()
