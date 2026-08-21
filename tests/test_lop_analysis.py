import unittest

from edgeforge.lop_analysis import analyze_lop


def _fixture(seed: int, *, group: str = "lop-group"):
    experiment_id = f"lop-seed-{seed}"
    metrics = []
    for step in range(4):
        metrics.extend(
            [
                {"name": "task.spectra.transformer_1.effective_rank", "value": 10.0 - step + seed * 0.01, "step": step, "context": {}},
                {"name": "plasticity.acc_gain", "value": 0.01 * step + seed * 0.001, "step": step, "context": {}},
            ]
        )
    return (
        {"experiment_id": experiment_id, "workload": "raeeg-lop", "protocol": "lop-v1", "method": "probe", "seed": seed, "spec": {"metadata": {"comparison_group": group}}},
        metrics,
    )


class LopAnalysisTests(unittest.TestCase):
    def test_lagged_analysis_is_deterministic_and_reports_ci(self):
        experiments, metrics = zip(*[_fixture(seed) for seed in (1, 2, 3)])
        result = analyze_lop(list(experiments), {item["experiment_id"]: values for item, values in zip(experiments, metrics)}, bootstrap_repeats=200, minimum_pairs=3, minimum_seeds=3)
        again = analyze_lop(list(experiments), {item["experiment_id"]: values for item, values in zip(experiments, metrics)}, bootstrap_repeats=200, minimum_pairs=3, minimum_seeds=3)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["pair_count"], 9)
        self.assertEqual(result["analysis_digest"], again["analysis_digest"])
        self.assertEqual(len(result["statistics"]["pearson_ci95"]), 2)
        self.assertFalse(result["scientific_conclusion_allowed"])

    def test_insufficient_seed_and_mixed_scope_are_blocked(self):
        first, first_metrics = _fixture(1)
        second, second_metrics = _fixture(2, group="other")
        result = analyze_lop([first, second], {first["experiment_id"]: first_metrics, second["experiment_id"]: second_metrics}, minimum_pairs=2, minimum_seeds=3)
        self.assertEqual(result["status"], "blocked-incomparable-scope")
        self.assertFalse(result["scope_consistent"])
        self.assertTrue(any("workload" in reason for reason in result["reasons"]))

    def test_method_and_checkpoint_transitions_must_be_comparable(self):
        first, first_metrics = _fixture(1)
        second, second_metrics = _fixture(2)
        second["method"] = "other-probe"
        method_result = analyze_lop(
            [first, second],
            {first["experiment_id"]: first_metrics, second["experiment_id"]: second_metrics},
            minimum_pairs=2,
        )
        self.assertEqual(method_result["status"], "blocked-incomparable-scope")

        second["method"] = "probe"
        second_metrics = [metric for metric in second_metrics if metric["step"] != 2]
        stage_result = analyze_lop(
            [first, second],
            {first["experiment_id"]: first_metrics, second["experiment_id"]: second_metrics},
            minimum_pairs=2,
        )
        self.assertEqual(stage_result["status"], "blocked-incomparable-stages")
        self.assertFalse(stage_result["stages_consistent"])

    def test_lag_uses_ordered_non_contiguous_checkpoint_stages(self):
        experiment, metrics = _fixture(1)
        stage_map = {0: 0, 1: 10, 2: 25, 3: 40}
        for metric in metrics:
            metric["step"] = stage_map[metric["step"]]
        result = analyze_lop(
            [experiment],
            {experiment["experiment_id"]: metrics},
            minimum_pairs=2,
            minimum_seeds=1,
        )
        transitions = [
            (pair["predictor_step"], pair["outcome_step"])
            for pair in result["pairs"]
        ]
        self.assertEqual(transitions, [(0, 10), (10, 25), (25, 40)])
        self.assertEqual(result["minimum_seeds"], 3)
        self.assertEqual(result["status"], "insufficient-seeds")

    def test_duplicate_seed_runs_are_not_independent_evidence(self):
        first, first_metrics = _fixture(1)
        second, second_metrics = _fixture(1)
        second["experiment_id"] = "lop-seed-1-rerun"
        result = analyze_lop(
            [first, second],
            {first["experiment_id"]: first_metrics, second["experiment_id"]: second_metrics},
            minimum_pairs=2,
        )
        self.assertEqual(result["status"], "blocked-duplicate-seeds")
        self.assertFalse(result["seeds_unique"])
        self.assertEqual(result["seed_count"], 1)

    def test_missing_pairs_and_constant_metrics_cannot_report_ok(self):
        fixtures = [_fixture(seed) for seed in (1, 2, 3)]
        experiments = [item[0] for item in fixtures]
        metrics_by_experiment = {
            item[0]["experiment_id"]: item[1]
            for item in fixtures
        }
        metrics_by_experiment[experiments[-1]["experiment_id"]] = []
        incomplete = analyze_lop(
            experiments,
            metrics_by_experiment,
            minimum_pairs=2,
        )
        self.assertEqual(incomplete["status"], "blocked-incomplete-evidence")

        for metrics in metrics_by_experiment.values():
            for metric in metrics:
                metric["value"] = 1.0
        metrics_by_experiment[experiments[-1]["experiment_id"]] = [
            {"name": "task.spectra.transformer_1.effective_rank", "value": 1.0, "step": step, "context": {}}
            for step in range(4)
        ] + [
            {"name": "plasticity.acc_gain", "value": 1.0, "step": step, "context": {}}
            for step in range(4)
        ]
        constant = analyze_lop(
            experiments,
            metrics_by_experiment,
            minimum_pairs=2,
        )
        self.assertEqual(constant["status"], "insufficient-variation")
        self.assertFalse(constant["variation_sufficient"])
        self.assertIsNone(constant["statistics"]["pearson"])

    def test_exact_context_does_not_cross_subjects(self):
        experiment, _ = _fixture(1)
        metrics = [
            {"name": "task.spectra.transformer_1.effective_rank", "value": 4, "step": 0, "context": {"subject": "a"}},
            {"name": "plasticity.acc_gain", "value": 0.1, "step": 1, "context": {"subject": "a"}},
            {"name": "task.spectra.transformer_1.effective_rank", "value": 8, "step": 0, "context": {"subject": "b"}},
            {"name": "plasticity.acc_gain", "value": 0.2, "step": 1, "context": {"subject": "c"}},
        ]
        result = analyze_lop([experiment], {experiment["experiment_id"]: metrics}, context_policy="exact", minimum_pairs=2, minimum_seeds=1)
        self.assertEqual(result["pair_count"], 1)

    def test_exact_context_grid_must_match_across_seeds(self):
        first, _ = _fixture(1)
        second, _ = _fixture(2)
        metrics = {
            first["experiment_id"]: [
                {"name": "task.spectra.transformer_1.effective_rank", "value": 4, "step": 0, "context": {"subject": "a"}},
                {"name": "plasticity.acc_gain", "value": 0.1, "step": 1, "context": {"subject": "a"}},
            ],
            second["experiment_id"]: [
                {"name": "task.spectra.transformer_1.effective_rank", "value": 5, "step": 0, "context": {"subject": "b"}},
                {"name": "plasticity.acc_gain", "value": 0.2, "step": 1, "context": {"subject": "b"}},
            ],
        }
        result = analyze_lop(
            [first, second],
            metrics,
            context_policy="exact",
            minimum_pairs=2,
        )
        self.assertEqual(result["status"], "blocked-incomparable-contexts")
        self.assertFalse(result["contexts_consistent"])


if __name__ == "__main__":
    unittest.main()
