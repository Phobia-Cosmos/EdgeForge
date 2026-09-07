import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze-eeg-lop-full-experiment.py"
SPEC = importlib.util.spec_from_file_location("edgeforge_test_eeg_lop_full_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _plan():
    return {
        "plan_digest": "plan-1",
        "roles": {"source": [1], "target": [2, 3], "retention": [4]},
        "profiles": [
            {"subject": 1, "class_counts": [50, 30, 20, 0, 0]},
            {"subject": 2, "class_counts": [20, 20, 20, 20, 20]},
            {"subject": 3, "class_counts": [20, 20, 20, 20, 20]},
            {"subject": 4, "class_counts": [20, 20, 20, 20, 20]},
        ],
        "orders": {"random-a": [2, 3], "random-b": [3, 2]},
    }


def _config():
    return {
        "experiment_version": "test",
        "phases": {
            "confirmatory": {
                "architectures": ["transformer"],
                "seeds": [11, 12],
                "orders": ["random-a", "random-b"],
                "target_limit": None,
                "source_epochs": 3,
                "budgets": [0, 5],
                "batch_size": 8,
                "source_lr": 0.002,
                "adapt_lr": 0.001,
                "source_eval_fraction": 0.2,
                "retention_max_samples": 100,
            }
        },
        "acceptance": {
            "minimum_independent_seeds": 2,
            "minimum_target_stages": 2,
            "source_accuracy_above_majority": 0.1,
            "fresh_learning_accuracy_gain": 0.1,
            "bootstrap_repeats": 100,
            "alpha": 0.05,
        },
    }


def _calibration_config():
    config = _config()
    config["phases"] = {
        "calibration": {
            "architectures": ["transformer"],
            "seeds": [11, 12],
            "orders": ["random-a"],
            "target_limit": None,
            "source_epochs": 3,
            "budgets": [0, 5],
            "batch_size": 8,
            "source_lr": 0.002,
            "adapt_lr": 0.001,
            "source_eval_fraction": 0.2,
            "retention_max_samples": 100,
        }
    }
    return config


def _metadata(order):
    return {
        "schema": "edgeforge.eeg-continuous-architecture-lop.v1",
        "source_subjects": [1],
        "target_subject_order": _plan()["orders"][order],
        "retention_subjects": [4],
        "budgets": [0, 5],
        "source_epochs": 3,
        "batch_size": 8,
        "source_lr": 0.002,
        "adapt_lr": 0.001,
        "source_eval_fraction": 0.2,
        "retention_max_samples": 100,
    }


def _summary(order, *, weak_stage=None, omit_seed=None, corrupt_gap=None):
    runs = []
    subjects = _plan()["orders"][order]
    for seed in (11, 12):
        if seed == omit_seed:
            continue
        stages = []
        for stage, subject in enumerate(subjects):
            initial = 0.20 + 0.01 * stage
            final = initial + (0.05 if weak_stage == (order, seed, stage) else 0.20)
            warm_initial = initial - 0.02 - 0.005 * (seed - 11)
            warm_final = final - 0.10 - (0.01 if order == "random-a" else 0.0)
            gaps = [
                {"step": 0, "fresh_gap": initial - warm_initial},
                {"step": 5, "fresh_gap": final - warm_final},
            ]
            if corrupt_gap == (order, seed, stage):
                gaps[-1]["fresh_gap"] += 0.5
            stages.append({
                "stage": stage,
                "subject": subject,
                "fresh": {"curve": [{"step": 0, "accuracy": initial}, {"step": 5, "accuracy": final}]},
                "warm": {"curve": [{"step": 0, "accuracy": warm_initial}, {"step": 5, "accuracy": warm_final}]},
                "gaps": gaps,
            })
        runs.append({"status": "complete", "architecture": "transformer", "seed": seed, "source": {"accuracy": 0.75}, "stages": stages})
    return {"metadata": _metadata(order), "architectures": ["transformer"], "seeds": [11, 12], "planned_runs": 2, "completed_runs": len(runs), "runs": runs}


def _write_inputs(root, **kwargs):
    paths = []
    for order in ("random-a", "random-b"):
        path = root / order / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_summary(order, **kwargs)), encoding="utf-8")
        paths.append((order, path))
    return paths


class FullLoPAnalysisTests(unittest.TestCase):
    def test_complete_result_is_candidate_only_and_has_two_way_ci(self):
        with tempfile.TemporaryDirectory() as directory:
            result = analysis.analyze(
                _write_inputs(Path(directory)),
                _plan(),
                _config(),
                phase_name="confirmatory",
                bootstrap_repeats=100,
                bootstrap_seed=7,
            )

        self.assertEqual(result["status"], "candidate-evidence-ready")
        self.assertTrue(result["candidate_evidence_ready"])
        self.assertFalse(result["scientific_conclusion_allowed"])
        self.assertEqual(result["audit"]["valid_cell_count"], 16)
        self.assertEqual(len(result["architecture_budget_statistics"]), 2)
        final = next(row for row in result["architecture_budget_statistics"] if row["budget"] == 5)
        self.assertEqual(final["seed_count"], 2)
        self.assertEqual(final["target_subject_count"], 2)
        self.assertEqual(len(final["two_way_seed_subject_cluster_bootstrap_ci95"]), 2)
        self.assertIsNotNone(final["holm_adjusted_p_value"])
        self.assertEqual(len(result["order_effects"]), 2)
        self.assertEqual(result["order_effects"][0]["paired_seed_subject_count"], 4)

    def test_learning_inadequate_stage_is_excluded_from_every_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            result = analysis.analyze(
                _write_inputs(Path(directory), weak_stage=("random-a", 11, 0)),
                _plan(),
                _config(),
                phase_name="confirmatory",
                bootstrap_repeats=20,
            )

        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["audit"]["learning_adequacy_passed"])
        self.assertEqual(result["audit"]["invalid_cell_count"], 2)
        self.assertEqual({row["valid_cell_count"] for row in result["architecture_budget_statistics"]}, {7})
        invalid = [row for row in result["cells"] if not row["valid"]]
        self.assertTrue(all("fresh_learning_insufficient" in row["invalid_reasons"] for row in invalid))

    def test_calibration_uses_seed_trajectory_median_and_keeps_warning_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            result = analysis.analyze(
                _write_inputs(Path(directory), weak_stage=("random-a", 11, 0))[:1],
                _plan(),
                _calibration_config(),
                phase_name="calibration",
                bootstrap_repeats=20,
            )

        self.assertEqual(result["status"], "candidate-evidence-ready")
        self.assertTrue(result["audit"]["learning_adequacy_passed"])
        self.assertTrue(result["audit"]["sample_size_passed"])
        self.assertEqual(result["audit"]["valid_cell_count"], result["audit"]["expected_cell_count"])
        self.assertEqual(result["acceptance"]["fresh_learning_aggregation"], "per-seed-median")
        self.assertEqual(result["acceptance"]["minimum_independent_seeds"], 2)
        self.assertTrue(any(issue["severity"] == "warning" for issue in result["audit"]["issues"]))
        self.assertEqual(result["trajectory_learning_audit"][0]["stage_insufficient_count"], 1)

    def test_missing_run_and_inconsistent_gap_block_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _write_inputs(root, corrupt_gap=("random-b", 11, 1))
            payload = json.loads(paths[0][1].read_text(encoding="utf-8"))
            payload["runs"] = payload["runs"][:1]
            payload["completed_runs"] = 1
            paths[0][1].write_text(json.dumps(payload), encoding="utf-8")
            result = analysis.analyze(paths, _plan(), _config(), phase_name="confirmatory", bootstrap_repeats=20)

        self.assertFalse(result["audit"]["completeness_passed"])
        self.assertFalse(result["audit"]["all_expected_cells_valid"])
        codes = {row["code"] for row in result["audit"]["issues"]}
        self.assertIn("missing_run", codes)
        self.assertTrue(any("fresh_gap_inconsistent" in row["invalid_reasons"] for row in result["cells"]))

    def test_manifest_discovers_one_summary_per_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _write_inputs(root)
            manifest_path = root / "manifest.json"
            manifest = {
                "commands": [
                    {"order": order, "output": str(path.parent), "status": "succeeded"}
                    for order, path in paths
                ]
            }
            discovered = analysis.summaries_from_manifest(manifest, manifest_path)
        self.assertEqual([(order, path.name) for order, path in discovered], [("random-a", "summary.json"), ("random-b", "summary.json")])

    def test_holm_adjustment_is_step_down_and_order_preserving(self):
        adjusted = analysis.holm_adjust([0.01, 0.04, 0.03, None], alpha=0.05)
        self.assertAlmostEqual(adjusted[0]["adjusted_p_value"], 0.03)
        self.assertAlmostEqual(adjusted[1]["adjusted_p_value"], 0.06)
        self.assertAlmostEqual(adjusted[2]["adjusted_p_value"], 0.06)
        self.assertIsNone(adjusted[3])

    def test_source_majority_uses_only_source_subjects(self):
        self.assertEqual(analysis.source_majority_accuracy(_plan()), 0.5)


if __name__ == "__main__":
    unittest.main()
