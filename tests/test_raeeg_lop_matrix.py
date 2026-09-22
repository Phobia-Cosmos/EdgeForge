import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run-raeeg-lop-matrix.py"
SPEC = importlib.util.spec_from_file_location("edgeforge_raeeg_lop_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)
REPORT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize-raeeg-lop-matrix.py"
REPORT_SPEC = importlib.util.spec_from_file_location("edgeforge_raeeg_matrix_report", REPORT_SCRIPT)
assert REPORT_SPEC is not None and REPORT_SPEC.loader is not None
report_module = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(report_module)


class RaeegLopMatrixTests(unittest.TestCase):
    def test_compact_manifest_expands_stage_condition_and_template(self):
        manifest = {
            "schema_version": 1,
            "matrix_id": "matrix-test",
            "version": "0.14.0",
            "brainuicl_root": "/brainuicl",
            "output_root": "/output",
            "defaults": {"seed": 4321, "device": "cpu", "max_files": 1},
            "datasets": [
                {
                    "name": "ISRUC",
                    "data_root": "/data/isruc",
                    "subjects": [1],
                    "stages": [0, 10],
                    "conditions": [
                        {"name": "clean"},
                        {"name": "noise", "data_root": "/data/noise"},
                    ],
                    "methods": [
                        {
                            "name": "finetune",
                            "checkpoint_root_template": "/runs/{method}/individual_{stage}",
                            "baseline_checkpoint_root": "/pretrain",
                        }
                    ],
                }
            ],
        }
        cells = matrix.normalize_cells(manifest)
        self.assertEqual(len(cells), 4)
        self.assertEqual({item["condition"]["name"] for item in cells}, {"clean", "noise"})
        self.assertEqual({item["checkpoint_stage"] for item in cells}, {0, 10})
        stage10 = next(item for item in cells if item["checkpoint_stage"] == 10 and item["condition"]["name"] == "clean")
        self.assertTrue(stage10["checkpoint_root"].endswith("/runs/finetune/individual_10"))
        self.assertEqual(stage10["baseline_checkpoint_root"], "/pretrain")
        self.assertEqual(stage10["split"], "subject-eval")
        noisy = next(item for item in cells if item["condition"]["name"] == "noise")
        self.assertEqual(noisy["data_root"], "/data/noise")

    def test_compact_manifest_expands_explicit_seed_grid(self):
        manifest = {
            "schema_version": 1,
            "matrix_id": "seed-grid",
            "version": "0.14.0",
            "brainuicl_root": "/brain",
            "output_root": "/output",
            "datasets": [{
                "name": "ISRUC",
                "data_root": "/data",
                "subjects": [1],
                "seeds": [4321, 4322, 4323],
                "stages": [0],
                "methods": [{"name": "finetune", "checkpoint_root": "/checkpoints"}],
            }],
        }
        cells = matrix.normalize_cells(manifest)
        self.assertEqual([cell["seed"] for cell in cells], [4321, 4322, 4323])

    def test_bundle_marks_condition_and_metric_role_without_mixing_seed_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell_dir = root / "cells" / "cell"
            cell_dir.mkdir(parents=True)
            instrumentation = {"metrics": [{"name": "task.spectra.transformer_1.effective_rank", "value": 4.0, "step": 0, "context": {"subject": "1"}}]}
            probe = {"metrics": [
                {"name": "plasticity.acc_gain", "value": 0.1, "step": 0, "context": {"subject": "1"}},
                {"name": "task.forgetting.checkpoint_acc_drop", "value": 0.2, "step": 0, "context": {"subject": "1"}},
            ]}
            (cell_dir / "instrumentation.json").write_text(json.dumps(instrumentation), encoding="utf-8")
            (cell_dir / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
            manifest = {"schema_version": 1, "matrix_id": "m", "version": "0.14.0", "brainuicl_root": "/brain", "output_root": str(root)}
            cell = {"dataset": "ISRUC", "method": "finetune", "condition": {"name": "clean"}, "subject": 1, "seed": 4321, "checkpoint_stage": 0, "data_root": "/data", "brainuicl_root": "/brain", "split": "subject-eval"}
            bundle = matrix._merge_bundle(manifest, cell, "cell", cell_dir, [("predictor", instrumentation), ("outcome", probe)])
            self.assertEqual(bundle["spec"]["seed"], 4321)
            self.assertEqual(bundle["spec"]["metadata"]["condition"], "clean")
            contexts = {row["context"]["metric_role"]: row["context"] for row in bundle["metrics"]}
            self.assertEqual(contexts["predictor"]["condition"], "clean")
            self.assertNotIn("seed", contexts["predictor"])
            self.assertEqual(contexts["retention"]["metric_role"], "retention")

    def test_output_root_must_not_overlap_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(matrix.MatrixError):
                matrix._assert_disjoint(root / "out", [root])
            with self.assertRaises(matrix.MatrixError):
                matrix._assert_disjoint(root, [root / "input"])

    def test_build_commands_contains_explicit_stage_and_probe_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = {
                "brainuicl_root": str(root / "brain"),
                "dataset": "ISRUC",
                "checkpoint_root": str(root / "checkpoint"),
                "data_root": str(root / "data"),
                "subject": 1,
                "seed": 4321,
                "checkpoint_stage": 25,
                "method": "ewc",
                "split": "subject-eval:clean",
                "device": "cpu",
                "max_files": 2,
                "batch_size": 1,
                "max_batches": 3,
                "importance_batches": 2,
                "probe_steps": "0,5,10",
                "lr": 1e-5,
                "baseline_checkpoint_root": str(root / "pretrain"),
                "fresh_checkpoint_root": str(root / "pretrain"),
            }
            commands = matrix.build_commands(Path("/repo"), "python3", cell, root / "cell")
            self.assertEqual([item[0] for item in commands], ["instrumentation", "probe"])
            instrumentation_argv = commands[0][1]
            probe_argv = commands[1][1]
            self.assertIn("--checkpoint-stage", instrumentation_argv)
            self.assertIn("25", instrumentation_argv)
            self.assertIn("--baseline-checkpoint-root", instrumentation_argv)
            self.assertIn("--probe-steps", probe_argv)
            self.assertIn("0,5,10", probe_argv)
            self.assertIn("--fresh-checkpoint-root", probe_argv)
            commands_with_diagnostic = matrix.build_commands(Path("/repo"), "python3", cell, root / "cell-diagnostic", with_unlabeled_diagnostics=True)
            self.assertEqual([item[0] for item in commands_with_diagnostic], ["instrumentation", "probe", "unlabeled"])
            diagnostic_argv = commands_with_diagnostic[2][1]
            self.assertIn("--noise-severity", diagnostic_argv)

    def test_build_commands_adds_explicit_retention_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = {
                "brainuicl_root": str(root / "brain"),
                "dataset": "ISRUC",
                "checkpoint_root": str(root / "checkpoint"),
                "data_root": str(root / "data"),
                "subject": 2,
                "seed": 4321,
                "checkpoint_stage": 10,
                "method": "finetune",
                "split": "subject-eval",
                "device": "cpu",
                "max_files": 2,
                "batch_size": 1,
                "probe_steps": "0,5",
                "retention": {
                    "data_root": str(root / "retention-data"),
                    "subjects": [1, 3],
                    "max_files": 2,
                    "batch_size": 2,
                },
            }
            commands = matrix.build_commands(Path("/repo"), "python3", cell, root / "cell")
            instrumentation_argv = commands[0][1]
            probe_argv = commands[1][1]
            self.assertNotIn("--retention-data-root", instrumentation_argv)
            self.assertIn("--retention-data-root", probe_argv)
            self.assertIn(str(root / "retention-data"), probe_argv)
            self.assertEqual(probe_argv.count("--retention-subject"), 2)
            self.assertIn("--retention-max-files", probe_argv)
            self.assertIn("--retention-batch-size", probe_argv)

    def test_normalize_cells_resolves_retention_subjects(self):
        manifest = {
            "schema_version": 1,
            "matrix_id": "retention-test",
            "version": "0.15.0",
            "brainuicl_root": "/brain",
            "output_root": "/output",
            "defaults": {"retention": {"data_root": "/old", "subjects": ["sub-003", 1]}},
            "datasets": [{
                "name": "ISRUC",
                "data_root": "/data",
                "subjects": [2],
                "stages": [0],
                "methods": [{"name": "finetune", "checkpoint_root": "/checkpoints"}],
            }],
        }
        cells = matrix.normalize_cells(manifest)
        self.assertEqual(cells[0]["retention"]["subjects"], [1, 3])
        self.assertEqual(cells[0]["retention"]["data_root"], "/old")

    def test_trajectory_catalog_groups_cells_into_one_seeded_stage_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = []
            for stage, rank in ((0, 5.0), (10, 4.5), (25, 4.0), (49, 3.5)):
                cell_dir = root / "cells" / f"stage-{stage}"
                cell_dir.mkdir(parents=True)
                bundle = {
                    "spec": {
                        "experiment_id": f"cell-{stage}",
                        "workload": "raeeg",
                        "dataset": {"name": "ISRUC", "condition": "clean"},
                        "model": {"name": "BrainUICL"},
                        "protocol": "eeg-lop-matrix-v1",
                        "method": "finetune",
                        "seed": 4321,
                        "runner": {"result_path": f"cells/stage-{stage}/bundle.json", "adapter": "edgeforge-bundle-v1"},
                        "metadata": {"comparison_group": "ISRUC:clean", "subject": "2", "checkpoint_stage": stage},
                    },
                    "metrics": [
                        {"name": "task.spectra.transformer_1.effective_rank", "value": rank, "step": stage, "context": {"dataset": "ISRUC", "subject": "2", "method": "finetune", "split": "subject-eval"}},
                        {"name": "plasticity.acc_gain", "value": rank / 100.0, "step": stage, "context": {"dataset": "ISRUC", "subject": "2", "method": "finetune", "split": "subject-eval"}},
                    ],
                }
                path = cell_dir / "bundle.json"
                path.write_text(json.dumps(bundle), encoding="utf-8")
                results.append({"status": "succeeded", "bundle": str(path.relative_to(root))})
            manifest = {"schema_version": 1, "matrix_id": "matrix", "version": "0.14.0"}
            catalog = matrix._trajectory_catalog(root, manifest, results)
            self.assertEqual(len(catalog["experiments"]), 1)
            trajectory = catalog["experiments"][0]
            self.assertEqual(trajectory["metadata"]["trajectory_stages"], [0, 10, 25, 49])
            trajectory_path = root / trajectory["runner"]["result_path"]
            payload = json.loads(trajectory_path.read_text())
            self.assertEqual(sorted({row["step"] for row in payload["metrics"]}), [0, 10, 25, 49])

    def test_descriptive_report_preserves_missing_values_and_marks_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "trajectory.json"
            bundle.write_text(json.dumps({"metrics": [{"name": "task.spectra.transformer_1.effective_rank", "value": 3.0, "step": 0}]}), encoding="utf-8")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"schema_version": 1, "worker_work_root": str(root), "experiments": [{"experiment_id": "e", "dataset": {"name": "ISRUC", "condition": "clean"}, "method": "finetune", "seed": 4321, "metadata": {"subject": "1"}, "runner": {"result_path": "trajectory.json"}}]}), encoding="utf-8")
            report = report_module.build_report(catalog)
            self.assertFalse(report["scientific_conclusion_allowed"])
            self.assertEqual(report["rows"][0]["values"]["effective_rank"], 3.0)
            self.assertIsNone(report["rows"][0]["values"]["fresh_gap"])
            self.assertEqual(report["metric_roles"]["retention_acc_drop"], "retention")

    def test_descriptive_report_pairs_shift_with_clean_without_calling_it_lop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = []
            for condition, rank in (("clean", 4.0), ("noise", 5.0)):
                name = f"{condition}.json"
                (root / name).write_text(json.dumps({"metrics": [{"name": "task.spectra.transformer_1.effective_rank", "value": rank, "step": 0}, {"name": "plasticity.acc_gain", "value": 0.1, "step": 0}]}), encoding="utf-8")
                specs.append({"experiment_id": condition, "dataset": {"name": "ISRUC", "condition": condition}, "method": "finetune", "seed": 4321, "metadata": {"subject": "1"}, "runner": {"result_path": name}})
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"schema_version": 1, "worker_work_root": str(root), "experiments": specs}), encoding="utf-8")
            report = report_module.build_report(catalog)
            self.assertEqual(len(report["condition_comparisons"]), 1)
            self.assertEqual(report["condition_comparisons"][0]["shift_minus_clean"]["effective_rank"], 1.0)
            self.assertIn("unlabeled_entropy", report["metric_definitions"])
            self.assertFalse(report["scientific_conclusion_allowed"])
            self.assertIn("Shift minus clean", report_module.markdown(report))

    def test_partial_resume_keeps_persisted_cells_when_rebuilding_catalogs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "cells" / "old-cell"
            old.mkdir(parents=True)
            (old / "status.json").write_text(json.dumps({"cell_id": "old-cell", "status": "succeeded", "bundle": "cells/old-cell/bundle.json"}), encoding="utf-8")
            current = [{"cell_id": "new-cell", "status": "succeeded", "bundle": "cells/new-cell/bundle.json"}]
            merged = matrix._all_cell_results(root, current)
            self.assertEqual([item["cell_id"] for item in merged], ["new-cell", "old-cell"])

    def test_preflight_marks_missing_checkpoint_without_running_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = {
                "brainuicl_root": str(root / "brain"),
                "data_root": str(root / "data"),
                "checkpoint_root": str(root / "checkpoint"),
                "baseline_checkpoint_root": str(root / "baseline"),
                "fresh_checkpoint_root": str(root / "fresh"),
                "seed": 4321,
            }
            errors = matrix._check_cell_inputs(cell, {})
            self.assertTrue(errors)
            self.assertTrue(any("checkpoint_root missing" in item for item in errors))
            self.assertTrue(any("data_root missing" in item for item in errors))

    def test_preflight_checks_retention_subject_data_and_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retention_subject = root / "retention" / "1"
            (retention_subject / "data").mkdir(parents=True)
            (retention_subject / "label").mkdir(parents=True)
            (retention_subject / "data" / "0.npy").write_bytes(b"data")
            cell = {
                "brainuicl_root": str(root / "brain"),
                "data_root": str(root / "data"),
                "checkpoint_root": str(root / "checkpoint"),
                "baseline_checkpoint_root": None,
                "fresh_checkpoint_root": None,
                "seed": 4321,
                "retention": {"data_root": str(root / "retention"), "subjects": [1], "max_files": 1},
            }
            errors = matrix._check_cell_inputs(cell, {})
            self.assertTrue(any("retention label missing" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
