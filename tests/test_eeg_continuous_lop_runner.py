import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "run-eeg-architecture-continuous-lop.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_continuous_lop_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_script()


class ContinuousLoPRunnerTests(unittest.TestCase):
    def test_source_holdout_uses_final_files_within_each_subject(self):
        root = Path("/dataset")
        paths = {
            1: [root / "source" / "1" / "data" / f"{index}.npy" for index in (1, 2, 3, 4)],
            2: [root / "source" / "2" / "data" / f"{index}.npy" for index in (11, 12, 13, 14)],
        }

        def fake_read(selected):
            selected = list(selected)
            values = torch.tensor([[int(path.stem)] for path in selected], dtype=torch.float32)
            labels = torch.tensor([int(path.stem) for path in selected], dtype=torch.int64)
            files = [
                {"data": str(path), "label": str(path), "data_sha256": path.stem, "label_sha256": path.stem}
                for path in selected
            ]
            return values, labels, files

        with (
            mock.patch.object(module, "_subject_files", side_effect=lambda _root, _group, subject: paths[subject]),
            mock.patch.object(module, "_read_subject_files", side_effect=fake_read),
        ):
            train_x, _, eval_x, _, train_files, eval_files = module.read_source_split(root, [1, 2], 0.25)

        self.assertEqual(train_x.flatten().tolist(), [1.0, 2.0, 3.0, 11.0, 12.0, 13.0])
        self.assertEqual(eval_x.flatten().tolist(), [4.0, 14.0])
        self.assertEqual([Path(item["data"]).name for item in train_files], ["1.npy", "2.npy", "3.npy", "11.npy", "12.npy", "13.npy"])
        self.assertEqual([Path(item["data"]).name for item in eval_files], ["4.npy", "14.npy"])

    def test_zero_source_holdout_preserves_training_set_evaluation(self):
        paths = [Path("/dataset/source/1/data/1.npy"), Path("/dataset/source/1/data/2.npy")]
        values = torch.tensor([[1.0], [2.0]])
        labels = torch.tensor([1, 2])
        files = [{"data": str(path), "label": str(path), "data_sha256": path.stem, "label_sha256": path.stem} for path in paths]
        with (
            mock.patch.object(module, "_subject_files", return_value=paths),
            mock.patch.object(module, "_read_subject_files", return_value=(values, labels, files)),
        ):
            train_x, train_y, eval_x, eval_y, _, eval_files = module.read_source_split(Path("/dataset"), [1], 0.0)
        self.assertIs(train_x, eval_x)
        self.assertIs(train_y, eval_y)
        self.assertEqual(eval_files, [])

    def test_retention_sample_limit_is_reproducible(self):
        values = torch.arange(100).reshape(20, 5)
        labels = torch.arange(20)
        first_x, first_y, first_indexes = module.limit_samples(values, labels, 7)
        second_x, second_y, second_indexes = module.limit_samples(values, labels, 7)
        self.assertEqual(first_indexes, sorted(first_indexes))
        self.assertEqual(first_indexes, second_indexes)
        self.assertTrue(torch.equal(first_x, second_x))
        self.assertTrue(torch.equal(first_y, labels[first_indexes]))
        self.assertTrue(torch.equal(second_y, labels[second_indexes]))

    def test_subject_roles_must_be_unique_and_disjoint(self):
        module.validate_subject_roles([1, 2], [3, 4], [5])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            module.validate_subject_roles([1, 2], [2, 3], [5])
        with self.assertRaisesRegex(ValueError, "duplicate subject"):
            module.validate_subject_roles([1, 1], [2], [3])

    def test_freeze_batch_norm_keeps_affine_parameters_trainable(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
        module.set_adaptation_train_mode(model, True)
        self.assertTrue(model.training)
        self.assertFalse(model[0].training)
        self.assertTrue(model[1].training)
        self.assertTrue(model[0].weight.requires_grad)
        module.set_adaptation_train_mode(model, False)
        self.assertTrue(model[0].training)

    def test_subject_seed_is_stable_and_independent_of_stage_order(self):
        first = module.stable_subject_seed("fresh-model", "tcn", 4321, 17)
        second = module.stable_subject_seed("fresh-model", "tcn", 4321, 17)
        self.assertEqual(first, second)
        self.assertNotEqual(first, module.stable_subject_seed("fresh-model", "tcn", 4321, 16))
        self.assertNotEqual(first, module.stable_subject_seed("target-train", "tcn", 4321, 17))

    def test_atomic_run_and_source_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_path = root / "run.json"
            run = {"status": "complete", "run_signature": "run-signature", "source": {}, "stages": []}
            module.atomic_write_json(run_path, run)
            self.assertEqual(module._load_completed_run(run_path, "run-signature"), run)
            self.assertIsNone(module._load_completed_run(run_path, "different"))

            stage_path = root / "warm-stage-0001.pt"
            stage_checkpoint = {
                "status": "complete",
                "run_signature": "partial-signature",
                "completed_stages": 1,
                "model_state_dict": {"weight": torch.tensor([2.0])},
            }
            module.atomic_save_checkpoint(stage_path, stage_checkpoint)
            partial = {
                "status": "partial",
                "run_signature": "partial-signature",
                "source": {},
                "stages": [{"stage": 0, "subject": 11}],
                "stage_checkpoint": {"file": stage_path.name, "completed_stages": 1},
            }
            module.atomic_write_json(run_path, partial)
            self.assertEqual(module._load_partial_run(run_path, "partial-signature", [11, 12]), partial)
            self.assertIsNone(module._load_partial_run(run_path, "partial-signature", [12, 11]))
            loaded_stage = module._load_stage_checkpoint(stage_path, "partial-signature", 1)
            self.assertIsNotNone(loaded_stage)
            self.assertTrue(torch.equal(loaded_stage["model_state_dict"]["weight"], torch.tensor([2.0])))

            checkpoint_path = root / "cache" / "source.pt"
            checkpoint = {
                "status": "complete",
                "source_signature": "source-signature",
                "model_state_dict": {"weight": torch.tensor([1.0])},
                "source_metrics": {"accuracy": 0.5},
            }
            module.atomic_save_checkpoint(checkpoint_path, checkpoint)
            loaded = module._load_source_checkpoint(checkpoint_path, "source-signature")
            self.assertIsNotNone(loaded)
            self.assertTrue(torch.equal(loaded["model_state_dict"]["weight"], torch.tensor([1.0])))
            self.assertIsNone(module._load_source_checkpoint(checkpoint_path, "different"))


if __name__ == "__main__":
    unittest.main()
