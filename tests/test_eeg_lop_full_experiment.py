import importlib.util
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "run-eeg-lop-full-experiment.py"
    spec = importlib.util.spec_from_file_location("edgeforge_test_full_lop_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_script()


class FullLoPExperimentTests(unittest.TestCase):
    def test_build_commands_expands_order_and_architecture(self):
        plan = {
            "roles": {"source": [1, 2], "target": [3, 4, 5], "retention": [6]},
            "orders": {"random-a": [5, 3, 4]},
        }
        config = {
            "phases": {
                "calibration": {
                    "architectures": ["tcn", "transformer"],
                    "seeds": [11, 12],
                    "orders": ["random-a"],
                    "target_limit": 2,
                    "source_epochs": 3,
                    "budgets": [0, 5],
                    "batch_size": 8,
                    "source_lr": 0.002,
                    "adapt_lr": 0.001,
                    "source_eval_fraction": 0.2,
                    "retention_max_samples": 20,
                    "input_scale": 100000.0,
                    "freeze_batch_norm": True,
                }
            }
        }
        records = module.build_commands(
            plan,
            config,
            phase_name="calibration",
            view=Path("/tmp/view"),
            output_root=Path("/tmp/output"),
            device="cpu",
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["target_subjects"], [5, 3])
        self.assertEqual(records[0]["seeds"], [11, 12])
        self.assertIn("--resume", records[0]["command"])
        self.assertIn("--source-eval-fraction", records[0]["command"])
        self.assertIn("--source-checkpoint-root", records[0]["command"])
        self.assertIn("--input-scale", records[0]["command"])
        self.assertIn("--freeze-batch-norm", records[0]["command"])
        self.assertEqual({item["architecture"] for item in records}, {"tcn", "transformer"})

    def test_role_view_uses_links_without_copying_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "canonical"
            for subject in (1, 2, 3):
                (data_root / str(subject) / "data").mkdir(parents=True)
                (data_root / str(subject) / "label").mkdir()
            plan = {"roles": {"source": [1], "target": [2], "retention": [3]}}
            view = root / "view"
            module._ensure_role_view(plan, data_root, view)
            self.assertTrue((view / "source" / "1").is_symlink())
            self.assertEqual((view / "target" / "2").resolve(), (data_root / "2").resolve())
            self.assertEqual(json.loads((view / "PLAN.json").read_text())["roles"], plan["roles"])
            module._ensure_role_view(plan, data_root, view)

    def test_role_view_initialization_is_concurrently_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "canonical"
            for subject in (1, 2, 3):
                (data_root / str(subject) / "data").mkdir(parents=True)
                (data_root / str(subject) / "label").mkdir()
            plan = {"roles": {"source": [1], "target": [2], "retention": [3]}}
            view = root / "view"
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(module._ensure_role_view, plan, data_root, view) for _ in range(2)]
                for future in futures:
                    future.result()
            self.assertEqual((view / "retention" / "3").resolve(), (data_root / "3").resolve())

    def test_role_view_rejects_link_to_another_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left"
            right = root / "right"
            for candidate in (left, right):
                (candidate / "1" / "data").mkdir(parents=True)
                (candidate / "1" / "label").mkdir()
            view = root / "view"
            (view / "source").mkdir(parents=True)
            (view / "source" / "1").symlink_to(right / "1", target_is_directory=True)
            plan = {"roles": {"source": [1], "target": [2], "retention": [3]}}
            with self.assertRaises(ValueError):
                module._ensure_role_view(plan, left, view)


if __name__ == "__main__":
    unittest.main()
