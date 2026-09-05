import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plan-isruc-full-lop.py"
SPEC = importlib.util.spec_from_file_location("edgeforge_test_isruc_full_lop_plan", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


LABEL_COUNTS = (
    (16, 1, 1, 1, 1),
    (1, 16, 1, 1, 1),
    (1, 1, 16, 1, 1),
    (1, 1, 1, 16, 1),
    (1, 1, 1, 1, 16),
    (8, 5, 3, 2, 2),
    (2, 2, 3, 5, 8),
)


def _labels_from_counts(counts):
    return np.concatenate([np.full(count, label, dtype=np.int64) for label, count in enumerate(counts)])


def _write_pair(
    root: Path,
    subject: int,
    *,
    index: int = 0,
    data_shape=(20, 8, 3000),
    labels=None,
) -> None:
    data_root = root / str(subject) / "data"
    label_root = root / str(subject) / "label"
    data_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    data = np.lib.format.open_memmap(data_root / f"{index}.npy", mode="w+", dtype=np.float32, shape=data_shape)
    del data
    if labels is None:
        labels = np.arange(20, dtype=np.int64) % 5
    np.save(label_root / f"{index}.npy", np.asarray(labels))


def _write_dataset(root: Path, subject_count: int) -> None:
    for subject in range(1, subject_count + 1):
        _write_pair(root, subject, labels=_labels_from_counts(LABEL_COUNTS[subject - 1]))


class ISRUCFullLoPPlanTests(unittest.TestCase):
    def test_roles_cover_subjects_without_overlap_and_plan_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "isruc"
            _write_dataset(source, 7)

            first = planner.build_plan(source, split_seed=37, source_count=2, target_count=3)
            second = planner.build_plan(source, split_seed=37, source_count=2, target_count=3)

            self.assertEqual(first, second)
            roles = [set(first["roles"][name]) for name in ("source", "target", "retention")]
            self.assertFalse(roles[0] & roles[1])
            self.assertFalse(roles[0] & roles[2])
            self.assertFalse(roles[1] & roles[2])
            self.assertEqual(set().union(*roles), set(first["subjects_available"]))
            self.assertGreater(first["totals"]["payload_bytes"], 0)

    def test_random_orders_are_distinct_even_when_raw_shuffles_collide(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "isruc"
            _write_dataset(source, 4)
            plan = planner.build_plan(source, split_seed=0, source_count=1, target_count=2)

            target = plan["roles"]["target"]
            raw_a = list(target)
            raw_b = list(target)
            random.Random(1).shuffle(raw_a)
            random.Random(2).shuffle(raw_b)
            self.assertEqual(raw_a, raw_b)
            self.assertNotEqual(plan["orders"]["random-a"], plan["orders"]["random-b"])
            self.assertEqual(set(plan["orders"]["random-a"]), set(target))
            self.assertEqual(set(plan["orders"]["random-b"]), set(target))

    def test_label_shift_orders_follow_js_divergence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "isruc"
            _write_dataset(source, 7)
            plan = planner.build_plan(source, split_seed=19, source_count=2, target_count=3)

            profiles = {row["subject"]: row for row in plan["profiles"]}
            ascending = plan["orders"]["label-shift-ascending"]
            expected = sorted(
                plan["roles"]["target"],
                key=lambda subject: (profiles[subject]["label_js_from_source"], subject),
            )
            self.assertEqual(ascending, expected)
            self.assertEqual(plan["orders"]["label-shift-descending"], list(reversed(expected)))

    def test_portable_plan_omits_absolute_source_path(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "isruc"
            _write_dataset(source, 4)

            portable = planner.build_plan(source, split_seed=3, source_count=1, target_count=2, portable=True)
            local = planner.build_plan(source, split_seed=3, source_count=1, target_count=2, portable=False)
            serialized = json.dumps(portable, sort_keys=True)

            self.assertIsNone(portable["source_root"])
            self.assertNotIn(str(source.resolve()), serialized)
            self.assertEqual(portable["plan_digest"], local["plan_digest"])

    def test_materialize_creates_complete_symlink_role_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "isruc"
            view = root / "view"
            _write_dataset(source, 4)
            plan = planner.build_plan(source, split_seed=11, source_count=1, target_count=2, portable=True)

            planner.materialize_view(plan, source, view)

            for role, subjects in plan["roles"].items():
                for subject in subjects:
                    link = view / role / str(subject)
                    self.assertTrue(link.is_symlink())
                    self.assertEqual(link.resolve(), (source / str(subject)).resolve())
            self.assertEqual(json.loads((view / "PLAN.json").read_text(encoding="utf-8")), plan)

    def test_unpaired_data_or_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_only = root / "data-only"
            _write_pair(data_only, 1)
            (data_only / "1" / "label" / "0.npy").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing label pair"):
                planner.inspect_subject(data_only, 1)

            orphan_label = root / "orphan-label"
            _write_pair(orphan_label, 1)
            np.save(orphan_label / "1" / "label" / "1.npy", np.zeros(20, dtype=np.int64))
            with self.assertRaisesRegex(FileNotFoundError, "missing data pair"):
                planner.inspect_subject(orphan_label, 1)

    def test_invalid_data_or_label_shape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_data = root / "bad-data"
            _write_pair(bad_data, 1, data_shape=(19, 8, 3000))
            with self.assertRaisesRegex(ValueError, "invalid ISRUC pair"):
                planner.inspect_subject(bad_data, 1)

            bad_labels = root / "bad-labels"
            _write_pair(bad_labels, 1, labels=np.zeros(19, dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "invalid ISRUC pair"):
                planner.inspect_subject(bad_labels, 1)

    def test_out_of_range_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.zeros(20, dtype=np.int64)
            labels[-1] = 5
            _write_pair(root, 1, labels=labels)
            with self.assertRaisesRegex(ValueError, "labels outside 0..4"):
                planner.inspect_subject(root, 1)


if __name__ == "__main__":
    unittest.main()
