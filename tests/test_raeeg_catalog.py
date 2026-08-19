import json
import tempfile
import unittest
from pathlib import Path

from edgeforge.raeeg_catalog import build_catalog, build_entry, discover_results


class RaeegCatalogTests(unittest.TestCase):
    def test_build_entry_preserves_digest_and_comparison_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "experiments" / "regularization_cl_eeg_runs" / "aligned_full49_seed4321" / "ewc" / "metrics.json"
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({"method": "ewc", "summary": {"acc": 0.7}}), encoding="utf-8")
            entry = build_entry(root, result, dataset_manifest_digest="sha256:dataset")
            self.assertEqual(entry["method"], "ewc")
            self.assertEqual(entry["metadata"]["comparison_group"], "aligned-full49")
            self.assertEqual(entry["dataset"]["manifest_digest"], "sha256:dataset")
            self.assertEqual(len(entry["metadata"]["source_digest"]), 64)

    def test_discover_and_build_catalog_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "experiments" / "isruc" / "brainuicl" / "metrics.json"
            second = root / "experiments" / "faced" / "spr" / "RESULTS.json"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")
            paths = discover_results(root)
            catalog = build_catalog(root, paths)
            self.assertEqual(catalog["source"]["result_count"], 2)
            by_dataset = {entry["dataset"]["name"]: entry for entry in catalog["experiments"]}
            self.assertIn("ISRUC-Group-I", by_dataset)
            self.assertEqual(by_dataset["FACED"]["method"], "spr_eeg")


if __name__ == "__main__":
    unittest.main()
