import json
import tempfile
import unittest
from pathlib import Path

from edgeforge.lop_audit import audit_catalog


class LopAuditTests(unittest.TestCase):
    def test_audit_reports_missing_predictor_and_runs_analysis_when_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "ewc" / "metrics.json"
            missing.parent.mkdir(parents=True)
            missing.write_text(json.dumps({"tasks": [{"task": 1, "subject": 1, "current_before": {"acc": 0.4}, "current_after": {"acc": 0.5}}]}), encoding="utf-8")
            entries = []
            for seed in (1, 2, 3):
                result_path = root / f"probe-{seed}" / "metrics.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                metrics = []
                for step in range(3):
                    metrics.extend([
                        {"name": "task.spectra.transformer_1.effective_rank", "value": step + seed, "step": step, "context": {}},
                        {"name": "plasticity.acc_gain", "value": step * 0.1 + seed * 0.01, "step": step, "context": {}},
                    ])
                result_path.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
                entries.append({
                    "experiment_id": f"probe-{seed}", "workload": "raeeg-lop", "protocol": "lop-v1",
                    "method": "probe", "seed": seed,
                    "runner": {"result_path": f"probe-{seed}/metrics.json"},
                    "metadata": {"comparison_group": "lop", "replay": False},
                })
            entries.append({
                "experiment_id": "ewc-1", "workload": "raeeg", "protocol": "eeg-cl-v1-aligned",
                "method": "ewc", "seed": 1, "runner": {"result_path": "ewc/metrics.json"},
                "metadata": {"comparison_group": "aligned-full49", "replay": False},
            })
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"schema_version": 1, "worker_work_root": str(root), "experiments": entries}), encoding="utf-8")
            result = audit_catalog(catalog, bootstrap_repeats=100, minimum_pairs=2)
            self.assertEqual(result["status_counts"]["missing-predictor"], 1)
            probe = next(item for item in result["analyses"] if item["method"] == "probe")
            ewc = next(item for item in result["analyses"] if item["method"] == "ewc")
            self.assertEqual(probe["result"]["status"], "ok")
            self.assertEqual(ewc["result"]["status"], "blocked-incomplete-evidence")
            self.assertEqual(result["method_summary"]["ewc"]["statuses"], ["missing-predictor"])
            self.assertEqual(result["method_summary"]["probe"]["predictor_available"], 3)
            self.assertEqual(result["analysis_scope"], "lop-er-plasticity")
            self.assertFalse(result["scientific_conclusion_allowed"])

            exploratory = audit_catalog(
                catalog,
                predictor="task.importance.mean",
                outcome="plasticity.acc_gain",
                bootstrap_repeats=100,
                minimum_pairs=2,
                methods=["probe"],
            )
            self.assertEqual(exploratory["analysis_scope"], "exploratory-custom-association")
            self.assertEqual(exploratory["method_filter"], ["probe"])

    def test_audit_tolerates_non_envelope_metrics_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.json"
            summary.write_text(json.dumps({"metrics": ["acc", "mf1"], "summary": {"acc": 0.7}}), encoding="utf-8")
            invalid = root / "invalid.json"
            invalid.write_text("{not-json", encoding="utf-8")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "schema_version": 1,
                "worker_work_root": str(root),
                "experiments": [
                    {"experiment_id": "summary", "workload": "raeeg", "protocol": "historical", "method": "summary", "seed": 1, "runner": {"result_path": "summary.json"}, "metadata": {"comparison_group": "historical"}},
                    {"experiment_id": "invalid", "workload": "raeeg", "protocol": "historical", "method": "invalid", "seed": 2, "runner": {"result_path": "invalid.json"}, "metadata": {"comparison_group": "historical"}},
                ],
            }), encoding="utf-8")
            result = audit_catalog(catalog, bootstrap_repeats=100)
            self.assertEqual(result["status_counts"]["invalid-source"], 1)
            self.assertEqual(result["status_counts"]["missing-predictor"], 1)


if __name__ == "__main__":
    unittest.main()
