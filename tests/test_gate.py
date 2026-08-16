import tempfile
import unittest
from pathlib import Path

from edgeforge import __version__
from edgeforge.db import Store


class CapabilityGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temporary.name) / "edgeforge.db")
        self.store.register_worker(
            {
                "id": "gate-worker",
                "hostname": "gate-worker",
                "capabilities": {
                    "architecture": "x86_64",
                    "accelerators": [],
                    "cpu_count": 4,
                    "memory_total_mb": 16000,
                },
                "metrics": {"load_1m": 0, "memory_available_mb": 12000},
                "labels": {},
                "version": __version__,
            }
        )

    def tearDown(self):
        self.temporary.cleanup()

    def record_experiment(
        self,
        experiment_id,
        method,
        summary,
        *,
        workload="raeeg",
        protocol="eeg-cl-v1-aligned",
        comparison_group="aligned-full49",
    ):
        spec = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "workload": workload,
            "dataset": {"name": "ISRUC"},
            "model": {"name": method},
            "protocol": protocol,
            "method": method,
            "seed": 4321,
            "runner": {"mode": "import", "result_path": "metrics.json", "adapter": "raeeg-metrics-v1"},
            "metadata": {"comparison_group": comparison_group},
        }
        task = self.store.create_task(
            {"kind": "experiment_run", "payload": {"spec": spec}, "requirements": {"worker_ids": ["gate-worker"]}}
        )
        self.assertEqual(self.store.lease_task("gate-worker")["id"], task["id"])
        self.store.complete_task(
            task["id"],
            "gate-worker",
            {
                "status": "succeeded",
                "runtime_version": __version__,
                "result": {
                    "experiment_id": experiment_id,
                    "workload": workload,
                    "experiment_bundle": {
                        "experiment_id": experiment_id,
                        "summary": summary,
                        "metrics": [
                            {
                                "namespace": "raeeg.research",
                                "name": "aggregate.score",
                                "value": summary["final_old_acc"],
                                "step": None,
                                "unit": "ratio",
                                "context": {},
                            }
                        ],
                        "source_result": {"digest": "a" * 64},
                    },
                },
            },
        )

    def register_model(self, model_id, experiment_id, name):
        return self.store.register_model(
            {
                "id": model_id,
                "name": name,
                "workload": "raeeg",
                "protocol": "eeg-cl-v1-aligned",
                "comparison_group": "aligned-full49",
                "source_experiment_id": experiment_id,
                "descriptor": {"purpose": "test"},
            }
        )

    def policy(self):
        return self.store.create_gate_policy(
            {
                "id": "aligned-policy-v1",
                "version": "1",
                "workload": "raeeg",
                "protocol": "eeg-cl-v1-aligned",
                "comparison_group": "aligned-full49",
                "rules": [
                    {"metric": "summary.final_old_acc", "operator": ">=", "threshold": 0.70},
                    {"metric": "summary.final_seen_acc", "operator": ">=", "threshold": 0.64},
                    {"metric": "summary.bwt_acc", "operator": ">=", "threshold": -0.02},
                ],
            }
        )

    def test_pass_and_fail_are_snapshotted_and_transition_models(self):
        self.record_experiment(
            "brainuicl-exp", "brainuicl", {"final_old_acc": 0.73, "final_seen_acc": 0.67, "bwt_acc": 0.01}
        )
        self.record_experiment(
            "finetune-exp", "finetune", {"final_old_acc": 0.60, "final_seen_acc": 0.55, "bwt_acc": -0.09}
        )
        self.register_model("brainuicl", "brainuicl-exp", "BrainUICL")
        self.register_model("finetune", "finetune-exp", "Finetune")
        self.policy()

        passed = self.store.evaluate_gate("brainuicl", "aligned-policy-v1")
        failed = self.store.evaluate_gate("finetune", "aligned-policy-v1")

        self.assertEqual(passed["status"], "PASS")
        self.assertTrue(all(item["passed"] for item in passed["rule_results"]))
        self.assertEqual(passed["metric_snapshot"][0]["value"], 0.73)
        self.assertEqual(failed["status"], "FAIL")
        self.assertFalse(all(item["passed"] for item in failed["rule_results"]))
        self.assertEqual(self.store.get_model("brainuicl")["status"], "accepted")
        self.assertEqual(self.store.get_model("finetune")["status"], "rejected")

        with self.assertRaisesRegex(RuntimeError, "policies are immutable"):
            self.policy()

    def test_series_metric_and_scope_are_strict(self):
        self.record_experiment(
            "series-exp", "brainuicl", {"final_old_acc": 0.71, "final_seen_acc": 0.65, "bwt_acc": 0.0}
        )
        self.register_model("series-model", "series-exp", "BrainUICL")
        policy = self.store.create_gate_policy(
            {
                "id": "series-policy",
                "version": "1",
                "workload": "raeeg",
                "protocol": "eeg-cl-v1-aligned",
                "comparison_group": "aligned-full49",
                "rules": [
                    {
                        "metric": "aggregate.score",
                        "namespace": "raeeg.research",
                        "operator": ">=",
                        "threshold": 0.70,
                    }
                ],
            }
        )
        evaluation = self.store.evaluate_gate("series-model", policy["id"])
        self.assertEqual(evaluation["metric_snapshot"][0]["source"], "experiment_metrics")

        self.record_experiment(
            "transfer-exp",
            "spr",
            {"final_old_acc": 0.90},
            protocol="spr-v1",
            comparison_group="method-transfer",
        )
        with self.assertRaises(ValueError):
            self.store.register_model(
                {
                    "id": "bad-scope",
                    "name": "SPR",
                    "workload": "raeeg",
                    "protocol": "eeg-cl-v1-aligned",
                    "comparison_group": "aligned-full49",
                    "source_experiment_id": "transfer-exp",
                }
            )

    def test_promote_switch_and_rollback_require_gate_approval(self):
        for model_id, score in (("brainuicl", 0.73), ("si", 0.71)):
            self.record_experiment(
                f"{model_id}-exp",
                model_id,
                {"final_old_acc": score, "final_seen_acc": 0.65, "bwt_acc": -0.01},
            )
            self.register_model(model_id, f"{model_id}-exp", model_id)
        self.policy()
        self.store.evaluate_gate("brainuicl", "aligned-policy-v1")
        self.store.evaluate_gate("si", "aligned-policy-v1")

        self.store.promote_model("brainuicl", "initial production selection")
        self.store.promote_model("si", "exercise production switch")
        self.assertEqual(self.store.get_model("brainuicl")["status"], "rolled_back")
        self.assertEqual(self.store.get_model("si")["status"], "production")

        result = self.store.rollback_model("si", "brainuicl", "restore known baseline")
        self.assertEqual(result["production"]["id"], "brainuicl")
        self.assertEqual(result["rolled_back"]["id"], "si")

    def test_gate_does_not_promote_and_candidate_can_be_explicitly_rejected(self):
        self.record_experiment(
            "candidate-exp", "candidate", {"final_old_acc": 0.72, "final_seen_acc": 0.66, "bwt_acc": 0.0}
        )
        self.register_model("candidate", "candidate-exp", "candidate")
        self.policy()
        self.store.evaluate_gate("candidate", "aligned-policy-v1")
        self.assertEqual(self.store.get_model("candidate")["status"], "accepted")

        self.record_experiment(
            "reject-exp", "reject", {"final_old_acc": 0.72, "final_seen_acc": 0.66, "bwt_acc": 0.0}
        )
        self.register_model("reject", "reject-exp", "reject")
        rejected = self.store.reject_model("reject", "operator cancelled candidate")
        self.assertEqual(rejected["status"], "rejected")
        with self.assertRaises(RuntimeError):
            self.store.promote_model("reject", "must not promote")


if __name__ == "__main__":
    unittest.main()
