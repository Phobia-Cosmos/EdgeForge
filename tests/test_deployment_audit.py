import copy
import hashlib
import json
import unittest

from edgeforge.deployment_audit import audit_deployment
from edgeforge.model_pipeline import normalize_model_manifest


def _probe(*, advertised=None, inferred=None, architecture="x86_64", accelerators=None):
    probe = {
        "schema_version": 1,
        "generated_at": "2026-08-21T00:00:00Z",
        "capabilities": {
            "architecture": architecture,
            "accelerators": accelerators or [],
        },
        "backend_claims": {
            "advertised": advertised or [],
            "inferred": inferred or [],
            "policy": "backend readiness requires explicit configuration and a successful runtime validation",
        },
    }
    digest_payload = {key: value for key, value in probe.items() if key != "generated_at"}
    probe["probe_digest"] = hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return probe


class DeploymentAuditTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "schema_version": 1,
            "model": {"name": "tiny", "format": "external"},
            "dataset": {"name": "synthetic", "manifest_digest": "dataset:v1"},
            "compiler": {"backend": "python-reference", "identity": "reference-v1"},
            "target": {"architecture": "x86_64"},
        }

    def test_missing_correctness_evidence_is_blocked(self):
        result = audit_deployment(self.manifest, _probe(advertised=["python-reference"]))
        self.assertEqual(result["status"], "blocked")
        reasons = {item["name"]: item["reason"] for item in result["checks"]}
        self.assertIn("no successful correctness evidence", reasons["model-correctness"])
        self.assertFalse(result["scientific_conclusion_allowed"])

    def test_matching_successful_model_run_makes_audit_ready(self):
        normalized = normalize_model_manifest(self.manifest)
        model_run = {
            "task_status": "succeeded",
            "correctness": True,
            "model_name": "tiny",
            "dataset_name": "synthetic",
            "compiler_backend": "python-reference",
            "target_architecture": "x86_64",
            "manifest": normalized,
        }
        result = audit_deployment(self.manifest, _probe(advertised=["python-reference"]), [model_run])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["evidence"]["successful_model_runs"], 1)
        self.assertTrue(all(item["status"] == "pass" for item in result["checks"]))

    def test_inferred_backend_and_wrong_architecture_stay_blocked(self):
        result = audit_deployment(
            self.manifest,
            _probe(inferred=["python-reference"], architecture="aarch64"),
        )
        self.assertEqual(result["status"], "blocked")
        checks = {item["name"]: item for item in result["checks"]}
        self.assertEqual(checks["backend-advertised"]["status"], "blocked")
        self.assertEqual(checks["target-architecture"]["status"], "blocked")

    def test_probe_digest_tampering_is_detected(self):
        probe = _probe(advertised=["python-reference"])
        tampered = copy.deepcopy(probe)
        tampered["capabilities"]["architecture"] = "aarch64"
        result = audit_deployment(self.manifest, tampered)
        checks = {item["name"]: item for item in result["checks"]}
        self.assertEqual(checks["probe-integrity"]["status"], "blocked")

    def test_model_runs_json_wrapper_is_supported(self):
        normalized = normalize_model_manifest(self.manifest)
        run = {
            "task_status": "succeeded",
            "correctness": True,
            "manifest": normalized,
        }
        result = audit_deployment(self.manifest, _probe(advertised=["python-reference"]), {"model_runs": [run]})
        self.assertEqual(result["status"], "ready")


if __name__ == "__main__":
    unittest.main()
