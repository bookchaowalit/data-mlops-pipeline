#!/usr/bin/env python3
"""Offline tests for the central MLOps release gate."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import urllib.request
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from mlops.evaluate import evaluate_manifest  # noqa: E402
from mlops.dataset import DatasetContractError, load_dataset_contract, prepare_dataset_bundle  # noqa: E402
from mlops.gold import materialize_gold_dataset  # noqa: E402
from mlops.monitor import monitor_predictions  # noqa: E402
from mlops.pipeline import run_pipeline  # noqa: E402
from mlops.registry import register_model  # noqa: E402
from mlops.serve import create_server  # noqa: E402
from mlops.staging import staging_dry_run  # noqa: E402


MODEL_ARTIFACT = ROOT / "mlops/artifacts/solo-empire-central-assistant/0.2.0/model.json"
GOLD_MANIFEST = ROOT / "mlops/gold/synthetic-gold-v1/dataset-manifest.json"
GOLD_CONTRACT = ROOT / "mlops/gold/synthetic-gold-v1/dataset-contract.json"
RELEASE_MANIFEST = ROOT / "mlops/model-release.json"


class MLOpsPipelineTests(unittest.TestCase):
    def test_baseline_passes_with_immutable_artifact_but_promotion_blocked(self):
        result = evaluate_manifest()
        self.assertEqual(result["release_gate"], "pass")
        self.assertTrue(result["checks"]["artifact_integrity"])
        self.assertTrue(result["checks"]["dataset_lineage"])
        self.assertTrue(result["artifact_eligible"])
        self.assertFalse(result["promotion_eligible"])
        self.assertIn("human-approval-required", result["promotion_blockers"])
        self.assertIn("production-environment-evidence-missing", result["promotion_blockers"])
        self.assertEqual(result["metrics"]["accuracy"], 1.0)
        self.assertEqual(result["metrics"]["macro_f1"], 1.0)
        self.assertEqual(result["stage"], "candidate")
        declared = result["artifact"]["declared_sha256"]
        actual = result["artifact"]["actual_sha256"]
        self.assertTrue(declared.startswith("sha256:"))
        self.assertEqual(declared, actual)
        self.assertTrue(MODEL_ARTIFACT.is_file())
        self.assertEqual(
            declared,
            f"sha256:{hashlib.sha256(MODEL_ARTIFACT.read_bytes()).hexdigest()}",
        )

    def test_checked_in_artifact_digest_matches_file(self):
        manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
        artifact = manifest["artifact"]
        path = ROOT.parent / artifact["uri"] if not Path(artifact["uri"]).is_absolute() else Path(artifact["uri"])
        if not path.is_file():
            path = (ROOT / Path(artifact["uri"]).relative_to("infra")).resolve() if str(artifact["uri"]).startswith("infra/") else ROOT / artifact["uri"]
        # Resolve via repository-relative path from PROJECT_ROOT-equivalent (infra parent)
        repo_root = ROOT.parent
        path = (repo_root / artifact["uri"]).resolve()
        self.assertTrue(path.is_file(), f"missing artifact: {path}")
        actual = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        self.assertEqual(artifact["sha256"], actual)
        self.assertEqual(artifact["type"], "naive-bayes-json")
        self.assertNotEqual(artifact["type"], "evaluation-fixture")

    def test_gold_dataset_lineage_and_privacy_boundary(self):
        manifest = json.loads(GOLD_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset_version"], "synthetic-gold-v1")
        self.assertEqual(manifest["source_layer"], "gold")
        self.assertEqual(manifest["source_kind"], "fixture")
        self.assertTrue(manifest["dataset_sha256"].startswith("sha256:"))
        self.assertTrue(manifest["training"]["source_files"][0]["sha256"].startswith("sha256:"))
        self.assertTrue(manifest["evaluation"]["source_files"][0]["sha256"].startswith("sha256:"))
        privacy = manifest["privacy"]
        self.assertFalse(privacy["contains_pii"])
        self.assertFalse(privacy["contains_credentials"])
        self.assertFalse(privacy["contains_provider_payloads"])
        self.assertEqual(privacy["boundary"], "synthetic-local-fixture-only")
        self.assertTrue(manifest["lineage"]["not_production_lake"])
        training = (ROOT.parent / manifest["training"]["path"]).read_bytes()
        evaluation = (ROOT.parent / manifest["evaluation"]["path"]).read_bytes()
        recomputed = f"sha256:{hashlib.sha256(training + evaluation).hexdigest()}"
        self.assertEqual(manifest["dataset_sha256"], recomputed)
        declared_manifest = (
            json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
            ["provenance"]["dataset"]["manifest_sha256"]
        )
        self.assertEqual(
            declared_manifest,
            f"sha256:{hashlib.sha256(GOLD_MANIFEST.read_bytes()).hexdigest()}",
        )

    def test_threshold_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.json"
            predictions.write_text(
                json.dumps([
                    {"id": "1", "label": "a", "prediction": "b"},
                    {"id": "2", "label": "a", "prediction": "a"},
                ]),
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "contract_version": "solo-empire.mlops.release.v1",
                "model_name": "test",
                "model_version": "test",
                "stage": "candidate",
                "artifact": {"type": "evaluation-fixture", "sha256": ""},
                "evaluation": {
                    "dataset": str(predictions),
                    "min_examples": 2,
                    "min_accuracy": 0.8,
                    "min_macro_f1": 0.8,
                },
            }), encoding="utf-8")
            result = evaluate_manifest(manifest)
            self.assertEqual(result["release_gate"], "fail")
            self.assertFalse(result["checks"]["minimum_accuracy"])
            self.assertFalse(result["promotion_eligible"])
            self.assertIn("offline-evaluation-failed", result["promotion_blockers"])

    def test_fixture_digest_mismatch_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.json"
            predictions.write_text(json.dumps([
                {"id": "1", "label": "a", "prediction": "a"},
            ]), encoding="utf-8")
            model = root / "model.json"
            model.write_text(json.dumps({"format": "test"}), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "contract_version": "solo-empire.mlops.release.v1",
                "model_name": "test",
                "model_version": "test",
                "stage": "candidate",
                "artifact": {
                    "type": "naive-bayes-json",
                    "uri": str(model),
                    "sha256": "sha256:not-the-model",
                },
                "evaluation": {
                    "dataset": str(predictions),
                    "min_examples": 1,
                    "min_accuracy": 0.0,
                    "min_macro_f1": 0.0,
                },
            }), encoding="utf-8")
            result = evaluate_manifest(manifest)
            self.assertEqual(result["release_gate"], "fail")
            self.assertFalse(result["checks"]["artifact_integrity"])
            self.assertIn("model-artifact-digest-mismatch", result["promotion_blockers"])

    def test_missing_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.json"
            predictions.write_text(json.dumps([
                {"id": "1", "label": "a", "prediction": "a"},
            ]), encoding="utf-8")
            missing = root / "does-not-exist-model.json"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "contract_version": "solo-empire.mlops.release.v1",
                "model_name": "test",
                "model_version": "test",
                "stage": "candidate",
                "artifact": {
                    "type": "naive-bayes-json",
                    "uri": str(missing),
                    "sha256": "sha256:" + ("a" * 64),
                },
                "evaluation": {
                    "dataset": str(predictions),
                    "min_examples": 1,
                    "min_accuracy": 0.0,
                    "min_macro_f1": 0.0,
                },
            }), encoding="utf-8")
            result = evaluate_manifest(manifest)
            self.assertEqual(result["release_gate"], "fail")
            self.assertFalse(result["checks"]["artifact_integrity"])
            self.assertFalse(result["artifact_eligible"])
            self.assertFalse(result["promotion_eligible"])
            self.assertIn("model-artifact-missing", result["promotion_blockers"])

    def test_missing_dataset_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "dataset-contract.json"
            contract.write_text(json.dumps({
                "contract_version": "solo-empire.mlops.dataset.v1",
                "dataset_id": "missing-source",
                "dataset_version": "v0",
                "source_layer": "gold",
                "training_uri": str(root / "missing-training.json"),
                "evaluation_uri": str(root / "missing-evaluation.json"),
            }), encoding="utf-8")
            loaded = load_dataset_contract(contract)
            with self.assertRaises(DatasetContractError):
                prepare_dataset_bundle(
                    loaded["training_path"],
                    loaded["evaluation_path"],
                    root / "out",
                    contract_path=contract,
                )

    def test_promotion_stays_blocked_without_approval_or_production_evidence(self):
        evaluation = evaluate_manifest()
        self.assertEqual(evaluation["release_gate"], "pass")
        self.assertTrue(evaluation["artifact_eligible"])
        self.assertFalse(evaluation["promotion_eligible"])
        staging = staging_dry_run(
            RELEASE_MANIFEST,
            evaluation,
            {"status": "registered", "rollback_target_available": False, "registered_versions": ["0.2.0"]},
        )
        self.assertEqual(staging["status"], "blocked")
        self.assertFalse(staging["deployment_enabled"])
        self.assertIn("human-approval-required", staging["blockers"])
        self.assertIn("production-environment-evidence-missing", staging["blockers"])
        self.assertIn("staging-environment-not-provisioned", staging["blockers"])
        self.assertIn("deployment-disabled", staging["blockers"])
        self.assertFalse(staging["safety"]["production_claim_allowed"])

        # Even if someone forges approval fields on a temp manifest, staging
        # still refuses deployment because environment provisioning is off.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forged = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
            forged["promotion"]["human_approval_id"] = "forged-approval"
            forged["promotion"]["production_evidence"] = {"note": "not real production"}
            forged_path = root / "forged-release.json"
            forged_path.write_text(json.dumps(forged), encoding="utf-8")
            forged_eval = evaluate_manifest(forged_path)
            # Approval fields make evaluation.promotion_eligible true only when
            # gate and artifact are also good; staging still fails closed.
            self.assertTrue(forged_eval["promotion_eligible"])
            forged_staging = staging_dry_run(
                forged_path,
                forged_eval,
                {"status": "registered", "rollback_target_available": True, "registered_versions": ["0.1.0", "0.2.0"]},
            )
            self.assertEqual(forged_staging["status"], "blocked")
            self.assertFalse(forged_staging["deployment_enabled"])
            self.assertIn("staging-environment-not-provisioned", forged_staging["blockers"])
            self.assertIn("deployment-disabled", forged_staging["blockers"])

    def test_local_vertical_slice_trains_registers_and_monitors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_pipeline(root / "release", root / "registry")
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["telemetry"]["contract"], "solo-empire.ai-telemetry.v1")
            self.assertEqual(result["telemetry"]["status"], "not_applicable")
            self.assertEqual(result["evaluation"]["release_gate"], "pass")
            self.assertTrue(result["evaluation"]["artifact_eligible"])
            self.assertFalse(result["evaluation"]["promotion_eligible"])
            self.assertIn("human-approval-required", result["evaluation"]["promotion_blockers"])
            self.assertIn("production-environment-evidence-missing", result["evaluation"]["promotion_blockers"])
            self.assertEqual(result["monitoring"]["drift_status"], "pass")
            self.assertEqual(result["dataset"]["source_kind"], "fixture")
            self.assertEqual(result["registry"]["status"], "registered")
            self.assertEqual(result["staging_dry_run"]["deployment_enabled"], False)
            self.assertEqual(result["staging_dry_run"]["status"], "blocked")
            self.assertTrue(Path(result["evidence"]["staging_dry_run"]).is_file())
            self.assertTrue((root / "registry" / "index.json").is_file())
            repeated = register_model(Path(result["train"]["manifest_path"]), root / "registry")
            self.assertEqual(repeated["artifact_sha256"], result["registry"]["artifact_sha256"])
            Path(result["dataset"]["manifest_path"]).write_text("{}\n", encoding="utf-8")
            tampered = evaluate_manifest(Path(result["train"]["manifest_path"]))
            self.assertEqual(tampered["release_gate"], "fail")
            self.assertFalse(tampered["checks"]["dataset_lineage"])

    def test_dataset_bundle_has_version_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = prepare_dataset_bundle(
                ROOT / "tests/fixtures/mlops/training_examples.json",
                ROOT / "tests/fixtures/mlops/evaluation_examples.json",
                Path(directory),
                dataset_version="test-v1",
                source_layer="fixture",
            )
            self.assertEqual(bundle["source_kind"], "fixture")
            manifest = json.loads(Path(bundle["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_version"], "test-v1")
            self.assertTrue(manifest["dataset_sha256"].startswith("sha256:"))
            self.assertTrue(manifest["training"]["source_files"][0]["sha256"].startswith("sha256:"))

    def test_approved_gold_materialization_is_idempotent_and_pipeline_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = materialize_gold_dataset(
                ROOT / "tests/fixtures/mlops/training_examples.json",
                ROOT / "tests/fixtures/mlops/evaluation_examples.json",
                root / "gold",
                dataset_version="gold-test-v1",
                approval_id="approval-test-001",
            )
            self.assertEqual(gold["status"], "materialized")
            repeated = materialize_gold_dataset(
                ROOT / "tests/fixtures/mlops/training_examples.json",
                ROOT / "tests/fixtures/mlops/evaluation_examples.json",
                root / "gold",
                dataset_version="gold-test-v1",
                approval_id="approval-test-001",
            )
            self.assertEqual(repeated["status"], "idempotent")
            result = run_pipeline(root / "release", root / "registry", dataset_manifest_path=Path(gold["contract_path"]))
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["dataset"]["dataset_version"], "gold-test-v1")
            self.assertTrue(result["evaluation"]["checks"]["dataset_lineage"])
            self.assertFalse(result["evaluation"]["promotion_eligible"])

    def test_checked_in_dataset_contract_points_at_synthetic_gold(self):
        contract = load_dataset_contract(ROOT / "mlops/dataset-contract.json")
        self.assertEqual(contract["source_layer"], "gold")
        self.assertEqual(contract["dataset_version"], "synthetic-gold-v1")
        self.assertEqual(contract["source_kind"], "fixture")
        self.assertTrue(contract["training_path"].is_file())
        self.assertTrue(contract["evaluation_path"].is_file())
        self.assertEqual(contract["training_path"].name, "training.json")
        self.assertEqual(contract["privacy"]["boundary"], "synthetic-local-fixture-only")
        self.assertFalse(contract["privacy"]["contains_pii"])
        with tempfile.TemporaryDirectory() as directory:
            bundle = prepare_dataset_bundle(
                contract["training_path"],
                contract["evaluation_path"],
                Path(directory),
                dataset_id=str(contract["dataset_id"]),
                dataset_version=str(contract["dataset_version"]),
                source_layer=str(contract["source_layer"]),
                contract_path=ROOT / "mlops/dataset-contract.json",
            )
            self.assertEqual(bundle["dataset_version"], "synthetic-gold-v1")
            self.assertTrue(bundle["dataset_sha256"].startswith("sha256:"))

    def test_pipeline_consumes_checked_in_gold_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_pipeline(
                root / "release",
                root / "registry",
                dataset_manifest_path=ROOT / "mlops/dataset-contract.json",
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["dataset"]["dataset_version"], "synthetic-gold-v1")
            self.assertEqual(result["evaluation"]["release_gate"], "pass")
            self.assertFalse(result["evaluation"]["promotion_eligible"])
            self.assertEqual(result["staging_dry_run"]["status"], "blocked")

    def test_monitor_detects_prediction_distribution_shift(self):
        baseline = [
            {"id": "a", "prediction": "general"},
            {"id": "b", "prediction": "action"},
        ]
        current = [
            {"id": "c", "prediction": "general"},
            {"id": "d", "prediction": "general"},
        ]
        result = monitor_predictions(current, baseline)
        self.assertEqual(result["drift_status"], "fail")
        self.assertGreaterEqual(result["psi"], result["psi_threshold"])

    def test_local_server_predicts_without_exposing_request_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = run_pipeline(root / "release")
            server = create_server(Path(pipeline["train"]["model_path"]))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                address = f"http://127.0.0.1:{server.server_address[1]}/predict"
                request = urllib.request.Request(
                    address,
                    data=json.dumps({"text": "urgent please reply"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(response.status, 200)
                self.assertEqual(payload["prediction"], "action")
                self.assertNotIn("urgent please reply", json.dumps(payload))
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_address[1]}/metrics", timeout=3
                ) as metrics_response:
                    metrics = metrics_response.read().decode("utf-8")
                self.assertIn("solo_empire_mlops_predictions_total", metrics)
                self.assertNotIn("urgent please reply", metrics)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
