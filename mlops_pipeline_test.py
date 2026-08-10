#!/usr/bin/env python3
"""Offline tests for the central MLOps release gate."""

from __future__ import annotations

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
from mlops.dataset import load_dataset_contract, prepare_dataset_bundle  # noqa: E402
from mlops.gold import materialize_gold_dataset  # noqa: E402
from mlops.monitor import monitor_predictions  # noqa: E402
from mlops.pipeline import run_pipeline  # noqa: E402
from mlops.registry import register_model  # noqa: E402
from mlops.serve import create_server  # noqa: E402


class MLOpsPipelineTests(unittest.TestCase):
    def test_baseline_passes_but_is_not_deployable(self):
        result = evaluate_manifest()
        self.assertEqual(result["release_gate"], "pass")
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(result["metrics"]["accuracy"], 1.0)
        self.assertEqual(result["metrics"]["macro_f1"], 1.0)

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

    def test_fixture_digest_mismatch_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.json"
            predictions.write_text(json.dumps([
                {"id": "1", "label": "a", "prediction": "a"},
            ]), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "contract_version": "solo-empire.mlops.release.v1",
                "model_name": "test",
                "model_version": "test",
                "stage": "candidate",
                "artifact": {
                    "type": "evaluation-fixture",
                    "sha256": "sha256:not-the-dataset",
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

    def test_local_vertical_slice_trains_registers_and_monitors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_pipeline(root / "release", root / "registry")
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["telemetry"]["contract"], "solo-empire.ai-telemetry.v1")
            self.assertEqual(result["telemetry"]["status"], "not_applicable")
            self.assertEqual(result["evaluation"]["release_gate"], "pass")
            self.assertTrue(result["evaluation"]["promotion_eligible"])
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

    def test_real_lake_contract_is_explicit_and_versioned(self):
        contract = load_dataset_contract(ROOT / "mlops/dataset-contract.json")
        self.assertEqual(contract["source_layer"], "gold")
        self.assertEqual(contract["dataset_version"], "gold-v1")
        self.assertEqual(contract["training_path"].parts[-1], "dataset=training")

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
