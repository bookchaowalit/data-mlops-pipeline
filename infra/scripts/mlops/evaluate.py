#!/usr/bin/env python3
"""Run the provider-neutral central MLOps release gate.

The gate evaluates a prediction artifact against the release manifest without
calling a model provider or writing to the operational database. It also
verifies the immutable digest of a local or fixture-backed model artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
from _env import PROJECT_ROOT  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "infra" / "mlops" / "model-release.json"


class MLOpsContractError(ValueError):
    """Raised when a release manifest or prediction artifact is invalid."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MLOpsContractError(f"cannot read JSON artifact: {path}") from exc


def _resolve_repo_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _resolve_artifact_path(value: str) -> Path:
    """Resolve file and fixture URIs used by the release contract."""
    if value.startswith("fixture://"):
        value = value.removeprefix("fixture://")
    elif value.startswith("file://"):
        value = unquote(urlparse(value).path)
    return _resolve_repo_path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predictions(path: Path) -> list[dict[str, str]]:
    value = _load_json(path)
    if not isinstance(value, list) or not value:
        raise MLOpsContractError("prediction artifact must be a non-empty JSON list")

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise MLOpsContractError(f"prediction row {index} is not an object")
        missing = [
            field for field in ("id", "label", "prediction") if not row.get(field)
        ]
        if missing:
            raise MLOpsContractError(
                f"prediction row {index} is missing: {', '.join(missing)}"
            )
        row_id = str(row["id"])
        if row_id in seen_ids:
            raise MLOpsContractError(f"duplicate prediction id: {row_id}")
        seen_ids.add(row_id)
        rows.append(
            {
                "id": row_id,
                "label": str(row["label"]),
                "prediction": str(row["prediction"]),
            }
        )
    return rows


def _f1_scores(rows: list[dict[str, str]]) -> dict[str, float]:
    labels = sorted(
        {row["label"] for row in rows} | {row["prediction"] for row in rows}
    )
    scores: dict[str, float] = {}
    for label in labels:
        true_positive = sum(
            row["label"] == label and row["prediction"] == label for row in rows
        )
        predicted = sum(row["prediction"] == label for row in rows)
        actual = sum(row["label"] == label for row in rows)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        scores[label] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return scores


def evaluate_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one release manifest and return a JSON-safe gate report."""
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise MLOpsContractError("release manifest must be a JSON object")
    if manifest.get("contract_version") != "solo-empire.mlops.release.v1":
        raise MLOpsContractError("unsupported MLOps contract version")

    evaluation = manifest.get("evaluation")
    artifact = manifest.get("artifact")
    if not isinstance(evaluation, dict) or not isinstance(artifact, dict):
        raise MLOpsContractError("manifest requires evaluation and artifact objects")

    dataset_value = str(evaluation.get("dataset") or "")
    prediction_value = str(evaluation.get("predictions") or dataset_value)
    dataset = (
        predictions_path.resolve()
        if predictions_path is not None
        else _resolve_repo_path(prediction_value)
    )
    if not dataset.is_file():
        raise MLOpsContractError(f"evaluation predictions do not exist: {dataset}")

    rows = _predictions(dataset)
    accuracy = sum(row["label"] == row["prediction"] for row in rows) / len(rows)
    f1_scores = _f1_scores(rows)
    macro_f1 = sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0.0
    minimum_examples = int(evaluation.get("min_examples", 1))
    minimum_accuracy = float(evaluation.get("min_accuracy", 0.0))
    minimum_macro_f1 = float(evaluation.get("min_macro_f1", 0.0))
    dataset_digest = _sha256(dataset)
    artifact_uri = str(artifact.get("uri") or "")
    artifact_digest = str(artifact.get("sha256") or "")
    artifact_type = str(artifact.get("type") or "")
    artifact_path = _resolve_artifact_path(artifact_uri) if artifact_uri else None
    artifact_actual_digest = (
        _sha256(artifact_path) if artifact_path and artifact_path.is_file() else None
    )
    artifact_integrity = (
        artifact_digest.startswith("sha256:")
        and artifact_path is not None
        and artifact_actual_digest == artifact_digest.removeprefix("sha256:")
    )

    artifact_is_deployable = (
        artifact.get("type") not in {None, "evaluation-fixture", "contract-only"}
        and bool(artifact_uri)
        and str(artifact.get("sha256") or "").startswith("sha256:")
        and artifact_integrity
    )
    provenance = manifest.get("provenance")
    dataset_lineage_integrity = True
    if isinstance(provenance, dict) and isinstance(provenance.get("dataset"), dict):
        dataset_lineage = provenance["dataset"]
        lineage_manifest_value = str(dataset_lineage.get("manifest_path") or "")
        if lineage_manifest_value:
            lineage_manifest_path = _resolve_repo_path(lineage_manifest_value)
            dataset_lineage_integrity = False
            if lineage_manifest_path.is_file():
                lineage_manifest = _load_json(lineage_manifest_path)
                declared_manifest_digest = str(
                    dataset_lineage.get("manifest_sha256") or ""
                )
                manifest_digest_matches = (
                    not declared_manifest_digest
                    or declared_manifest_digest
                    == f"sha256:{_sha256(lineage_manifest_path)}"
                )
                dataset_lineage_integrity = (
                    isinstance(lineage_manifest, dict)
                    and lineage_manifest.get("contract_version")
                    == "solo-empire.mlops.dataset.v1"
                    and lineage_manifest.get("dataset_id")
                    == dataset_lineage.get("dataset_id")
                    and lineage_manifest.get("dataset_version")
                    == dataset_lineage.get("dataset_version")
                    and lineage_manifest.get("dataset_sha256")
                    == dataset_lineage.get("dataset_sha256")
                    and manifest_digest_matches
                )

    checks = {
        "minimum_examples": len(rows) >= minimum_examples,
        "minimum_accuracy": accuracy >= minimum_accuracy,
        "minimum_macro_f1": macro_f1 >= minimum_macro_f1,
        "artifact_integrity": artifact_integrity,
        "dataset_lineage": dataset_lineage_integrity,
    }
    gate_passed = all(checks.values())

    promotion = (
        manifest.get("promotion") if isinstance(manifest.get("promotion"), dict) else {}
    )
    human_approval_id = promotion.get("human_approval_id")
    production_evidence = promotion.get("production_evidence")
    human_approval_present = bool(
        isinstance(human_approval_id, str) and human_approval_id.strip()
    )
    production_evidence_present = bool(
        isinstance(production_evidence, dict) and production_evidence
    ) or bool(isinstance(production_evidence, str) and production_evidence.strip())

    # Artifact eligibility is separate from environment promotion. A real
    # digest-verified candidate may pass offline evaluation and still be
    # blocked from any staging/production claim until human approval and
    # production evidence are recorded explicitly on the release manifest.
    promotion_blockers: list[str] = []
    if not artifact_is_deployable:
        if not artifact_uri or not artifact_path or not artifact_path.is_file():
            promotion_blockers.append("model-artifact-missing")
        elif not artifact_integrity:
            promotion_blockers.append("model-artifact-digest-mismatch")
        else:
            promotion_blockers.append(
                "real-immutable-model-artifact-digest-not-registered"
            )
    if not gate_passed:
        promotion_blockers.append("offline-evaluation-failed")
    if not human_approval_present:
        promotion_blockers.append("human-approval-required")
    if not production_evidence_present:
        promotion_blockers.append("production-environment-evidence-missing")
    if manifest.get("stage") == "production":
        promotion_blockers.append(
            "stage-must-not-claim-production-without-promotion-gate"
        )

    artifact_eligible = gate_passed and artifact_is_deployable
    promotion_eligible = (
        artifact_eligible and human_approval_present and production_evidence_present
    )

    return {
        "contract_version": manifest["contract_version"],
        "model_name": manifest.get("model_name"),
        "model_version": manifest.get("model_version"),
        "stage": manifest.get("stage"),
        "dataset": str(dataset.relative_to(PROJECT_ROOT))
        if dataset.is_relative_to(PROJECT_ROOT)
        else str(dataset),
        "dataset_sha256": dataset_digest,
        "predictions": str(dataset.relative_to(PROJECT_ROOT))
        if dataset.is_relative_to(PROJECT_ROOT)
        else str(dataset),
        "artifact": {
            "type": artifact_type,
            "path": (
                str(artifact_path.relative_to(PROJECT_ROOT))
                if artifact_path and artifact_path.is_relative_to(PROJECT_ROOT)
                else str(artifact_path)
                if artifact_path
                else None
            ),
            "declared_sha256": artifact_digest,
            "actual_sha256": f"sha256:{artifact_actual_digest}"
            if artifact_actual_digest
            else None,
        },
        "dataset_lineage": {
            "status": "pass" if dataset_lineage_integrity else "fail",
            "manifest_path": (
                str(provenance.get("dataset", {}).get("manifest_path"))
                if isinstance(provenance, dict)
                and isinstance(provenance.get("dataset"), dict)
                else None
            ),
        },
        "examples": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "metrics": {
            "accuracy": round(accuracy, 6),
            "macro_f1": round(macro_f1, 6),
            "per_label_f1": {
                label: round(score, 6) for label, score in sorted(f1_scores.items())
            },
        },
        "thresholds": {
            "min_examples": minimum_examples,
            "min_accuracy": minimum_accuracy,
            "min_macro_f1": minimum_macro_f1,
        },
        "checks": checks,
        "release_gate": "pass" if gate_passed else "fail",
        "artifact_eligible": artifact_eligible,
        "promotion_eligible": promotion_eligible,
        "promotion_blockers": promotion_blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the central MLOps release gate")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report", type=Path, help="Write the JSON report to a file")
    args = parser.parse_args()

    try:
        result = evaluate_manifest(args.manifest.resolve(), args.predictions)
    except MLOpsContractError as exc:
        result = {"release_gate": "error", "error": str(exc)}

    rendered = json.dumps(result, ensure_ascii=False, indent=None if args.json else 2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result.get("release_gate") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
