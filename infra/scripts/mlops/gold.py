#!/usr/bin/env python3
"""Materialize an explicitly approved labeled source into a local Gold dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
from _env import PROJECT_ROOT  # noqa: E402

try:
    from .dataset import DatasetContractError, _canonical_json, load_dataset_rows
except ImportError:  # pragma: no cover - exercised by direct CLI execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mlops.dataset import DatasetContractError, _canonical_json, load_dataset_rows


SAFE_VALUE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
DATASET_CONTRACT = "solo-empire.mlops.dataset.v1"


class GoldActivationError(ValueError):
    """Raised when an approved Gold materialization is unsafe or conflicting."""


def _write(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json(value))


def _validate_value(name: str, value: str) -> str:
    if not SAFE_VALUE.fullmatch(value):
        raise GoldActivationError(
            f"{name} must contain only safe approval/path characters"
        )
    return value


def materialize_gold_dataset(
    training_source: Path,
    evaluation_source: Path,
    output_root: Path,
    *,
    dataset_id: str = "central-assistant-intent",
    dataset_version: str,
    approval_id: str,
    source_system: str = "operator-approved",
    privacy_class: str = "private",
) -> dict[str, Any]:
    """Write a deterministic Gold bundle without auto-labeling or overwriting."""
    dataset_id = _validate_value("dataset_id", dataset_id)
    dataset_version = _validate_value("dataset_version", dataset_version)
    approval_id = _validate_value("approval_id", approval_id)
    source_system = _validate_value("source_system", source_system)
    if privacy_class not in {"internal", "private", "restricted"}:
        raise GoldActivationError(
            "Gold datasets must be internal, private, or restricted"
        )

    training_rows, training_evidence = load_dataset_rows(training_source.resolve())
    evaluation_rows, evaluation_evidence = load_dataset_rows(
        evaluation_source.resolve()
    )
    training_ids = {row["id"] for row in training_rows}
    evaluation_ids = {row["id"] for row in evaluation_rows}
    overlap = training_ids & evaluation_ids
    if overlap:
        raise GoldActivationError(f"training/evaluation IDs overlap: {sorted(overlap)}")
    training_labels = {row["label"] for row in training_rows}
    evaluation_labels = {row["label"] for row in evaluation_rows}
    if len(training_labels) < 2:
        raise GoldActivationError("Gold training data must contain at least two labels")
    if not evaluation_labels.issubset(training_labels):
        raise GoldActivationError(
            "evaluation labels must be present in training labels"
        )

    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    training_bytes = _canonical_json(training_rows)
    evaluation_bytes = _canonical_json(evaluation_rows)
    dataset_digest = hashlib.sha256(training_bytes + evaluation_bytes).hexdigest()
    contract = {
        "contract_version": DATASET_CONTRACT,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "source_layer": "gold",
        "source_kind": "approved-materialization",
        "training_uri": "training.json",
        "evaluation_uri": "evaluation.json",
        "required_columns": ["id", "text", "label"],
        "privacy": {
            "class": privacy_class,
            "runtime_metrics_must_exclude": ["text", "raw_payload", "credentials"],
        },
        "materialization": {
            "approval_id": approval_id,
            "source_system": source_system,
            "source_training": training_evidence,
            "source_evaluation": evaluation_evidence,
            "training_rows": len(training_rows),
            "evaluation_rows": len(evaluation_rows),
            "training_sha256": f"sha256:{hashlib.sha256(training_bytes).hexdigest()}",
            "evaluation_sha256": f"sha256:{hashlib.sha256(evaluation_bytes).hexdigest()}",
            "dataset_sha256": f"sha256:{dataset_digest}",
            "created_by": os.environ.get("GITHUB_SHA")
            or os.environ.get("GIT_COMMIT")
            or "unavailable-local",
        },
    }

    with tempfile.TemporaryDirectory(
        prefix="solo-empire-gold-", dir=output_root.parent
    ) as temp_dir:
        temporary = Path(temp_dir)
        _write(temporary / "training.json", training_rows)
        _write(temporary / "evaluation.json", evaluation_rows)
        _write(temporary / "dataset-contract.json", contract)
        if output_root.exists():
            expected = {
                name: (temporary / name).read_bytes()
                for name in (
                    "training.json",
                    "evaluation.json",
                    "dataset-contract.json",
                )
            }
            actual = {
                name: (output_root / name).read_bytes()
                if (output_root / name).is_file()
                else None
                for name in expected
            }
            if all(actual[name] == data for name, data in expected.items()):
                return {
                    "status": "idempotent",
                    "dataset_id": dataset_id,
                    "dataset_version": dataset_version,
                    "output_root": str(output_root),
                    "contract_path": str(output_root / "dataset-contract.json"),
                    "dataset_sha256": f"sha256:{dataset_digest}",
                }
            raise GoldActivationError(
                f"Gold dataset conflict; refusing to overwrite: {output_root}"
            )
        temporary.replace(output_root)

    return {
        "status": "materialized",
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "output_root": str(output_root),
        "contract_path": str(output_root / "dataset-contract.json"),
        "dataset_sha256": f"sha256:{dataset_digest}",
        "training_rows": len(training_rows),
        "evaluation_rows": len(evaluation_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize an approved MLOps Gold dataset"
    )
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--evaluation-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default="central-assistant-intent")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--source-system", default="operator-approved")
    parser.add_argument(
        "--privacy-class",
        choices=["internal", "private", "restricted"],
        default="private",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = materialize_gold_dataset(
            args.training_source,
            args.evaluation_source,
            args.output_root,
            dataset_id=args.dataset_id,
            dataset_version=args.dataset_version,
            approval_id=args.approval_id,
            source_system=args.source_system,
            privacy_class=args.privacy_class,
        )
    except (DatasetContractError, GoldActivationError, OSError) as exc:
        result = {"status": "error", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=None if args.json else 2))
    return 0 if result.get("status") in {"materialized", "idempotent"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
