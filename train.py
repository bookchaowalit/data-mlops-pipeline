#!/usr/bin/env python3
"""Train the local reference model and create a candidate release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
from _env import PROJECT_ROOT  # noqa: E402

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mlops.model import load_model, predict, save_model, train_model
else:
    from .model import load_model, predict, save_model, train_model


DEFAULT_TRAINING = PROJECT_ROOT / "infra/tests/fixtures/mlops/training_examples.json"
DEFAULT_EVALUATION = PROJECT_ROOT / "infra/tests/fixtures/mlops/evaluation_examples.json"


def load_examples(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"examples must be a JSON list: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_release(
    training_path: Path = DEFAULT_TRAINING,
    evaluation_path: Path = DEFAULT_EVALUATION,
    output_dir: Path = Path("/tmp/solo-empire-mlops-release"),
    *,
    model_name: str = "solo-empire-central-assistant",
    model_version: str = "0.2.0",
    dataset_lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    training_rows = load_examples(training_path)
    evaluation_rows = load_examples(evaluation_path)
    model = train_model(
        training_rows,
        model_name=model_name,
        model_version=model_version,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.json"
    predictions_path = output_dir / "predictions.json"
    manifest_path = output_dir / "model-release.json"
    save_model(model, model_path)

    loaded_model = load_model(model_path)
    predictions = []
    for row in evaluation_rows:
        prediction, _scores = predict(loaded_model, str(row["text"]))
        predictions.append({
            "id": str(row["id"]),
            "label": str(row["label"]),
            "prediction": prediction,
        })
    predictions_path.write_text(
        json.dumps(predictions, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "contract_version": "solo-empire.mlops.release.v1",
        "model_name": model_name,
        "model_version": model_version,
        "stage": "candidate",
        "artifact": {
            "type": "naive-bayes-json",
            "uri": str(model_path),
            "sha256": f"sha256:{sha256(model_path)}",
        },
        "evaluation": {
            "dataset": str(evaluation_path),
            "predictions": str(predictions_path),
            "min_examples": len(evaluation_rows),
            "min_accuracy": 0.8,
            "min_macro_f1": 0.75,
        },
        "provenance": {
            "dataset": dataset_lineage or {
                "dataset_id": "unversioned-fixture",
                "dataset_version": "fixture",
                "source_kind": "fixture",
                "source_layer": "fixture",
                "dataset_sha256": None,
                "manifest_path": None,
            },
            "code_revision": (
                os.environ.get("GITHUB_SHA")
                or os.environ.get("CI_COMMIT_SHA")
                or os.environ.get("GIT_COMMIT")
                or "unavailable-local"
            ),
        },
        "promotion": {
            "mode": "manual",
            "target_environments": ["dev", "staging", "production"],
            "requires": [
                "offline-evaluation",
                "runtime-dependency-audit",
                "immutable-artifact-digest",
                "dataset-lineage",
                "human-approval",
                "production-environment-evidence",
            ],
            "human_approval_id": None,
            "production_evidence": None,
        },
    }
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )
    return {
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
        "manifest_path": str(manifest_path),
        "model_sha256": sha256(model_path),
        "training_examples": len(training_rows),
        "evaluation_examples": len(evaluation_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the local MLOps reference model")
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/solo-empire-mlops-release"))
    parser.add_argument("--model-name", default="solo-empire-central-assistant")
    parser.add_argument("--model-version", default="0.2.0")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = train_release(
        args.training.resolve(),
        args.evaluation.resolve(),
        args.output_dir.resolve(),
        model_name=args.model_name,
        model_version=args.model_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
