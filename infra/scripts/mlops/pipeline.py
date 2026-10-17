#!/usr/bin/env python3
"""Run the local train -> evaluate -> register -> monitor MLOps slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
from _env import PROJECT_ROOT  # noqa: E402

if __package__ in {None, ""}:  # pragma: no cover - direct CLI support
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mlops.evaluate import evaluate_manifest
    from mlops.dataset import load_dataset_contract, prepare_dataset_bundle
    from mlops.monitor import monitor_predictions
    from mlops.registry import register_model
    from mlops.staging import staging_dry_run
    from mlops.train import load_examples, train_release
else:
    from .dataset import load_dataset_contract, prepare_dataset_bundle
    from .evaluate import evaluate_manifest
    from .monitor import monitor_predictions
    from .registry import register_model
    from .staging import staging_dry_run
    from .train import load_examples, train_release


DEFAULT_TRAINING = PROJECT_ROOT / "infra/tests/fixtures/mlops/training_examples.json"
DEFAULT_EVALUATION = (
    PROJECT_ROOT / "infra/tests/fixtures/mlops/evaluation_examples.json"
)
TELEMETRY_CONTRACT = "solo-empire.ai-telemetry.v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def run_pipeline(
    output_dir: Path,
    registry_dir: Path | None = None,
    *,
    training_path: Path = DEFAULT_TRAINING,
    evaluation_path: Path = DEFAULT_EVALUATION,
    dataset_manifest_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if dataset_manifest_path is not None:
        contract = load_dataset_contract(dataset_manifest_path.resolve())
        training_path = contract["training_path"]
        evaluation_path = contract["evaluation_path"]
        dataset = prepare_dataset_bundle(
            training_path,
            evaluation_path,
            output_dir / "dataset",
            dataset_id=str(contract.get("dataset_id", "central-assistant-intent")),
            dataset_version=str(contract.get("dataset_version", "local-v1")),
            source_layer=str(contract.get("source_layer", "gold")),
            contract_path=dataset_manifest_path,
        )
    else:
        training_path = training_path.resolve()
        evaluation_path = evaluation_path.resolve()
        dataset = prepare_dataset_bundle(
            training_path,
            evaluation_path,
            output_dir / "dataset",
            dataset_id="central-assistant-intent",
            dataset_version="fixture-v1",
            source_layer="fixture",
        )
    trained = train_release(
        Path(dataset["training_path"]),
        Path(dataset["evaluation_path"]),
        output_dir,
        dataset_lineage=dataset,
    )
    evaluation = evaluate_manifest(Path(trained["manifest_path"]))
    evaluation_report_path = output_dir / "evaluation-report.json"
    _write_json(evaluation_report_path, evaluation)

    baseline_rows = [
        {"id": str(row["id"]), "prediction": str(row["label"])}
        for row in load_examples(Path(dataset["training_path"]))
    ]
    baseline_path = output_dir / "baseline_predictions.json"
    _write_json(baseline_path, baseline_rows)
    current_rows = json.loads(
        Path(trained["predictions_path"]).read_text(encoding="utf-8")
    )
    monitoring = monitor_predictions(current_rows, baseline_rows)
    monitoring_report_path = output_dir / "monitoring-report.json"
    _write_json(monitoring_report_path, monitoring)

    if evaluation.get("release_gate") == "pass" and registry_dir is not None:
        registry = register_model(
            Path(trained["manifest_path"]), registry_dir.resolve()
        )
    else:
        registry = {
            "status": "skipped",
            "reason": "release gate failed or no registry directory was supplied",
        }
    staging = staging_dry_run(Path(trained["manifest_path"]), evaluation, registry)
    staging_report_path = output_dir / "staging-dry-run.json"
    _write_json(staging_report_path, staging)
    status = (
        "pass"
        if evaluation.get("release_gate") == "pass"
        and monitoring.get("drift_status") == "pass"
        else "fail"
    )
    return {
        "status": status,
        "promotion_mode": "manual",
        "telemetry": {
            "contract": TELEMETRY_CONTRACT,
            "status": "not_applicable",
            "reason": "local reference model has no provider token usage; real AI run telemetry lives in solo-ai-gateway",
            "required_fields": [
                "route",
                "provider",
                "model",
                "prompt_version",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "token_budget",
                "latency_ms",
                "estimated_cost_usd",
                "cache_hit",
                "status",
            ],
        },
        "dataset": dataset,
        "train": {**trained, "baseline_path": str(baseline_path)},
        "evaluation": evaluation,
        "monitoring": monitoring,
        "registry": registry,
        "staging_dry_run": staging,
        "evidence": {
            "evaluation_report": str(evaluation_report_path),
            "monitoring_report": str(monitoring_report_path),
            "staging_dry_run": str(staging_report_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local central MLOps pipeline")
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument(
        "--dataset-manifest", type=Path, help="Versioned Gold/lake dataset contract"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/solo-empire-mlops-release")
    )
    parser.add_argument("--registry-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = run_pipeline(
            args.output_dir,
            args.registry_dir,
            training_path=args.training,
            evaluation_path=args.evaluation,
            dataset_manifest_path=args.dataset_manifest,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"status": "error", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=None if args.json else 2))
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
