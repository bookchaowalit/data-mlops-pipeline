#!/usr/bin/env python3
"""Monitor categorical prediction drift without storing request text."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


class MonitoringError(ValueError):
    """Raised when prediction telemetry is malformed."""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise MonitoringError("prediction telemetry must be a JSON list")
    return value


def _distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(str(row["prediction"]) for row in rows)
    total = sum(counts.values())
    return {label: round(count / total, 6) for label, count in sorted(counts.items())}


def _validate_rows(rows: list[dict[str, Any]], name: str) -> None:
    if not rows:
        raise MonitoringError(f"{name} telemetry cannot be empty")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MonitoringError(f"{name} row {index} is not an object")
        row_id = str(row.get("id") or "")
        prediction = str(row.get("prediction") or "")
        if not row_id or not prediction:
            raise MonitoringError(f"{name} row {index} requires id and prediction")
        if row_id in seen:
            raise MonitoringError(f"duplicate {name} prediction id: {row_id}")
        seen.add(row_id)


def monitor_predictions(
    current_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    psi_threshold: float = 0.2,
) -> dict[str, Any]:
    """Return a deterministic PSI-based drift report for prediction labels."""
    _validate_rows(current_rows, "current")
    _validate_rows(baseline_rows, "baseline")
    if psi_threshold <= 0:
        raise MonitoringError("psi threshold must be positive")

    baseline_distribution = _distribution(baseline_rows)
    current_distribution = _distribution(current_rows)
    labels = sorted(set(baseline_distribution) | set(current_distribution))
    epsilon = 1e-6
    psi = 0.0
    for label in labels:
        baseline = max(baseline_distribution.get(label, 0.0), epsilon)
        current = max(current_distribution.get(label, 0.0), epsilon)
        psi += (current - baseline) * math.log(current / baseline)

    labeled_rows = [row for row in current_rows if str(row.get("label") or "")]
    accuracy = (
        round(
            sum(str(row["label"]) == str(row["prediction"]) for row in labeled_rows)
            / len(labeled_rows),
            6,
        )
        if labeled_rows
        else None
    )
    return {
        "events": len(current_rows),
        "baseline_events": len(baseline_rows),
        "accuracy": accuracy,
        "baseline_distribution": baseline_distribution,
        "current_distribution": current_distribution,
        "psi": round(psi, 6),
        "psi_threshold": psi_threshold,
        "drift_status": "fail" if psi >= psi_threshold else "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor MLOps prediction drift")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = monitor_predictions(
            _load_rows(args.predictions.resolve()),
            _load_rows(args.baseline.resolve()),
            psi_threshold=args.psi_threshold,
        )
    except (OSError, json.JSONDecodeError, MonitoringError) as exc:
        result = {"drift_status": "error", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=None if args.json else 2))
    return 0 if result.get("drift_status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
