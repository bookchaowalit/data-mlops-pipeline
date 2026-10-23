#!/usr/bin/env python3
"""Prediction drift monitor: PSI maths, validation and fail-closed thresholds."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from mlops.monitor import MonitoringError, monitor_predictions


def rows(
    *predictions: str, labels: tuple[str, ...] | None = None
) -> list[dict[str, str]]:
    out = []
    for index, prediction in enumerate(predictions):
        row = {"id": f"r{index}", "prediction": prediction}
        if labels is not None:
            row["label"] = labels[index]
        out.append(row)
    return out


class MonitorTests(unittest.TestCase):
    def test_identical_distributions_have_zero_psi_and_pass(self) -> None:
        report = monitor_predictions(rows("a", "b", "a", "b"), rows("b", "a", "b", "a"))
        self.assertEqual(report["psi"], 0.0)
        self.assertEqual(report["drift_status"], "pass")
        self.assertEqual(report["current_distribution"], {"a": 0.5, "b": 0.5})

    def test_psi_matches_the_formula(self) -> None:
        report = monitor_predictions(rows("a", "a", "a", "b"), rows("a", "b", "b", "b"))
        expected = (0.75 - 0.25) * math.log(0.75 / 0.25) + (0.25 - 0.75) * math.log(
            0.25 / 0.75
        )
        self.assertAlmostEqual(report["psi"], round(expected, 6))
        self.assertEqual(report["drift_status"], "fail")

    def test_label_missing_from_baseline_counts_as_drift(self) -> None:
        report = monitor_predictions(rows("new", "new"), rows("old", "old"))
        self.assertEqual(report["drift_status"], "fail")
        self.assertEqual(set(report["current_distribution"]), {"new"})

    def test_accuracy_uses_only_labelled_rows(self) -> None:
        current = rows("a", "b", "a", labels=("a", "a", ""))
        report = monitor_predictions(current, rows("a", "b"))
        self.assertEqual(report["accuracy"], 0.5)
        self.assertIsNone(monitor_predictions(rows("a"), rows("a"))["accuracy"])

    def test_malformed_telemetry_is_rejected(self) -> None:
        for current in (
            [],
            [{"id": "x"}],
            [{"id": "x", "prediction": "a"}, {"id": "x", "prediction": "b"}],
            ["x"],
        ):
            with self.subTest(current=current), self.assertRaises(MonitoringError):
                monitor_predictions(current, rows("a"))  # type: ignore[arg-type]

    def test_threshold_must_be_a_positive_finite_number(self) -> None:
        # NaN compares false with everything, so a NaN threshold used to pass every run (fail open).
        for threshold in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(threshold=threshold), self.assertRaises(MonitoringError):
                monitor_predictions(
                    rows("a", "a"), rows("b", "b"), psi_threshold=threshold
                )


if __name__ == "__main__":
    unittest.main()
