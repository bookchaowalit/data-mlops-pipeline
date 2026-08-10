#!/usr/bin/env python3
"""Offline contract checks for the MLOps Prometheus/Grafana integration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROMETHEUS_CONFIG = ROOT / "self-hosted-stack/prometheus/prometheus.yml"
PROMETHEUS_TARGETS = ROOT / "self-hosted-stack/prometheus/mlops-targets.json"
PROMETHEUS_RULES = ROOT / "self-hosted-stack/prometheus/rules.yml"
GRAFANA_DASHBOARD = ROOT / "self-hosted-stack/grafana/provisioning/dashboards/mlops-reference.json"


class MLOpsObservabilityTests(unittest.TestCase):
    def test_scrape_target_is_empty_by_default(self):
        self.assertEqual(json.loads(PROMETHEUS_TARGETS.read_text(encoding="utf-8")), [])
        config = PROMETHEUS_CONFIG.read_text(encoding="utf-8")
        self.assertIn("solo-empire-mlops-reference", config)
        self.assertIn("mlops-targets.json", config)

    def test_alert_rules_are_present_and_disabled_until_target_exists(self):
        rules = PROMETHEUS_RULES.read_text(encoding="utf-8")
        self.assertIn("MLOpsReferenceServingDown", rules)
        self.assertIn("MLOpsReferenceErrorRateHigh", rules)
        self.assertIn("MLOpsReferenceModelMissing", rules)
        self.assertIn("empty by default", rules)

    def test_grafana_dashboard_has_runtime_queries(self):
        dashboard = json.loads(GRAFANA_DASHBOARD.read_text(encoding="utf-8"))
        self.assertEqual(dashboard["uid"], "solo-empire-mlops")
        expressions = [
            target["expr"]
            for panel in dashboard["panels"]
            for target in panel.get("targets", [])
        ]
        self.assertTrue(any("solo_empire_mlops_requests_total" in expr for expr in expressions))
        self.assertTrue(any("solo_empire_mlops_predictions_total" in expr for expr in expressions))
        self.assertTrue(any("solo_empire_mlops_errors_total" in expr for expr in expressions))
        self.assertTrue(any("solo_empire_mlops_model_info" in expr for expr in expressions))


if __name__ == "__main__":
    unittest.main()
