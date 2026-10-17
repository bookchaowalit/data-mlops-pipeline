#!/usr/bin/env python3
"""Create a non-mutating staging promotion rehearsal report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class StagingGateError(ValueError):
    """Raised when staging evidence cannot be evaluated."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StagingGateError(f"cannot read staging evidence: {path}") from exc


def staging_dry_run(
    manifest_path: Path,
    evaluation: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate staging prerequisites without changing GitOps or cluster state."""
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise StagingGateError("release manifest must be a JSON object")
    promotion = manifest.get("promotion")
    if not isinstance(promotion, dict):
        raise StagingGateError("release manifest requires promotion metadata")
    targets = promotion.get("target_environments", [])
    human_approval_id = promotion.get("human_approval_id")
    production_evidence = promotion.get("production_evidence")
    human_approval = bool(
        isinstance(human_approval_id, str) and human_approval_id.strip()
    )
    production_evidence_present = bool(
        isinstance(production_evidence, dict) and production_evidence
    ) or bool(isinstance(production_evidence, str) and production_evidence.strip())
    artifact_eligible = evaluation.get("artifact_eligible") is True or (
        evaluation.get("promotion_eligible") is True
        and evaluation.get("release_gate") == "pass"
    )
    checks = {
        "release_gate_passed": evaluation.get("release_gate") == "pass",
        "immutable_artifact_eligible": artifact_eligible,
        "registry_registered": registry.get("status") == "registered",
        "manual_promotion_mode": promotion.get("mode") == "manual",
        "staging_environment_provisioned": False,
        "human_approval": human_approval,
        "production_environment_evidence": production_evidence_present,
        "staging_target_declared": "staging" in targets,
        "rollback_target_available": registry.get("rollback_target_available") is True,
        "deployment_disabled": True,
    }
    # Always fail closed for environment promotion until humans and production
    # evidence exist. Local artifact eligibility never enables deployment.
    blockers = [
        "staging-environment-not-provisioned",
        "human-approval-required",
        "production-environment-evidence-missing",
        "deployment-disabled",
    ]
    always_blocked = {
        "staging_environment_provisioned",
        "human_approval",
        "production_environment_evidence",
        "deployment_disabled",
    }
    for name, passed in checks.items():
        if not passed and name not in always_blocked:
            blockers.append(name.replace("_", "-") + "-failed")
    return {
        "environment": "staging",
        "status": "blocked",
        "deployment_enabled": False,
        "promotion_action": "none",
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "manifest_path": str(manifest_path.resolve()),
        "model_name": manifest.get("model_name"),
        "model_version": manifest.get("model_version"),
        "safety": {
            "cluster_mutated": False,
            "gitops_changed": False,
            "secrets_read": False,
            "production_claim_allowed": False,
        },
        "rollback": {
            "ready": registry.get("rollback_target_available") is True,
            "registered_versions": registry.get("registered_versions", []),
            "strategy": "manual-revert-to-previous-immutable-digest",
            "required_evidence": [
                "previous-model-digest",
                "last-good-evaluation-report",
                "operator-approval",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a non-mutating staging promotion rehearsal"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        evaluation = _load_json(args.evaluation.resolve())
        registry = (
            _load_json(args.registry.resolve())
            if args.registry
            else {"status": "not-registered"}
        )
        if not isinstance(evaluation, dict) or not isinstance(registry, dict):
            raise StagingGateError(
                "evaluation and registry evidence must be JSON objects"
            )
        result = staging_dry_run(args.manifest.resolve(), evaluation, registry)
    except StagingGateError as exc:
        result = {"status": "error", "error": str(exc)}
    rendered = (
        json.dumps(result, ensure_ascii=False, indent=None if args.json else 2) + "\n"
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result.get("status") == "ready-for-manual-approval" else 1


if __name__ == "__main__":
    raise SystemExit(main())
