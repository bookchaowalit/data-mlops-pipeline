---
title: Local MLOps Model Artifacts
description: Digest-verified local reference model artifacts for the central MLOps pipeline
status: local-reference-only
updated: 2026-08-10
---

# Local MLOps Model Artifacts

This directory holds immutable **local reference** model artifacts used by the
central MLOps release contract. Artifacts are provider-neutral JSON models for
lifecycle evidence. They are not production deployments.

## Current candidate

| Field | Value |
|---|---|
| Model | `solo-empire-central-assistant` |
| Version | `0.2.0` |
| Type | `naive-bayes-json` |
| Path | `solo-empire-central-assistant/0.2.0/model.json` |
| SHA-256 | `sha256:0d6b3bc3476d39f23abcc71ebd07dc15854c84921277c621c51c354b010d8ee6` |
| Predictions | `solo-empire-central-assistant/0.2.0/predictions.json` |

Verify:

```bash
sha256sum infra/mlops/artifacts/solo-empire-central-assistant/0.2.0/model.json
python3 infra/scripts/mlops/evaluate.py --json
```

Do not treat a passing offline evaluation as production readiness. Promotion
still requires human approval and production-environment evidence.
