---
title: Central MLOps Contract
description: Model release, local runtime, and evaluation contracts for the Solo Empire central pipeline
status: local-promotion-evidence-ready
tags: [documentation, infra]
created: 2026-08-08
updated: 2026-08-10
---

# Central MLOps Contract

This directory is the repository-owned boundary for model release metadata and
local promotion evidence. It does not contain provider credentials, private
training rows from production systems, or a hosted deployment.

The local contract is intentionally small and reproducible:

```text
synthetic Gold fixture (versioned, no PII)
        -> local reference training
        -> immutable model artifact + prediction events
        -> deterministic evaluation and drift gates
        -> local registry
        -> DevSecOps gates
        -> manual promotion decision (fail closed)
```

## Checked-in local evidence

| Path | Role |
|---|---|
| `artifacts/solo-empire-central-assistant/0.2.0/model.json` | Digest-verified naive-Bayes JSON artifact |
| `artifacts/.../predictions.json` | Offline predictions used by the release gate |
| `gold/synthetic-gold-v1/` | Deterministic synthetic Gold training/evaluation bundle |
| `gold/synthetic-gold-v1/dataset-manifest.json` | Version, source hashes, lineage, privacy boundary |
| `model-release.json` | Candidate release manifest referencing the real artifact |
| `dataset-contract.json` | Integration point for the synthetic Gold package |

`model-release.json` is a **candidate** release manifest. Offline evaluation may
pass when the artifact digest and dataset lineage check out. That is **not**
deployment eligibility. Promotion remains blocked until human approval,
production-environment evidence, and a provisioned target exist. The repository
never marks this candidate as production-ready by default.

Privacy boundary for the synthetic Gold package:

- `contains_pii: false`
- `contains_credentials: false`
- `contains_provider_payloads: false`
- `boundary: synthetic-local-fixture-only`
- not a production data-lake claim

## Commands

Run from the repository root:

```bash
python3 infra/scripts/mlops/evaluate.py --json
python3 infra/scripts/mlops/gold.py \
  --training-source /path/to/approved-training.json \
  --evaluation-source /path/to/approved-evaluation.json \
  --output-root data/lake/gold/product=central-assistant/dataset=central-assistant-intent/version=gold-v1 \
  --dataset-version gold-v1 \
  --approval-id review-2026-08-08 \
  --json
python3 infra/scripts/mlops/dataset.py --contract infra/mlops/dataset-contract.json --output-dir "$(mktemp -d)" --json
python3 infra/scripts/mlops/pipeline.py \
  --output-dir "$(mktemp -d)" \
  --registry-dir "$(mktemp -d)" \
  --json
python3 infra/scripts/mlops/pipeline.py \
  --dataset-manifest infra/mlops/dataset-contract.json \
  --output-dir "$(mktemp -d)" \
  --registry-dir "$(mktemp -d)" \
  --json
python3 infra/scripts/mlops/serve.py --model infra/mlops/artifacts/solo-empire-central-assistant/0.2.0/model.json --host 127.0.0.1 --port 8081
python3 infra/scripts/mlops/monitor.py --predictions /path/to/predictions.json --baseline /path/to/baseline.json --json
python3 infra/tests/mlops/mlops_pipeline_test.py
```

## Limitations and next activation gate

What this lane proves today:

- immutable local model artifact with verifiable SHA-256
- synthetic Gold dataset version, source hashes, lineage, and privacy boundary
- deterministic offline evaluation gate
- fail-closed promotion without human approval or production evidence

What remains blocked / not claimed:

- hosted training, registry, or inference
- production lake-backed labeled data
- provider credentials or OpenAI-compatible generation as part of this release
- automatic promotion, GitOps deploy, or cluster mutation
- GPU runtime, vector DB, Redis, Kafka, or any new platform dependency

Next activation gate before any staging/production claim:

1. human approval ID recorded on the release manifest
2. production-environment evidence (target contract, restore, budget, owner)
3. staging environment provisioned with `deploymentEnabled` still gated
4. DevSecOps gates green for the promotion candidate
5. rollback target (previous digest + last-good evaluation) available
