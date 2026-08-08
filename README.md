---
title: Central MLOps Contract
description: Model release, local runtime, and evaluation contracts for the Solo Empire central pipeline
status: phase-4-gold-activation-ready
tags: [documentation, infra]
created: 2026-08-08
updated: 2026-08-08
---

# Central MLOps Contract

This directory is the repository-owned boundary for model release metadata. It
does not contain model weights, prompts with private data, credentials, or a
provider-specific training system.

The local contract is intentionally small and reproducible:

```text
approved fixture or lake-backed dataset
        -> local reference training
        -> model artifact + prediction events
        -> deterministic evaluation and drift gates
        -> local registry
        -> DevSecOps gates
        -> manual promotion decision
```

`model-release.json` is a candidate release manifest. The Phase 4 local
pipeline can consume a versioned Gold/lake contract, records source and
dataset digests in provenance, creates a digest-verified JSON artifact, and
emits a non-mutating staging rehearsal. Hosted runtime telemetry, environment
deployment, and human approval remain separate requirements.

Run the gate from the repository root:

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
python3 infra/scripts/mlops/serve.py --model /path/to/model.json --host 127.0.0.1 --port 8081
python3 infra/scripts/mlops/monitor.py --predictions /path/to/predictions.json --baseline /path/to/baseline.json --json
```

The local registry is a filesystem evidence boundary, not a hosted model
registry. `dataset-contract.json` is the real-lake integration point; it must
resolve to a Gold dataset containing `id`, `text`, and `label`. A hosted
adapter can be added later behind the same manifest fields without making the
central pipeline depend on a single ML vendor.

`gold.py` never auto-labels, overwrites an existing version, or stores source
payloads in the manifest. It requires an operator approval ID and writes the
normalized private Gold rows plus a versioned contract under the ignored
`data/lake/` boundary. The checked-in
`infra/self-hosted-stack/prometheus/mlops-targets.json` is intentionally empty;
add a network-reachable serving target only after `serve.py` is running.
The provisioned Grafana dashboard is `Solo Empire - MLOps Reference`; the
Prometheus rules remain inactive while that target file is empty.
