# data-mlops-pipeline

Status as of 2026-10-23. Describes only what is in this repository today.

## Purpose

A small, reproducible reference MLOps pipeline for Solo Empire: dataset contract → train →
evaluate → local registry → staging dry run → local serving → drift monitoring, with a
fail-closed manual promotion gate. It exists to exercise the lifecycle and its gates without
credentials or heavy ML dependencies; it is not a production model.

## Users

The owner (lab / reference implementation). No external users, no hosted endpoint.

## What exists and works

- `infra/scripts/mlops/` (stdlib only): `dataset.py` (contract + bundle hashes), `gold.py`
  (synthetic Gold materialisation), `train.py`/`model.py`, `evaluate.py` (release gate),
  `registry.py`, `staging.py`, `serve.py` (HTTP predict on 127.0.0.1, request text not
  logged), `monitor.py` (PSI drift on prediction labels), `pipeline.py` (end-to-end slice).
- Checked-in evidence under `infra/mlops/`: candidate release manifest `model-release.json`,
  digest-verified artifact `artifacts/solo-empire-central-assistant/0.2.0/model.json`,
  offline predictions, and the synthetic Gold bundle `gold/synthetic-gold-v1`.
- Tests: `infra/tests/mlops/*_test.py` (pipeline/release gate, model, monitor), run by CI and by
  the "MLOps Release Gate" workflow, which also runs `evaluate.py` and the local pipeline.

## Model and evaluation status

- Model: `solo-empire-central-assistant` 0.2.0, multinomial Naive Bayes (JSON artifact,
  format `solo-empire.naive-bayes-json.v1`), 5 intent labels: action, finance, general,
  newsletter, work. Stage: **candidate**.
- Data: synthetic Gold `synthetic-gold-v1` (10 training rows, 5 evaluation rows, no PII,
  English only).
- Eval (2026-10-23, `python3 infra/scripts/mlops/evaluate.py --json`): accuracy 1.0, macro-F1
  1.0 on 5 examples; thresholds min 5 examples, accuracy ≥ 0.8, macro-F1 ≥ 0.75; artifact
  integrity and dataset lineage checks pass. Promotion stays blocked:
  `human-approval-required`, `production-environment-evidence-missing`.
- gap: the eval set is 5 synthetic rows, so the scores say the pipeline works, not that the
  model is useful; there is no real-data eval set.

## How to run

```bash
for t in infra/tests/mlops/*_test.py; do python3 "$t"; done
python3 infra/scripts/mlops/evaluate.py --json
python3 infra/scripts/mlops/pipeline.py --output-dir "$(mktemp -d)" --registry-dir "$(mktemp -d)" --json
```

## AI

No LLM provider is called. The pipeline's telemetry block marks provider token telemetry as
`not_applicable` (real AI run telemetry lives in the AI gateway).

## Limits

- Local only; no deployment, no runtime to monitor (so no Sentry). Prometheus/Grafana
  config for MLOps lives in the `homelab` repository, not here.
- No Python package manifest (stdlib-only scripts), so there is no lockfile or mypy config.

## Disposition

Keep as the lab reference for the MLOps lifecycle and its gates.
