---
title: Local Synthetic Gold Datasets
description: Versioned synthetic Gold packages for local MLOps promotion evidence
status: local-reference-only
updated: 2026-08-10
---

# Local Synthetic Gold Datasets

Repository-owned synthetic Gold packages used for local MLOps evaluation and
lineage evidence. These are **not** production lake datasets.

## synthetic-gold-v1

| Field | Value |
|---|---|
| Dataset ID | `central-assistant-intent` |
| Version | `synthetic-gold-v1` |
| Source kind | `fixture` |
| Training rows | 10 |
| Evaluation rows | 5 |
| Dataset SHA-256 | `sha256:61d3483c73cf9d85cdf34614c8092b485c6cb8b99b30aec670a3995dec41d676` |
| Privacy class | `internal` |
| PII / credentials / provider payloads | none |
| Boundary | `synthetic-local-fixture-only` |

Contents:

- `training.json` / `evaluation.json` — labeled synthetic text/label rows
- `dataset-contract.json` — materialization contract + approval metadata
- `dataset-manifest.json` — source file digests and lineage for the release gate

Upstream fixtures live under `infra/tests/fixtures/mlops/`. Re-materialize an
approved private Gold version into the ignored `data/lake/` boundary with
`infra/scripts/mlops/gold.py` when real lake data is ready; that path remains
fail closed when missing.
