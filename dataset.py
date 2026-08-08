#!/usr/bin/env python3
"""Prepare versioned MLOps datasets from local Gold/lake or test fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_CONTRACT = "solo-empire.mlops.dataset.v1"
SUPPORTED_SUFFIXES = {".json", ".jsonl", ".ndjson", ".parquet"}


class DatasetContractError(ValueError):
    """Raised when a versioned dataset cannot be loaded safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _resolve_path(value: str, *, base: Path = PROJECT_ROOT) -> Path:
    if value.startswith("file://"):
        parsed = urlparse(value)
        return Path(unquote(parsed.path)).expanduser().resolve()
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    from_base = (base / candidate).resolve()
    if from_base.exists():
        return from_base
    return (PROJECT_ROOT / candidate).resolve()


def _source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        files = sorted(
            item for item in source.rglob("*")
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if files:
            return files
    raise DatasetContractError(f"dataset source does not contain supported files: {source}")


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore
    except ImportError as exc:
        raise DatasetContractError(
            "Parquet dataset reads require PyArrow; install "
            "infra/requirements-data-lake.txt before using a real lake dataset"
        ) from exc
    return [dict(row) for row in parquet.read_table(path).to_pylist()]


def _read_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return _read_parquet(path)
    try:
        raw = path.read_text(encoding="utf-8")
        if suffix in {".jsonl", ".ndjson"}:
            value = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"cannot read dataset source: {path}") from exc
    if isinstance(value, dict):
        value = value.get("rows")
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise DatasetContractError(f"dataset source must contain a list of objects: {path}")
    return [dict(row) for row in value]


def _normalize_row(row: dict[str, Any], index: int, source: Path) -> dict[str, str]:
    candidate = dict(row)
    payload = candidate.get("payload_json")
    if isinstance(payload, str) and payload.strip():
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DatasetContractError(f"invalid payload_json at {source}:{index}") from exc
        if isinstance(parsed, dict):
            candidate = {**parsed, **candidate}
    row_id = str(
        candidate.get("id")
        or candidate.get("record_id")
        or candidate.get("source_record_id")
        or candidate.get("event_id")
        or ""
    ).strip()
    text = str(candidate.get("text") or candidate.get("input") or candidate.get("content") or "").strip()
    label = str(candidate.get("label") or candidate.get("target") or "").strip()
    missing = [name for name, value in (("id", row_id), ("text", text), ("label", label)) if not value]
    if missing:
        raise DatasetContractError(f"dataset row {index} in {source} is missing: {', '.join(missing)}")
    return {"id": row_id, "text": text, "label": label}


def load_dataset_rows(source: Path) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Load and validate rows, returning normalized rows and source evidence."""
    files = _source_files(source)
    rows: list[dict[str, str]] = []
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in files:
        source_rows = _read_file(path)
        normalized = [_normalize_row(row, index, path) for index, row in enumerate(source_rows)]
        for row in normalized:
            if row["id"] in seen:
                raise DatasetContractError(f"duplicate dataset id across source files: {row['id']}")
            seen.add(row["id"])
        rows.extend(normalized)
        evidence.append({
            "path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
            "sha256": f"sha256:{_sha256(path)}",
            "rows": len(normalized),
            "format": path.suffix.lower().lstrip(".") or "unknown",
        })
    if not rows:
        raise DatasetContractError(f"dataset source is empty: {source}")
    return rows, evidence


def load_dataset_contract(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"cannot read dataset contract: {path}") from exc
    if not isinstance(value, dict) or value.get("contract_version") != DATASET_CONTRACT:
        raise DatasetContractError(f"unsupported dataset contract: {path}")
    training = str(value.get("training_uri") or value.get("training", {}).get("uri") or "")
    evaluation = str(value.get("evaluation_uri") or value.get("evaluation", {}).get("uri") or "")
    if not training or not evaluation:
        raise DatasetContractError("dataset contract requires training_uri and evaluation_uri")
    return {
        **value,
        "contract_path": str(path.resolve()),
        "training_path": _resolve_path(training, base=path.parent),
        "evaluation_path": _resolve_path(evaluation, base=path.parent),
    }


def prepare_dataset_bundle(
    training_source: Path,
    evaluation_source: Path,
    output_dir: Path,
    *,
    dataset_id: str = "central-assistant-intent",
    dataset_version: str = "local-v1",
    source_layer: str = "gold",
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Normalize sources and emit immutable evidence for one dataset version."""
    training_rows, training_evidence = load_dataset_rows(training_source.resolve())
    evaluation_rows, evaluation_evidence = load_dataset_rows(evaluation_source.resolve())
    training_ids = {row["id"] for row in training_rows}
    overlap = training_ids & {row["id"] for row in evaluation_rows}
    if overlap:
        raise DatasetContractError(f"training/evaluation IDs overlap: {sorted(overlap)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "training.json"
    evaluation_path = output_dir / "evaluation.json"
    training_path.write_bytes(_canonical_json(training_rows))
    evaluation_path.write_bytes(_canonical_json(evaluation_rows))
    dataset_digest = hashlib.sha256(training_path.read_bytes() + evaluation_path.read_bytes()).hexdigest()

    source_paths = [Path(item["path"]) for item in training_evidence + evaluation_evidence]
    source_kind = "data_lake" if any("data/lake" in str(path) for path in source_paths) else "fixture" if any("infra/tests/fixtures" in str(path) for path in source_paths) else "local"
    manifest = {
        "contract_version": DATASET_CONTRACT,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "source_layer": source_layer,
        "source_kind": source_kind,
        "schema": {"required_columns": ["id", "text", "label"], "text_policy": "request text must not enter runtime metrics"},
        "training": {
            "path": str(training_path),
            "rows": len(training_rows),
            "sha256": f"sha256:{_sha256(training_path)}",
            "source_files": training_evidence,
        },
        "evaluation": {
            "path": str(evaluation_path),
            "rows": len(evaluation_rows),
            "sha256": f"sha256:{_sha256(evaluation_path)}",
            "source_files": evaluation_evidence,
        },
        "dataset_sha256": f"sha256:{dataset_digest}",
        "contract_source": str(contract_path.resolve()) if contract_path else None,
    }
    manifest_path = output_dir / "dataset-manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    manifest_sha256 = f"sha256:{_sha256(manifest_path)}"
    return {
        "manifest_path": str(manifest_path),
        "training_path": str(training_path),
        "evaluation_path": str(evaluation_path),
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "dataset_sha256": f"sha256:{dataset_digest}",
        "manifest_sha256": manifest_sha256,
        "source_kind": source_kind,
        "source_layer": source_layer,
        "training_examples": len(training_rows),
        "evaluation_examples": len(evaluation_rows),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a versioned MLOps dataset bundle")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="central-assistant-intent")
    parser.add_argument("--dataset-version", default="local-v1")
    parser.add_argument("--source-layer", default="gold")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        contract = load_dataset_contract(args.contract.resolve()) if args.contract else {}
        training = args.training.resolve() if args.training else contract.get("training_path")
        evaluation = args.evaluation.resolve() if args.evaluation else contract.get("evaluation_path")
        if not isinstance(training, Path) or not isinstance(evaluation, Path):
            raise DatasetContractError("provide --contract or both --training and --evaluation")
        result = prepare_dataset_bundle(
            training,
            evaluation,
            args.output_dir.resolve(),
            dataset_id=str(contract.get("dataset_id", args.dataset_id)),
            dataset_version=str(contract.get("dataset_version", args.dataset_version)),
            source_layer=str(contract.get("source_layer", args.source_layer)),
            contract_path=args.contract,
        )
    except (DatasetContractError, OSError) as exc:
        result = {"status": "error", "error": str(exc)}
    else:
        result["status"] = "pass"
    print(json.dumps(result, ensure_ascii=False, indent=None if args.json else 2))
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
