#!/usr/bin/env python3
"""Register immutable local model artifacts for the central MLOps pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

try:
    from .evaluate import _load_json, _resolve_artifact_path
except ImportError:  # pragma: no cover - exercised by direct CLI execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mlops.evaluate import _load_json, _resolve_artifact_path


NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class RegistryError(ValueError):
    """Raised when a candidate cannot be registered safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    os.replace(temporary, path)


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def register_model(manifest_path: Path, registry_dir: Path) -> dict[str, Any]:
    """Copy a digest-verified candidate into a deterministic local registry."""
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RegistryError("release manifest must be a JSON object")
    if manifest.get("contract_version") != "solo-empire.mlops.release.v1":
        raise RegistryError("unsupported MLOps contract version")

    model_name = str(manifest.get("model_name") or "")
    model_version = str(manifest.get("model_version") or "")
    if not NAME_RE.fullmatch(model_name) or not NAME_RE.fullmatch(model_version):
        raise RegistryError("model name and version must contain only safe path characters")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise RegistryError("manifest requires an artifact object")
    artifact_uri = str(artifact.get("uri") or "")
    declared_digest = str(artifact.get("sha256") or "")
    if not artifact_uri or not declared_digest.startswith("sha256:"):
        raise RegistryError("artifact uri and sha256 digest are required")
    source = _resolve_artifact_path(artifact_uri)
    if not source.is_file():
        raise RegistryError(f"artifact does not exist: {source}")
    actual_digest = _sha256(source)
    if declared_digest.removeprefix("sha256:") != actual_digest:
        raise RegistryError("artifact digest does not match the manifest")

    registry_dir = registry_dir.resolve()
    release_dir = registry_dir / model_name / model_version
    model_destination = release_dir / "model.json"
    manifest_destination = release_dir / "model-release.json"
    if model_destination.exists() and _sha256(model_destination) != actual_digest:
        raise RegistryError(f"registry conflict: {model_destination}")

    registered_manifest = json.loads(json.dumps(manifest))
    registered_manifest["artifact"]["uri"] = str(model_destination)
    manifest_bytes = _canonical_json(registered_manifest)
    if manifest_destination.exists() and manifest_destination.read_bytes() != manifest_bytes:
        raise RegistryError(f"registry conflict: {manifest_destination}")

    if not model_destination.exists():
        _atomic_write(model_destination, source.read_bytes())
    if not manifest_destination.exists():
        _atomic_write(manifest_destination, manifest_bytes)

    index_path = registry_dir / "index.json"
    index: dict[str, Any] = {"format": "solo-empire.local-registry.v1", "models": []}
    if index_path.exists():
        loaded_index = _load_json(index_path)
        if not isinstance(loaded_index, dict) or not isinstance(loaded_index.get("models"), list):
            raise RegistryError("registry index is invalid")
        index = loaded_index
    entries = [entry for entry in index["models"] if isinstance(entry, dict)]
    entry = {
        "model_name": model_name,
        "model_version": model_version,
        "artifact_sha256": f"sha256:{actual_digest}",
        "artifact_path": str(model_destination.relative_to(registry_dir)),
        "manifest_path": str(manifest_destination.relative_to(registry_dir)),
        "status": "registered",
    }
    entries = [
        existing for existing in entries
        if not (existing.get("model_name") == model_name and existing.get("model_version") == model_version)
    ]
    entries.append(entry)
    index["models"] = sorted(entries, key=lambda item: (str(item.get("model_name")), str(item.get("model_version"))))
    _atomic_write(index_path, _canonical_json(index))
    registered_versions = [
        str(item.get("model_version"))
        for item in index["models"]
        if item.get("model_name") == model_name
    ]
    return {
        "status": "registered",
        "model_name": model_name,
        "model_version": model_version,
        "artifact_sha256": f"sha256:{actual_digest}",
        "registry_path": str(model_destination),
        "index_path": str(index_path),
        "registered_versions": registered_versions,
        "rollback_target_available": any(version != model_version for version in registered_versions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a local MLOps model artifact")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--manifest", type=Path, required=True)
    register_parser.add_argument("--registry-dir", type=Path, required=True)
    register_parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = register_model(args.manifest.resolve(), args.registry_dir.resolve())
    except (OSError, RegistryError, json.JSONDecodeError) as exc:
        result = {"status": "error", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=None if args.json else 2))
    return 0 if result.get("status") == "registered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
