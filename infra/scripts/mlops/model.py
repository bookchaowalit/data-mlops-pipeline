"""Small provider-neutral reference model for the central MLOps lab.

This is deliberately a deterministic multinomial Naive Bayes classifier. It is
not intended to compete with a real LLM or production ML stack; it provides a
real artifact, serving interface, and testable lifecycle for the central
pipeline without requiring credentials or heavyweight ML dependencies.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
MODEL_FORMAT = "solo-empire.naive-bayes-json.v1"


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _validate_examples(
    rows: list[dict[str, Any]], *, require_text: bool = True
) -> None:
    if not rows:
        raise ValueError("training/evaluation examples cannot be empty")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"example {index} is not an object")  # noqa: TRY004 - invalid input data, not a programming error; callers handle ValueError
        required = ("id", "label", "text") if require_text else ("id", "label")
        missing = [field for field in required if not str(row.get(field) or "").strip()]
        if missing:
            raise ValueError(f"example {index} is missing: {', '.join(missing)}")
        row_id = str(row["id"])
        if row_id in seen:
            raise ValueError(f"duplicate example id: {row_id}")
        seen.add(row_id)


def train_model(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    model_version: str,
) -> dict[str, Any]:
    _validate_examples(rows)
    labels = sorted({str(row["label"]) for row in rows})
    token_counts = {label: {} for label in labels}
    class_counts = {label: 0 for label in labels}
    total_token_counts = {label: 0 for label in labels}
    vocabulary: set[str] = set()

    for row in rows:
        label = str(row["label"])
        class_counts[label] += 1
        for token in tokenize(str(row["text"])):
            vocabulary.add(token)
            token_counts[label][token] = token_counts[label].get(token, 0) + 1
            total_token_counts[label] += 1

    return {
        "format": MODEL_FORMAT,
        "model_name": model_name,
        "model_version": model_version,
        "labels": labels,
        "class_counts": class_counts,
        "token_counts": token_counts,
        "total_token_counts": total_token_counts,
        "vocabulary_size": len(vocabulary),
        "training_examples": len(rows),
    }


def predict(model: dict[str, Any], text: str) -> tuple[str, dict[str, float]]:
    if model.get("format") != MODEL_FORMAT:
        raise ValueError("unsupported model format")
    labels = [str(label) for label in model.get("labels", [])]
    if not labels:
        raise ValueError("model has no labels")
    class_counts = model["class_counts"]
    token_counts = model["token_counts"]
    total_token_counts = model["total_token_counts"]
    vocabulary_size = max(int(model.get("vocabulary_size", 0)), 1)
    total_examples = sum(int(class_counts[label]) for label in labels)
    tokens = tokenize(text)
    scores: dict[str, float] = {}

    for label in labels:
        prior = (int(class_counts[label]) + 1) / (total_examples + len(labels))
        denominator = int(total_token_counts[label]) + vocabulary_size + 1
        score = math.log(prior)
        for token in tokens:
            likelihood = (int(token_counts[label].get(token, 0)) + 1) / denominator
            score += math.log(likelihood)
        scores[label] = score

    predicted = max(labels, key=lambda label: (scores[label], label))
    return predicted, scores


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def save_model(model: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(model))


def load_model(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model artifact must be a JSON object")  # noqa: TRY004 - invalid input data, not a programming error; callers handle ValueError
    return value
