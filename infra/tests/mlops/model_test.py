#!/usr/bin/env python3
"""Reference Naive Bayes model: training, prediction and artifact round-trip."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from mlops.model import (
    MODEL_FORMAT,
    canonical_json,
    load_model,
    predict,
    save_model,
    tokenize,
    train_model,
)

EXAMPLES = [
    {"id": "1", "label": "billing", "text": "Refund my invoice payment"},
    {"id": "2", "label": "billing", "text": "Invoice total is wrong, refund please"},
    {"id": "3", "label": "tech", "text": "The app crashes on login"},
    {"id": "4", "label": "tech", "text": "Login error after the update crashes"},
]


class ModelTests(unittest.TestCase):
    def test_tokenize_is_case_folded_and_unicode_aware(self) -> None:
        # Thai vowel and tone marks are combining characters (category Mn); the old
        # [^\W_]+ pattern split "สวัสดี" into "สว" + "สด" and dropped the marks.
        self.assertEqual(
            tokenize("Hello, WORLD_x สวัสดี ครับ"), ["hello", "world", "x", "สวัสดี", "ครับ"]
        )
        self.assertEqual(tokenize("café naïve"), ["café", "naïve"])

    def test_training_counts_and_predicts_the_obvious_class(self) -> None:
        model = train_model(EXAMPLES, model_name="m", model_version="1")
        self.assertEqual(model["format"], MODEL_FORMAT)
        self.assertEqual(model["labels"], ["billing", "tech"])
        self.assertEqual(model["class_counts"], {"billing": 2, "tech": 2})
        self.assertEqual(model["training_examples"], 4)
        self.assertEqual(predict(model, "refund the invoice")[0], "billing")
        self.assertEqual(predict(model, "login crashes")[0], "tech")

    def test_unknown_text_falls_back_to_a_deterministic_label(self) -> None:
        model = train_model(EXAMPLES, model_name="m", model_version="1")
        first = predict(model, "zzz qqq")
        self.assertEqual(first, predict(model, "zzz qqq"))
        # Equal priors; for unseen tokens the class with fewer training tokens (billing: 10,
        # tech: 11) has the larger smoothed likelihood.
        self.assertEqual(first[0], "billing")
        tied = train_model(
            [
                {"id": "1", "label": "b", "text": "x"},
                {"id": "2", "label": "a", "text": "y"},
            ],
            model_name="m",
            model_version="1",
        )
        # A full tie breaks on the label name (max), so the result never depends on dict order.
        self.assertEqual(predict(tied, "unseen")[0], "b")

    def test_invalid_training_data_is_rejected(self) -> None:
        bad_sets = (
            [],
            [{"id": "1", "label": "a"}],
            [
                {"id": "1", "label": "a", "text": "x"},
                {"id": "1", "label": "b", "text": "y"},
            ],
            ["not an object"],
        )
        for rows in bad_sets:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                train_model(rows, model_name="m", model_version="1")  # type: ignore[arg-type]

    def test_predict_rejects_unknown_formats(self) -> None:
        with self.assertRaises(ValueError):
            predict({"format": "other", "labels": ["a"]}, "text")
        with self.assertRaises(ValueError):
            predict({"format": MODEL_FORMAT, "labels": []}, "text")

    def test_artifact_round_trip_is_byte_stable(self) -> None:
        model = train_model(EXAMPLES, model_name="m", model_version="1")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "model.json"
            save_model(model, path)
            self.assertEqual(path.read_bytes(), canonical_json(model))
            self.assertEqual(load_model(path), model)
            path.write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model(path)


if __name__ == "__main__":
    unittest.main()
