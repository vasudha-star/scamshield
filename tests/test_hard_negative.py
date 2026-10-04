"""Unit tests for Phase 16 Hard-Negative Evaluation using unittest."""

import json
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.hard_negative_eval import (
    HARD_NEGATIVE_KEYWORDS,
    KEYWORD_REGEX,
    PROJECT_ROOT,
)


class TestHardNegative(unittest.TestCase):
    def test_keyword_regex(self):
        for kw in HARD_NEGATIVE_KEYWORDS:
            text = f"Your {kw} has been received."
            self.assertTrue(bool(KEYWORD_REGEX.search(text)))

        benign_sample = "Hello how are you doing today"
        self.assertFalse(bool(KEYWORD_REGEX.search(benign_sample)))

    def test_gold_benchmark_file(self):
        bm_path = PROJECT_ROOT / "data" / "gold" / "hard_negatives_benchmark.csv"
        self.assertTrue(bm_path.exists())

        df = pd.read_csv(bm_path)
        self.assertEqual(len(df), 40)
        self.assertTrue((df["ground_truth"] == 0).all())
        expected_cats = {"banking_otp", "security_2fa", "delivery_logistics", "statutory_gov", "multilingual"}
        self.assertEqual(set(df["category"].unique()), expected_cats)


if __name__ == "__main__":
    unittest.main()
