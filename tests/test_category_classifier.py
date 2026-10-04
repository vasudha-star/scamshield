"""Unit tests for Phase 13 Multi-Class Category Classifier using unittest."""

import json
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.models.category_classifier import (
    PROJECT_ROOT,
    TAXONOMY_MAP,
    load_categorized_data,
)


class TestCategoryClassifier(unittest.TestCase):
    def test_taxonomy_map_integrity(self):
        expected_categories = {
            "police_digital_arrest",
            "police_blackmail",
            "bank_kyc",
            "aadhaar",
            "lottery",
            "amazon",
            "relative",
            "legitimate",
        }
        for cat in expected_categories:
            self.assertIn(cat, TAXONOMY_MAP)
            self.assertIn(
                TAXONOMY_MAP[cat],
                {
                    "digital_arrest",
                    "extortion_blackmail",
                    "kyc_banking_identity",
                    "lottery_reward",
                    "impersonation_scam",
                    "legitimate",
                },
            )

    def test_data_filtering_rule_18(self):
        data_path = PROJECT_ROOT / "data" / "processed" / "cleaned.parquet"
        if not data_path.exists():
            self.skipTest("cleaned.parquet not found.")

        df = load_categorized_data(data_path)
        self.assertTrue((df["source_dataset"] == "indian_scam").all())
        self.assertFalse(df["taxonomy"].isna().any())
        self.assertTrue((df["taxonomy"] != "other").all())
        self.assertEqual(len(df), 743)

    def test_serialized_category_artifacts(self):
        models_dir = PROJECT_ROOT / "models"
        model_path = models_dir / "category_classifier.joblib"
        vec_path = models_dir / "category_vectorizer.joblib"
        meta_path = models_dir / "category_metadata.json"

        if not model_path.exists():
            self.skipTest("Category classifier model not yet trained.")

        model = joblib.load(model_path)
        vec = joblib.load(vec_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(vec, "transform"))
        self.assertEqual(len(meta["classes"]), 6)

        # Test inference on sample text
        sample_text = ["Aapka KYC pending hai account block ho jayega."]
        X_sample = vec.transform(sample_text)
        pred = model.predict(X_sample)
        probs = model.predict_proba(X_sample)

        self.assertEqual(len(pred), 1)
        self.assertEqual(probs.shape, (1, 6))
        self.assertTrue(np.isclose(probs.sum(), 1.0))


if __name__ == "__main__":
    unittest.main()
