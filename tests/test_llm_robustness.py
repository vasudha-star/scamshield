"""Unit tests for Phase 18: Human vs. LLM-Generated Phishing Robustness."""

import json
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestLLMRobustness(unittest.TestCase):
    def test_zenodo_dataset_split(self):
        parquet_path = PROJECT_ROOT / "data" / "processed" / "cleaned.parquet"
        self.assertTrue(parquet_path.exists(), "cleaned.parquet must exist")
        df = pd.read_parquet(parquet_path)
        test_df = df[df["split"] == "test"]
        llm_test = test_df[test_df["source_dataset"] == "llm_phishing"]

        self.assertEqual(len(llm_test), 1498, "Test LLM phishing benchmark must contain 1,498 samples")
        self.assertEqual(
            (llm_test["augmentation_type"] == "none").sum(),
            763,
            "Human phishing in test split must equal 763",
        )
        self.assertEqual(
            (llm_test["augmentation_type"] == "llm_generated").sum(),
            735,
            "LLM phishing in test split must equal 735",
        )

    def test_report_artifacts_exist(self):
        reports_dir = PROJECT_ROOT / "reports"
        json_path = reports_dir / "llm_robustness_report.json"
        md_path = reports_dir / "llm_robustness_report.md"
        plot_rates = reports_dir / "llm_vs_human_detection_rates.png"
        plot_dens = reports_dir / "llm_vs_human_probability_density.png"
        plot_intent = reports_dir / "llm_intent_feature_comparison.png"
        plot_len = reports_dir / "llm_text_length_distribution.png"

        self.assertTrue(json_path.exists(), "JSON report must exist")
        self.assertTrue(md_path.exists(), "Markdown report must exist")
        self.assertTrue(plot_rates.exists(), "Detection rates plot must exist")
        self.assertTrue(plot_dens.exists(), "Probability density plot must exist")
        self.assertTrue(plot_intent.exists(), "Intent comparison plot must exist")
        self.assertTrue(plot_len.exists(), "Length distribution plot must exist")

    def test_detection_rate_integrity(self):
        json_path = PROJECT_ROOT / "reports" / "llm_robustness_report.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["split_hash"], "85851abd839f4971")
        llm_results = data["results"]["llm_generated_phishing"]
        human_results = data["results"]["human_phishing"]

        # All classical models (M1-M4) must have 100% recall on LLM phishing
        for m in ["M1_Text", "M2_Text_URL", "M3_Text_Intent", "M4_Full"]:
            self.assertEqual(
                llm_results[m]["recall"],
                1.0,
                f"{m} must achieve 100% recall on LLM-generated phishing",
            )
            self.assertEqual(
                llm_results[m]["false_negatives"],
                0,
                f"{m} must have 0 false negatives on LLM phishing",
            )

        # M4 recall on human phishing must exceed 97%
        self.assertGreaterEqual(
            human_results["M4_Full"]["recall"],
            0.97,
            "M4 must achieve >= 97% recall on human phishing",
        )

        # XLM-RoBERTa recall must exceed 95% on LLM phishing
        self.assertGreaterEqual(
            llm_results["XLM_RoBERTa"]["recall"],
            0.95,
            "XLM-RoBERTa must achieve >= 95% recall on LLM phishing",
        )

    def test_intent_feature_divergence(self):
        json_path = PROJECT_ROOT / "reports" / "llm_robustness_report.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        intent_comp = data["profiles"]["intent"]
        # Urgency and Credential Request must be substantially elevated in LLM phishing
        self.assertGreater(
            intent_comp["urgency"]["llm_mean"],
            intent_comp["urgency"]["human_mean"],
            "Urgency activation must be higher in LLM phishing",
        )
        self.assertGreater(
            intent_comp["credential_request"]["llm_mean"],
            intent_comp["credential_request"]["human_mean"],
            "Credential request activation must be higher in LLM phishing",
        )
        self.assertGreater(
            intent_comp["social_engineering_score"]["llm_mean"],
            intent_comp["social_engineering_score"]["human_mean"],
            "Composite social engineering score must be higher in LLM phishing",
        )


if __name__ == "__main__":
    unittest.main()
