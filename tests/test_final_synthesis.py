"""Unit tests for Phase 24: Final Benchmark Synthesis & Project Completion."""

import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestFinalSynthesis(unittest.TestCase):
    def test_synthesis_reports_exist(self):
        reports_dir = PROJECT_ROOT / "reports"
        synthesis_report = reports_dir / "FINAL_SCAMSHIELD_BENCHMARK_SYNTHESIS.md"
        viva_guide = reports_dir / "viva_defense_guide.md"

        self.assertTrue(synthesis_report.exists(), "Final benchmark synthesis report must exist")
        self.assertTrue(viva_guide.exists(), "Viva defense guide must exist")

        with open(synthesis_report, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("85851abd839f4971", content)
            self.assertIn("M4 (Full ScamShield Early Fusion)", content)
            self.assertIn("15,028", content)

        with open(viva_guide, "r", encoding="utf-8") as f:
            viva_content = f.read()
            self.assertIn("Top 15 Technical Questions", viva_content)
            self.assertIn("Early Fusion", viva_content)

    def test_master_audit_log_completeness(self):
        log_path = PROJECT_ROOT / "experiment_log.csv"
        self.assertTrue(log_path.exists(), "experiment_log.csv must exist")
        df_log = pd.read_csv(log_path)
        self.assertGreaterEqual(len(df_log), 14, "Audit log must contain at least 14 recorded phases")

        exp_ids = set(df_log["experiment_id"])
        required_exps = {
            "M1_TEXT_BASELINE",
            "M2_TEXT_URL",
            "M3_TEXT_INTENT",
            "M4_FULL_SCAMSHIELD",
            "CATEGORY_CLASSIFIER",
            "XLM_ROBERTA_MULTILINGUAL",
            "TELUGU_LOW_RESOURCE_CURVE",
            "HARD_NEGATIVE_BENCHMARK",
            "UNSEEN_SCAM_LOCO",
            "LLM_ROBUSTNESS_BENCHMARK",
            "EXPLAINABILITY_AND_RISK_SCORING",
            "END_TO_END_INFERENCE_PIPELINE",
            "STREAMLIT_INTERACTIVE_DASHBOARD",
            "FASTAPI_PRODUCTION_BACKEND",
        }
        for exp in required_exps:
            self.assertIn(exp, exp_ids, f"Experiment {exp} must be recorded in audit log")

    def test_all_serialized_champion_models_exist(self):
        models_dir = PROJECT_ROOT / "models"
        required_models = [
            "m1_text_baseline.joblib",
            "m2_text_url.joblib",
            "m3_text_intent.joblib",
            "m4_full_scamshield.joblib",
            "category_classifier.joblib",
            "category_vectorizer.joblib",
            "tfidf_vectorizer.joblib",
            "url_scaler.joblib",
            "intent_scaler.joblib",
        ]
        for m in required_models:
            self.assertTrue((models_dir / m).exists(), f"Model artifact {m} must exist")


if __name__ == "__main__":
    unittest.main()
