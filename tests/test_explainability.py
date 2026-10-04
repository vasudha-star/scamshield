"""Unit tests for Phase 19 & 20: Explainability & Calibrated Risk Scoring."""

import json
import unittest
from pathlib import Path

from src.explainability.risk_scorer import ScamShieldRiskScorer
from src.explainability.rule_explainer import RuleBasedExplainer
from src.explainability.shap_explainer import ScamShieldSHAPExplainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestExplainabilityAndRiskScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shap_explainer = ScamShieldSHAPExplainer()
        cls.rule_explainer = RuleBasedExplainer()
        cls.risk_scorer = ScamShieldRiskScorer()

    def test_shap_explainer_initialization(self):
        self.assertEqual(self.shap_explainer.n_features, 15028, "Total features must be 15,028")
        self.assertEqual(len(self.shap_explainer.weights), 15028, "Weights vector must have 15,028 elements")
        self.assertIsInstance(self.shap_explainer.intercept, float, "Intercept must be a float")

    def test_global_feature_importance(self):
        global_info = self.shap_explainer.get_global_feature_importance(top_n=20)
        self.assertIn("top_threat_features", global_info)
        self.assertIn("top_benign_features", global_info)
        self.assertEqual(len(global_info["top_threat_features"]), 20)
        self.assertEqual(len(global_info["top_benign_features"]), 20)

        # Confirm modalities are identified
        modalities = {f["modality"] for f in global_info["top_threat_features"]}
        self.assertTrue("TEXT" in modalities or "INTENT" in modalities or "URL" in modalities)

    def test_local_explanation(self):
        scam_text = (
            "URGENT: Your CBI arrest warrant has been issued. Click http://192.168.1.1/cbi to verify "
            "immediately or police will arrest you within 2 hours."
        )
        exp = self.shap_explainer.explain_text(scam_text, top_k=5)
        self.assertIn("predicted_probability", exp)
        self.assertGreater(exp["predicted_probability"], 0.70, "Scam text probability must be > 0.70")
        self.assertGreater(len(exp["threat_attributions"]), 0, "Must have threat attributions")
        self.assertGreater(exp["total_active_features"], 0, "Must have active features")

    def test_rule_explainer(self):
        text = "URGENT: Delhi Police cybercrime cell warrants issued. Connect to video call immediately."
        res = self.rule_explainer.explain(text, predicted_probability=0.92, category="digital_arrest")

        self.assertIn("summary", res)
        self.assertIn("evidence_pillars", res)
        self.assertIn("recommendations", res)

        pillars = res["evidence_pillars"]
        self.assertGreater(len(pillars["textual_evidence"]), 0, "Must find threat keywords")
        self.assertGreater(len(pillars["psychological_intent_evidence"]), 0, "Must find intent signals")

    def test_rule_explainer_hard_negative(self):
        otp_text = "Your HDFC Bank NetBanking OTP is 123456. Valid for 10 minutes. Never share OTP."
        res = self.rule_explainer.explain(otp_text, predicted_probability=0.10, category="legitimate")
        self.assertIn("Legitimate", res["summary"], "Must explain legitimate status for hard-negative")

    def test_risk_scorer_tiers(self):
        # LOW
        res_low = self.risk_scorer.calculate_risk(predicted_probability=0.05)
        self.assertEqual(res_low["risk_tier"], "LOW")
        self.assertLess(res_low["risk_score"], 30.0)

        # MEDIUM
        res_med = self.risk_scorer.calculate_risk(predicted_probability=0.45)
        self.assertEqual(res_med["risk_tier"], "MEDIUM")
        self.assertTrue(30.0 <= res_med["risk_score"] < 60.0)

        # HIGH
        res_high = self.risk_scorer.calculate_risk(predicted_probability=0.75)
        self.assertEqual(res_high["risk_tier"], "HIGH")
        self.assertTrue(60.0 <= res_high["risk_score"] < 85.0)

        # CRITICAL
        res_crit = self.risk_scorer.calculate_risk(predicted_probability=0.95)
        self.assertEqual(res_crit["risk_tier"], "CRITICAL")
        self.assertGreaterEqual(res_crit["risk_score"], 85.0)

    def test_report_artifacts_exist(self):
        reports_dir = PROJECT_ROOT / "reports"
        self.assertTrue((reports_dir / "explainability_report.md").exists())
        self.assertTrue((reports_dir / "explainability_report.json").exists())
        self.assertTrue((reports_dir / "shap_global_feature_importance.png").exists())
        self.assertTrue((reports_dir / "shap_local_waterfall_scam.png").exists())
        self.assertTrue((reports_dir / "shap_local_waterfall_benign.png").exists())
        self.assertTrue((reports_dir / "risk_score_distribution.png").exists())


if __name__ == "__main__":
    unittest.main()
