"""Unit tests for Improvement 4: Explanation Consistency and Faithfulness.

Ensures that the Explainability Engine (SHAP + Rule-based narrative) is strictly faithful:
1. Absent features must never be reported as active drivers.
2. If no URL is provided, no URL structural red flags or URL SHAP drivers may be claimed.
3. If urgency lexical terms and intent score are zero, urgency must not be claimed in the narrative.
4. Contrast consistency: Coercive threats must produce strictly higher intent scores than benign neutral text.
"""

import unittest
from pathlib import Path

from src.explainability.rule_explainer import RuleBasedExplainer
from src.explainability.shap_explainer import ScamShieldSHAPExplainer
from src.pipeline.scamshield_pipeline import ScamShieldPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestExplanationConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = ScamShieldPipeline.get_instance()
        cls.rule_explainer = RuleBasedExplainer()

    def test_no_url_produces_no_url_red_flags(self):
        """When URL is absent, explanation must NOT report URL red flags."""
        message = "Dear customer, your statement is ready for review."
        res = self.pipeline.scan(message, url=None)
        
        url_ev = res["explanation"]["evidence_pillars"]["url_structural_evidence"]
        self.assertEqual(len(url_ev), 0, f"Expected 0 URL evidence entries when URL is None, got: {url_ev}")

    def test_neutral_message_has_no_high_urgency_claim(self):
        """When message has no urgency indicators, narrative must not claim high urgency."""
        message = "The library will be open on Sunday from 10am to 4pm."
        res = self.pipeline.scan(message, url=None)
        
        intent_ev = res["explanation"]["evidence_pillars"]["psychological_intent_evidence"]
        urgency_claimed = any("urgency" in i.get("intent", "").lower() for i in intent_ev)
        self.assertFalse(urgency_claimed, f"Neutral text should not trigger urgency vector, got: {intent_ev}")

    def test_contrast_consistency_urgency(self):
        """Coercive text with deadline must produce higher intent/risk than polite neutral text."""
        coercive = "IMMEDIATE ACTION REQUIRED: Arrest warrant issued! You must pay within 2 hours or face police raid!"
        neutral = "Thank you for attending the community meeting yesterday. Next meeting is scheduled for next month."
        
        res_coercive = self.pipeline.scan(coercive)
        res_neutral = self.pipeline.scan(neutral)
        
        self.assertGreater(
            res_coercive["risk_assessment"]["risk_score"],
            res_neutral["risk_assessment"]["risk_score"],
            "Coercive message must have strictly higher risk score than neutral text"
        )
        self.assertTrue(res_coercive["verdict"]["is_threat"])
        self.assertFalse(res_neutral["verdict"]["is_threat"])

    def test_absent_credential_solicitation_not_flagged(self):
        """A notification without password/OTP solicitation must not claim credential harvesting."""
        message = "Your package has been delivered to your front porch. Have a great day."
        res = self.pipeline.scan(message)
        
        intent_ev = res["explanation"]["evidence_pillars"]["psychological_intent_evidence"]
        cred_claimed = any("credential" in i.get("intent", "").lower() for i in intent_ev)
        self.assertFalse(cred_claimed, f"Delivery notice should not trigger credential vector: {intent_ev}")


if __name__ == "__main__":
    unittest.main()
