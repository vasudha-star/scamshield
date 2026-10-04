"""Unit tests for Phase 21: Unified End-to-End Inference Pipeline."""

import unittest
from pathlib import Path

from src.pipeline.scamshield_pipeline import ScamShieldPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestScamShieldPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = ScamShieldPipeline.get_instance()

    def test_pipeline_singleton_and_health(self):
        pipe2 = ScamShieldPipeline.get_instance()
        self.assertIs(self.pipeline, pipe2, "Pipeline must follow singleton pattern")

        health = self.pipeline.get_health()
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["models_loaded"]["feature_dimensions"]["total_fused_features"], 15028)
        self.assertIn("digital_arrest", health["models_loaded"]["category_classes"])

    def test_pipeline_scan_scam(self):
        scam_text = (
            "URGENT NOTICE: CBI and Delhi Police cybercrime cell have issued an arrest warrant "
            "against your Aadhaar for illegal money laundering. Connect immediately to Skype video call "
            "http://192.168.1.1/cbi-warrant"
        )
        res = self.pipeline.scan(scam_text)
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["verdict"]["is_threat"])
        self.assertIn(res["verdict"]["threat_level"], ["HIGH", "CRITICAL"])
        self.assertEqual(res["category"]["predicted_category"], "digital_arrest")
        self.assertGreater(len(res["explanation"]["top_threat_drivers"]), 0)
        self.assertGreater(len(res["risk_assessment"]["action_checklist"]), 0)

    def test_pipeline_scan_benign(self):
        benign_text = "Hi team, please find attached the meeting minutes and project slide deck for review. Thanks, John."
        res = self.pipeline.scan(benign_text)
        self.assertEqual(res["status"], "success")
        self.assertFalse(res["verdict"]["is_threat"])
        self.assertEqual(res["verdict"]["threat_level"], "LOW")
        self.assertEqual(res["category"]["predicted_category"], "legitimate")
        self.assertLess(res["risk_assessment"]["risk_score"], 30.0)

    def test_pipeline_scan_batch(self):
        items = [
            "Hi team, see you at the standup.",
            "SBI Bank Account blocked. Update KYC immediately at http://sbi-kyc.xyz",
        ]
        results = self.pipeline.scan_batch(items)
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0]["verdict"]["is_threat"])
        self.assertTrue(results[1]["verdict"]["is_threat"])

    def test_pipeline_error_handling(self):
        res_empty = self.pipeline.scan("")
        self.assertEqual(res_empty["status"], "error")

        res_space = self.pipeline.scan("   ")
        self.assertEqual(res_space["status"], "error")


if __name__ == "__main__":
    unittest.main()
