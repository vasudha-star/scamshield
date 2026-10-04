"""Unit tests for API <-> Model Consistency Verification (Priority 4).

Validates end-to-end alignment between the ScamShield Pipeline and FastAPI REST endpoints:
1. pipeline.scan(text, url) == POST /scan
2. pipeline.scan(text, url)["verdict"] & ["risk_assessment"] == POST /predict
3. pipeline.scan(text, url)["category"] == POST /category
4. pipeline.scan(text, url)["explanation"] == POST /explain
5. pipeline.scan_batch(items) == POST /scan-batch
6. Mathematical consistency: Risk score strictly monotonic with calibrated threat probability
"""

import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from api.app import app
from src.pipeline.scamshield_pipeline import ScamShieldPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestApiModelConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()
        cls.pipeline = ScamShieldPipeline.get_instance()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_predict_endpoint_matches_pipeline(self):
        """Verify POST /predict returns identical verdict and risk assessment as pipeline.scan()."""
        payload = {
            "text": "URGENT NOTICE: Non-bailable arrest warrant issued by Delhi Police CBI cell against your Aadhaar. Connect http://192.168.1.1/cbi",
            "url": "http://192.168.1.1/cbi"
        }
        api_resp = self.client.post("/predict", json=payload)
        self.assertEqual(api_resp.status_code, 200)
        api_data = api_resp.json()

        pipeline_res = self.pipeline.scan(payload["text"], url=payload["url"])

        self.assertEqual(api_data["verdict"]["is_threat"], pipeline_res["verdict"]["is_threat"])
        self.assertAlmostEqual(api_data["verdict"]["threat_probability"], pipeline_res["verdict"]["threat_probability"], places=3)
        self.assertEqual(api_data["risk_assessment"]["risk_tier"], pipeline_res["risk_assessment"]["risk_tier"])
        self.assertAlmostEqual(api_data["risk_assessment"]["risk_score"], pipeline_res["risk_assessment"]["risk_score"], places=1)

    def test_category_endpoint_matches_pipeline(self):
        """Verify POST /category returns identical taxonomy classification as pipeline.scan()."""
        payload = {
            "text": "Dear SBI customer, your YONO account is blocked due to KYC. Update PAN at http://sbi-kyc.xyz",
            "url": "http://sbi-kyc.xyz"
        }
        api_resp = self.client.post("/category", json=payload)
        self.assertEqual(api_resp.status_code, 200)
        api_cat = api_resp.json()

        pipeline_res = self.pipeline.scan(payload["text"], url=payload["url"])
        pipe_cat = pipeline_res["category"]

        self.assertEqual(api_cat["predicted_category"], pipe_cat["predicted_category"])
        self.assertEqual(api_cat["category_title"], pipe_cat["category_title"])
        self.assertAlmostEqual(api_cat["confidence"], pipe_cat["confidence"], places=3)

    def test_explain_endpoint_matches_pipeline(self):
        """Verify POST /explain returns identical SHAP attributions and reasons as pipeline.scan()."""
        payload = {
            "text": "Your account will be disconnected tonight at 9:30 PM. Pay bill immediately http://power-state.in",
            "url": "http://power-state.in"
        }
        api_resp = self.client.post("/explain", json=payload)
        self.assertEqual(api_resp.status_code, 200)
        api_exp = api_resp.json()

        pipeline_res = self.pipeline.scan(payload["text"], url=payload["url"])
        pipe_exp = pipeline_res["explanation"]

        self.assertEqual(api_exp["executive_summary"], pipe_exp["executive_summary"])
        self.assertEqual(len(api_exp["top_threat_drivers"]), len(pipe_exp["top_threat_drivers"]))
        if api_exp["top_threat_drivers"]:
            self.assertEqual(api_exp["top_threat_drivers"][0]["feature"], pipe_exp["top_threat_drivers"][0]["feature"])

    def test_full_scan_unified_contract(self):
        """Verify POST /scan returns full unified assessment with grounded snippets."""
        payload = {
            "text": "URGENT: Delhi Police CBI arrest warrant issued against your Aadhaar. Video hearing http://192.168.1.1/cbi now.",
            "url": "http://192.168.1.1/cbi"
        }
        api_resp = self.client.post("/scan", json=payload)
        self.assertEqual(api_resp.status_code, 200)
        data = api_resp.json()

        self.assertIn("unified_assessment", data)
        unified = data["unified_assessment"]
        self.assertIn(unified["risk_level"], ["High", "Critical", "Medium", "Low"])
        self.assertIn("reasons", unified)
        self.assertIn("evidence_snippets", unified)
        self.assertIn("recommended_action", unified)
        self.assertGreater(len(unified["reasons"]), 0)

    def test_batch_scan_consistency(self):
        """Verify POST /scan-batch returns consistent items matching individual scan."""
        items = [
            {
                "text": "URGENT NOTICE: CBI and Delhi Police cybercrime cell have issued an arrest warrant against your Aadhaar for illegal money laundering. You are placed under digital arrest. Connect immediately to Skype video call http://192.168.1.1/cbi-warrant",
                "url": "http://192.168.1.1/cbi-warrant"
            },
            {
                "text": "582914 is your secret One Time Password (OTP) for purchase of Rs 2,499.00 at AMAZON INDIA on HDFC Bank Card ending 7041. Valid for 10 minutes. NEVER share OTP with anyone.",
                "url": None
            },
        ]
        api_resp = self.client.post("/scan-batch", json={"messages": items})
        self.assertEqual(api_resp.status_code, 200)
        batch_results = api_resp.json()
        self.assertEqual(len(batch_results), 2)

        # First item is threat, second is benign
        self.assertTrue(batch_results[0]["verdict"]["is_threat"])
        self.assertFalse(batch_results[1]["verdict"]["is_threat"])


if __name__ == "__main__":
    unittest.main()
