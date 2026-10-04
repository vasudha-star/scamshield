"""Unit tests for Phase 23: Production FastAPI REST Backend."""

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestFastAPIBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Using TestClient as context manager to run lifespan startup
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("documentation", data)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(
            data["models_loaded"]["feature_dimensions"]["total_fused_features"],
            15028,
        )
        self.assertIn("digital_arrest", data["models_loaded"]["category_classes"])

    def test_predict_endpoint_scam(self):
        payload = {
            "text": (
                "URGENT NOTICE: CBI and Delhi Police cybercrime cell have issued an arrest warrant "
                "against your Aadhaar for illegal money laundering. You are placed under digital arrest. "
                "Connect immediately to Skype video call http://192.168.1.1/cbi-warrant"
            ),
            "url": "http://192.168.1.1/cbi-warrant",
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["verdict"]["is_threat"])
        self.assertIn(data["risk_assessment"]["risk_tier"], ["HIGH", "CRITICAL"])

    def test_category_endpoint(self):
        payload = {
            "text": "Dear SBI customer, your KYC is expired. Update PAN card at http://sbi-kyc.xyz",
        }
        response = self.client.post("/category", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["predicted_category"], ["kyc_banking_identity", "digital_arrest", "impersonation_scam"])
        self.assertIn("confidence", data)

    def test_explain_endpoint(self):
        payload = {
            "text": "Pay 0.5 Bitcoin immediately or webcam video will be leaked to your family within 24 hours.",
        }
        response = self.client.post("/explain", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("executive_summary", data)
        self.assertGreater(len(data["top_threat_drivers"]), 0)
        self.assertIn("evidence_pillars", data)

    def test_scan_endpoint_full(self):
        payload = {
            "text": "Hi team, please find attached the engineering presentation slides for review. Thanks, John.",
        }
        response = self.client.post("/scan", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertFalse(data["verdict"]["is_threat"])
        self.assertEqual(data["verdict"]["threat_level"], "LOW")
        self.assertEqual(data["category"]["predicted_category"], "legitimate")
        self.assertLess(data["risk_assessment"]["risk_score"], 30.0)

    def test_scan_batch_endpoint(self):
        payload = {
            "messages": [
                {"text": "Meeting at 3 PM today in boardroom. Thanks, John."},
                {"text": "Dear SBI Customer, your NetBanking account has been blocked due to pending KYC verification. Please click http://sbi-kyc-update.xyz/login to verify PAN card and enter your password immediately."},
            ]
        }
        response = self.client.post("/scan-batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertFalse(data[0]["verdict"]["is_threat"])
        self.assertTrue(data[1]["verdict"]["is_threat"])

    def test_validation_error_on_empty_text(self):
        payload = {"text": ""}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
