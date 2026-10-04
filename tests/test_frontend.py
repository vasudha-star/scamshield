"""Unit tests for ScamShield Web Frontend and Static Mounting."""

import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from api.app import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestFrontendEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_frontend_app_index_served(self):
        """Verify that /app/ serves the index.html file."""
        response = self.client.get("/app/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ScamShield AI", response.text)
        self.assertIn("Defense Matrix", response.text)
        self.assertIn("Suspicious Communication Analyzer", response.text)

    def test_frontend_assets_served(self):
        """Verify style.css and app.js are properly served."""
        css_resp = self.client.get("/app/style.css")
        self.assertEqual(css_resp.status_code, 200)
        self.assertIn("Cyber Defense Design System", css_resp.text)

        js_resp = self.client.get("/app/app.js")
        self.assertEqual(js_resp.status_code, 200)
        self.assertIn("ScamShield AI", js_resp.text)

    def test_reports_static_served(self):
        """Verify empirical reports and figures are accessible."""
        resp = self.client.get("/reports/M1_M4_ablation_dashboard.png")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/png")

    def test_html_accept_root_negotiation(self):
        """Verify browser Accept header receives index.html at root /."""
        response = self.client.get("/", headers={"accept": "text/html,application/xhtml+xml"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("<!DOCTYPE html>", response.text)


if __name__ == "__main__":
    unittest.main()
