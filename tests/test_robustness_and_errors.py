"""Unit tests for Error Analysis and Adversarial Robustness Modules (Priorities 4 & 5)."""

import json
import unittest
from pathlib import Path

from src.evaluation.adversarial_robustness import (
    apply_code_mixing,
    apply_paraphrase,
    apply_spacing,
    apply_typos,
    apply_unicode_homoglyphs,
)
from src.evaluation.error_analysis import classify_error_cause

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


class TestRobustnessAndErrorAnalysis(unittest.TestCase):
    def test_perturbation_generators(self):
        """Verify adversarial perturbation functions alter text as expected."""
        orig = "Your account has been blocked. Verify immediately."
        
        # 1. Typos
        typo_text = apply_typos(orig)
        self.assertIn("acount", typo_text)
        self.assertIn("blokced", typo_text)
        self.assertIn("immediatly", typo_text)

        # 2. Spacing
        space_text = apply_spacing(orig)
        self.assertIn("b l o c k e d", space_text)

        # 3. Unicode Homoglyphs
        homo_text = apply_unicode_homoglyphs(orig)
        # Should contain Cyrillic characters (ord > 127)
        has_non_ascii = any(ord(ch) > 127 for ch in homo_text)
        self.assertTrue(has_non_ascii, "Homoglyph text should contain non-ASCII Unicode characters")

        # 4. Code-mixing (Hinglish)
        code_mix_orig = "Your account and debit card has been blocked today."
        code_mix = apply_code_mixing(code_mix_orig)
        self.assertIn("Aapka account aur debit card block ho gaya hai", code_mix)

    def test_error_cause_classification(self):
        """Verify heuristic error cause categorization."""
        # Benign 2FA OTP
        otp_cause = classify_error_cause("582914 is your secret OTP for Amazon purchase", 0, "meajor", "english")
        self.assertEqual(otp_cause, "Legitimate 2FA / OTP Verification")

        # Threat with Indic language
        indic_cause = classify_error_cause("Mee account block ayindi", 1, "dravidian_sms", "telugu")
        self.assertEqual(indic_cause, "Indic Script Subword OOV")

    def test_error_analysis_artifacts(self):
        """Verify error analysis report exists and has required schema."""
        json_path = REPORTS_DIR / "error_analysis_report.json"
        md_path = REPORTS_DIR / "error_analysis_report.md"
        img_path = REPORTS_DIR / "error_distribution.png"

        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())
        self.assertTrue(img_path.exists())

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("error_summary", data)
        self.assertIn("false_positive_examples", data)
        self.assertIn("false_negative_examples", data)
        self.assertIn("category_discrepancy_breakdown", data)
        self.assertAlmostEqual(data["category_discrepancy_breakdown"]["overall_accuracy"], 0.9018, places=2)


if __name__ == "__main__":
    unittest.main()
