"""Unit tests for Phase 25: Statistical Rigor, Bootstrap CIs, and Calibration."""

import unittest
import numpy as np
from pathlib import Path

from src.evaluation.statistical_rigor import (
    compute_calibration_metrics,
    run_mcnemar_test,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


class TestStatisticalRigor(unittest.TestCase):
    def test_mcnemar_known_distribution(self):
        """Verify McNemar calculation against known hand-calculated contingency table."""
        # Setup: n01 = 20, n10 = 5.
        # Chi2 = (|20 - 5| - 1)^2 / (20 + 5) = (14)^2 / 25 = 196 / 25 = 7.84
        y_true = np.array([1] * 50)
        # Model A: correct for first 20, wrong for next 5, correct for remaining 25
        y_pred_a = np.array([1] * 20 + [0] * 5 + [1] * 25)
        # Model B: wrong for first 20, correct for next 5, correct for remaining 25
        y_pred_b = np.array([0] * 20 + [1] * 5 + [1] * 25)

        res = run_mcnemar_test(y_true, y_pred_a, y_pred_b, "ModelA", "ModelB")
        self.assertEqual(res["n01_a_correct_b_wrong"], 20)
        self.assertEqual(res["n10_a_wrong_b_correct"], 5)
        self.assertAlmostEqual(res["mcnemar_chi2"], 7.84, places=2)
        self.assertTrue(res["statistically_significant_05"])

    def test_calibration_metrics(self):
        """Verify ECE and Brier score computation."""
        # Perfectly confident & accurate
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.99, 0.99, 0.01, 0.01])
        cal = compute_calibration_metrics(y_true, y_prob, n_bins=5)
        self.assertLess(cal["brier_score"], 0.01)
        self.assertLess(cal["ece"], 0.05)

    def test_statistical_report_artifacts_exist(self):
        """Verify that statistical report files exist and contain required keys."""
        json_path = REPORTS_DIR / "statistical_significance_report.json"
        md_path = REPORTS_DIR / "statistical_significance_report.md"
        calib_img = REPORTS_DIR / "calibration_curves.png"
        mcnemar_img = REPORTS_DIR / "mcnemar_contingency_matrices.png"

        if json_path.exists():
            import json
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("bootstrap_confidence_intervals", data)
            self.assertIn("mcnemar_significance_tests", data)
            self.assertIn("calibration_metrics", data)

            self.assertTrue(md_path.exists())
            self.assertTrue(calib_img.exists())
            self.assertTrue(mcnemar_img.exists())


if __name__ == "__main__":
    unittest.main()
