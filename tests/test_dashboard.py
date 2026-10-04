"""Unit tests for Phase 22: Streamlit Interactive Web Dashboard."""

import py_compile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDashboardApp(unittest.TestCase):
    def test_dashboard_compiles(self):
        app_path = PROJECT_ROOT / "dashboard" / "app.py"
        self.assertTrue(app_path.exists(), "dashboard/app.py must exist")
        # Compile to ensure no syntax errors
        py_compile.compile(str(app_path), doraise=True)

    def test_dashboard_referenced_report_assets(self):
        reports_dir = PROJECT_ROOT / "reports"
        required_plots = [
            "M1_M4_ablation_dashboard.png",
            "M1_M4_PRAUC_comparison.png",
            "telugu_learning_curve.png",
            "xlm_vs_m4_multilingual_comparison.png",
            "hard_negative_in_distribution_comparison.png",
            "llm_vs_human_detection_rates.png",
            "llm_intent_feature_comparison.png",
            "shap_global_feature_importance.png",
            "risk_score_distribution.png",
            "category_confusion_matrix.png",
        ]
        for plot_name in required_plots:
            plot_file = reports_dir / plot_name
            self.assertTrue(plot_file.exists(), f"Dashboard requires report asset: {plot_name}")


if __name__ == "__main__":
    unittest.main()
