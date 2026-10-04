"""Unit tests for Phase 15 Telugu Low-Resource Experiment using unittest."""

import json
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.experiments.telugu_low_resource import (
    PROJECT_ROOT,
    evaluate_predictions,
)


class TestTeluguExperiment(unittest.TestCase):
    def test_evaluate_predictions_bounds(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        metrics = evaluate_predictions(y_true, y_pred)

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.6667, places=3)
        self.assertEqual(metrics["accuracy"], 0.75)

    def test_telugu_report_artifacts(self):
        reports_dir = PROJECT_ROOT / "reports"
        json_path = reports_dir / "telugu_low_resource_report.json"
        md_path = reports_dir / "telugu_low_resource_report.md"
        plot_path = reports_dir / "telugu_learning_curve.png"
        model_path = PROJECT_ROOT / "models" / "telugu_champion_model.joblib"

        if not json_path.exists():
            self.skipTest("Telugu report not yet generated.")

        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())
        self.assertTrue(plot_path.exists())
        self.assertTrue(model_path.exists())

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["split_hash"], "85851abd839f4971")
        self.assertIn("monolingual_scaling", data)
        self.assertIn("global_crosslingual_scaling", data)
        self.assertEqual(len(data["monolingual_scaling"]), 6)
        self.assertEqual(len(data["global_crosslingual_scaling"]), 6)


if __name__ == "__main__":
    unittest.main()
