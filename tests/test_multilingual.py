"""Unit tests for Phase 14 Multilingual Evaluation using unittest."""

import json
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.multilingual_eval import (
    PROJECT_ROOT,
    compute_slice_metrics,
)


class TestMultilingualEval(unittest.TestCase):
    def test_compute_slice_metrics_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        m = compute_slice_metrics(y_true, y_pred, y_prob)

        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["fpr"], 0.0)
        self.assertEqual(m["fnr"], 0.0)
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["tn"], 2)

    def test_multilingual_report_artifacts(self):
        reports_dir = PROJECT_ROOT / "reports"
        json_path = reports_dir / "multilingual_evaluation_report.json"
        md_path = reports_dir / "multilingual_evaluation_report.md"
        f1_plot = reports_dir / "multilingual_f1_by_language.png"
        cm_plot = reports_dir / "multilingual_confusion_matrices.png"

        self.assertTrue(json_path.exists(), "multilingual_evaluation_report.json missing")
        self.assertTrue(md_path.exists(), "multilingual_evaluation_report.md missing")
        self.assertTrue(f1_plot.exists(), "multilingual_f1_by_language.png missing")
        self.assertTrue(cm_plot.exists(), "multilingual_confusion_matrices.png missing")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["split_hash"], "85851abd839f4971")
        self.assertEqual(data["total_test_samples"], 17901)
        for lang in ["en", "hi", "hinglish", "te", "overall"]:
            self.assertIn(lang, data["language_metrics"])
            self.assertGreater(data["language_metrics"][lang]["f1"], 0.85)


if __name__ == "__main__":
    unittest.main()
