"""Unit tests for Phase 17 Unseen Scam LOCO Experiment using unittest."""

import json
import unittest
from pathlib import Path

from src.experiments.unseen_scam_loco import (
    CATEGORIES,
    PROJECT_ROOT,
    TAXONOMY_MAP,
)


class TestUnseenScamLoco(unittest.TestCase):
    def test_taxonomy_map_categories(self):
        for cat in CATEGORIES:
            self.assertIn(cat, TAXONOMY_MAP.values())

    def test_loco_report_artifacts(self):
        reports_dir = PROJECT_ROOT / "reports"
        json_path = reports_dir / "unseen_scam_loco_report.json"
        md_path = reports_dir / "unseen_scam_loco_report.md"
        plot_comp = reports_dir / "loco_unseen_scam_comparison.png"
        plot_trade = reports_dir / "loco_macro_recall_tradeoff.png"

        if not json_path.exists():
            self.skipTest("LOCO report not yet generated.")

        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())
        self.assertTrue(plot_comp.exists())
        self.assertTrue(plot_trade.exists())

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["split_hash"], "85851abd839f4971")
        for cat in CATEGORIES:
            self.assertIn(cat, data["loco_by_category"])
            for m in ["M1_Text", "M2_Text_URL", "M3_Text_Intent", "M4_Full"]:
                self.assertIn(m, data["loco_by_category"][cat])
                rec = data["loco_by_category"][cat][m]["zero_day_recall"]
                self.assertGreaterEqual(rec, 0.70)


if __name__ == "__main__":
    unittest.main()
