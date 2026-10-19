"""
tests/test_delivery.py
───────────────────────
Unit tests for alert system and monthly brief generator.
"""

import json
import unittest
import numpy as np
import pandas as pd

from src.delivery.alerts import AlertSystem, ConflictAlert
from src.delivery.brief_generator import MonthlyBriefGenerator


class TestDeliverySystem(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame([
            {"admin1": "Zamfara", "forecast_month": "2024-11", "p_m2": 0.95, "low_coverage_flag": 1, "reliability_score": 0.42},
            {"admin1": "Kaduna", "forecast_month": "2024-11", "p_m2": 0.88, "low_coverage_flag": 0, "reliability_score": 0.78},
            {"admin1": "Lagos", "forecast_month": "2024-11", "p_m2": 0.15, "low_coverage_flag": 0, "reliability_score": 0.85},
        ])

    def test_alert_generation(self):
        alert_sys = AlertSystem(default_threshold=0.70)
        alerts = alert_sys.generate_alerts(self.sample_df, prob_col="p_m2")

        # Zamfara (0.95) and Kaduna (0.88) should breach threshold
        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0].state, "Zamfara")
        self.assertEqual(alerts[0].risk_level, "CRITICAL")
        self.assertTrue(alerts[0].is_underreporting_risk)

        self.assertEqual(alerts[1].state, "Kaduna")
        self.assertEqual(alerts[1].risk_level, "HIGH")
        self.assertFalse(alerts[1].is_underreporting_risk)

    def test_alert_bulletin_formatting(self):
        alert_sys = AlertSystem(default_threshold=0.70)
        alerts = alert_sys.generate_alerts(self.sample_df, prob_col="p_m2")
        bulletin = alert_sys.format_alert_bulletin(alerts)

        self.assertIn("Early Warning Alert Bulletin", bulletin)
        self.assertIn("Zamfara", bulletin)
        self.assertIn("Kaduna", bulletin)
        self.assertIn("HIGH BIAS RISK", bulletin)

    def test_brief_generator_templates(self):
        brief_gen = MonthlyBriefGenerator(self.sample_df)

        ngo_brief = brief_gen.generate_ngo_brief(target_month="2024-11")
        self.assertIn("Humanitarian Early Warning & Access Brief", ngo_brief)
        self.assertIn("Corridors at Risk", ngo_brief)
        self.assertIn("Zamfara", ngo_brief)

        gov_brief = brief_gen.generate_government_brief(target_month="2024-11")
        self.assertIn("Conflict Risk Assessment & Policy Early Warning", gov_brief)
        self.assertIn("Harmattan Dry Season Window", gov_brief)
        self.assertIn("Do No Digital Harm", gov_brief)

        corp_brief = brief_gen.generate_corporate_brief(target_month="2024-11")
        self.assertIn("Commercial Operations Security", corp_brief)
        self.assertIn("Supply Chain", corp_brief)


if __name__ == "__main__":
    unittest.main()
