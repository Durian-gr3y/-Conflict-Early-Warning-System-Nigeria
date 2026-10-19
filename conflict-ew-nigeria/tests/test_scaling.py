"""
tests/test_scaling.py
─────────────────────
Unit tests for Phase 6 geographic scaling and ethical screening.
"""

import unittest
import numpy as np
import pandas as pd

from src.processing.geography import (
    load_states_config,
    get_state_centroids,
    get_pilot_states,
    get_enabled_states,
    validate_geographic_coverage,
)
from src.delivery.screening import ClientEthicsScreening


class TestGeographicScaling(unittest.TestCase):

    def test_states_registry_loading(self):
        states = load_states_config()
        self.assertGreaterEqual(len(states), 37)  # 36 states + FCT
        self.assertIn("Zamfara", states)
        self.assertIn("Kaduna", states)
        self.assertIn("FCT", states)

    def test_pilot_states(self):
        pilots = get_pilot_states()
        self.assertEqual(sorted(pilots), ["Kaduna", "Katsina", "Sokoto", "Zamfara"])

    def test_centroids_bounds(self):
        centroids = get_state_centroids()
        # Nigeria bounding box: Lat ~ 4 to 14, Lon ~ 2.5 to 15
        for st, (lat, lon) in centroids.items():
            self.assertTrue(4.0 <= lat <= 14.5, f"Lat out of bounds for {st}: {lat}")
            self.assertTrue(2.5 <= lon <= 15.0, f"Lon out of bounds for {st}: {lon}")

    def test_geographic_coverage_validation(self):
        sample_df = pd.DataFrame({"admin1": ["Zamfara", "Katsina", "Sokoto", "Kaduna"]})
        cov = validate_geographic_coverage(sample_df)
        self.assertEqual(cov["covered_states_count"], 4)
        self.assertFalse(cov["is_full_coverage"])
        self.assertGreater(len(cov["missing_states"]), 0)


class TestEthicsScreening(unittest.TestCase):

    def setUp(self):
        self.screening = ClientEthicsScreening()

    def test_humanitarian_approval(self):
        decision = self.screening.screen_client(
            client_id="NGO-001",
            organization="Doctors Without Borders / MSF",
            client_type="humanitarian",
            use_case_statement="Planning nutritional convoy access corridors and civilian displacement camp assistance.",
            confirms_non_kinetic=True,
            confirms_no_profiling=True,
        )
        self.assertTrue(decision.approved)
        self.assertEqual(len(decision.rejection_reasons), 0)

    def test_kinetic_prohibition_rejection(self):
        decision = self.screening.screen_client(
            client_id="DEF-999",
            organization="Private Military Security Contractor",
            client_type="corporate",
            use_case_statement="Identifying enemy positions to coordinate kinetic airstrike and offensive patrol targeting.",
            confirms_non_kinetic=False,
            confirms_no_profiling=True,
        )
        self.assertFalse(decision.approved)
        self.assertGreater(len(decision.prohibited_keywords_detected), 0)
        self.assertIn("kinetic", decision.prohibited_keywords_detected)

    def test_government_do_no_harm_screening(self):
        decision = self.screening.screen_client(
            client_id="GOV-012",
            organization="State Emergency Management Agency",
            client_type="government",
            use_case_statement="Pre-positioning non-food relief items and flood-conflict displacement shelters.",
            confirms_non_kinetic=True,
            confirms_no_profiling=True,
        )
        self.assertTrue(decision.approved)
        self.assertTrue(decision.do_no_harm_checklist_passed)


if __name__ == "__main__":
    unittest.main()
