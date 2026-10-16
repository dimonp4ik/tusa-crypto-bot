import unittest

import crypto_5m_cross_section_confirmation_geometry_oct16_audit as audit


class CrossSectionConfirmationGeometryTests(unittest.TestCase):
    def test_selection_score_uses_worst_development_year(self):
        exact = {
            "2022": {"economic_gate": {"projected_monthly_return_pct": 2},
                     "return_pct": 4, "mean_signed_price_return_pct": .2, "dd_pct": 1},
            "2023": {"economic_gate": {"projected_monthly_return_pct": 1},
                     "return_pct": 3, "mean_signed_price_return_pct": .1, "dd_pct": 2},
            "2024": {"economic_gate": {"projected_monthly_return_pct": 3},
                     "return_pct": 5, "mean_signed_price_return_pct": .3, "dd_pct": 3},
        }
        self.assertEqual(audit.score(exact), (1, 3, .1, -3))


if __name__ == "__main__":
    unittest.main()
