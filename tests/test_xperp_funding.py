import unittest

from reprice_xperp_funding import funding_impact


class FundingRepricingTests(unittest.TestCase):
    def test_positive_rate_is_paid_by_long_and_received_by_short(self):
        self.assertAlmostEqual(funding_impact("LONG", 0.0001, 0.01), -0.01)
        self.assertAlmostEqual(funding_impact("SHORT", 0.0001, 0.01), 0.01)

    def test_negative_rate_reverses_cash_flow(self):
        self.assertAlmostEqual(funding_impact("LONG", -0.0001, 0.01, 2), 0.02)

    def test_invalid_inputs_fail_closed(self):
        for risk in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                funding_impact("LONG", 0.0001, risk)
