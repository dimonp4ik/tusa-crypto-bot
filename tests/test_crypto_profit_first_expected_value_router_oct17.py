import unittest

import numpy as np
import pandas as pd

import crypto_profit_first_expected_value_router_oct17_audit as audit


class ExpectedValueRouterAuditTest(unittest.TestCase):
    def test_expected_payoff_uses_target_to_stop_ratio(self):
        probability = np.array([.5])
        self.assertAlmostEqual(audit.expected_payoff(probability, 1.0)[0], 0.0)
        self.assertAlmostEqual(audit.expected_payoff(probability, 2.0)[0], .005)

    def test_router_chooses_best_per_row_and_copies_its_outcome(self):
        base = pd.DataFrame({
            "symbol": ["BTC", "ETH"], "entry_time": [1, 2],
            "direction": [1, -1], "signed_return": [.005, .005],
            "exit_time": [3, 4], "target_hit": [True, True],
            "exit_reason": [1, 1], "literal_target_pct": [.5, .5],
            "hold_bars": [24, 24],
        })
        larger = base.copy()
        larger["signed_return"] = [.02, -.01]
        larger["literal_target_pct"] = 2.0
        probability = np.array([[.70, .60], [.90, .20]])
        routed = audit.route_profiles([base, larger], probability, (.5, 2.0))
        self.assertEqual(routed.routed_target_pct.tolist(), [2.0, .5])
        self.assertEqual(routed.signed_return.tolist(), [.02, .005])

    def test_objective_does_not_gate_on_win_rate(self):
        value = {"months": 2, "dd_pct": 9,
                 "trades_per_calendar_day": 3,
                 "months_ge_30pct": 2, "wr": 35}
        self.assertTrue(audit.objective_ok(value))


if __name__ == "__main__":
    unittest.main()
