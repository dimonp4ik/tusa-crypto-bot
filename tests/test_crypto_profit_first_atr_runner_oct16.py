import unittest

import numpy as np
import pandas as pd

import crypto_profit_first_atr_runner_oct16_audit as audit


class ProfitFirstAtrRunnerAuditTests(unittest.TestCase):
    def test_runner_target_is_clamped_to_literal_range(self):
        frame = pd.DataFrame({"atr14_pct": [.001, .01, .05]})
        self.assertEqual(
            audit.runner_targets(frame, 3.0).tolist(),
            [2.0, 3.0, 8.0])

    def test_ambiguous_first_bar_counts_stop_before_target(self):
        signed, _, first_hits, runner_hits, reasons = audit.resolve(
            np.array([100.0]), np.array([104.0]), np.array([98.0]),
            np.array([101.0]), np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64), np.array([1], dtype=np.int8),
            np.array([3.0]), .25, 1)
        self.assertAlmostEqual(signed[0], -.01)
        self.assertEqual(first_hits[0], 0)
        self.assertEqual(runner_hits[0], 0)
        self.assertEqual(reasons[0], 0)

    def test_selection_gate_ignores_win_rate(self):
        value = {
            "n": 800, "wr": 1.0, "return_pct": 5.0,
            "dd_pct": 10.0, "trades_per_calendar_day": 2.1,
        }
        self.assertTrue(audit.selection_ok(value))


if __name__ == "__main__":
    unittest.main()
