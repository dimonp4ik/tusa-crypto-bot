import unittest

import numpy as np
import pandas as pd

import crypto_profit_first_literal_runner_oct16_audit as audit


class LiteralRunnerAuditTest(unittest.TestCase):
    def test_pessimistic_same_bar_collision_is_stop(self):
        values, _, hits, reasons = audit.resolve(
            np.array([100.0]), np.array([103.0]), np.array([98.0]),
            np.array([101.0]), np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64), np.array([1], dtype=np.int8),
            2.0, 24)
        self.assertAlmostEqual(values[0], -.01)
        self.assertEqual(hits[0], 0)
        self.assertEqual(reasons[0], 0)

    def test_long_and_short_literal_targets(self):
        open_ = np.array([100.0, 100.0])
        high = np.array([102.1, 100.5])
        low = np.array([99.5, 97.9])
        close = np.array([102.0, 98.0])
        times = np.array([0, 3600], dtype=np.int64)
        entries = np.array([0, 1], dtype=np.int64)
        directions = np.array([1, -1], dtype=np.int8)
        values, _, hits, reasons = audit.resolve(
            open_, high, low, close, times, entries, directions, 2.0, 1)
        np.testing.assert_allclose(values, [.02, .02])
        np.testing.assert_array_equal(hits, [1, 1])
        np.testing.assert_array_equal(reasons, [1, 1])

    def test_objective_does_not_require_a_win_rate(self):
        value = {"months": 3, "dd_pct": 8,
                 "trades_per_calendar_day": 3.1,
                 "months_ge_30pct": 3, "wr": 42}
        self.assertTrue(audit.objective_ok(value))


if __name__ == "__main__":
    unittest.main()
