import unittest

import numpy as np

import crypto_profit_first_rolling_entry_oct16_audit as audit


class RollingProfitFirstEntryAuditTest(unittest.TestCase):
    def test_every_walkforward_training_window_precedes_test_year(self):
        for year, training in audit.TRAINING_YEARS.items():
            self.assertTrue(training)
            self.assertLess(max(training), year)

    def test_cutoff_uses_requested_upper_training_quantile(self):
        values = np.arange(101, dtype=float)
        self.assertEqual(audit.training_cutoff(values), 97.0)

    def test_objective_is_profit_frequency_and_drawdown_not_win_rate(self):
        passed = {"months": 2, "dd_pct": 10, "trades_per_calendar_day": 3,
                  "months_ge_30pct": 2, "wr": 40}
        self.assertTrue(audit.objective_ok(passed))
        self.assertFalse(audit.objective_ok({**passed, "months_ge_30pct": 1}))
        self.assertFalse(audit.objective_ok({**passed, "dd_pct": 10.01}))
        self.assertFalse(audit.objective_ok({**passed,
                                             "trades_per_calendar_day": 2.99}))


if __name__ == "__main__":
    unittest.main()
