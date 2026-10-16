import unittest

import numpy as np

import crypto_profit_first_rolling_literal_runner_oct17_audit as audit


class RollingLiteralRunnerAuditTest(unittest.TestCase):
    def test_training_years_are_strictly_earlier(self):
        for year, training in audit.TRAINING_YEARS.items():
            self.assertLess(max(training), year)

    def test_training_cutoff_is_fixed_upper_decile(self):
        self.assertEqual(audit.training_cutoff(np.arange(101)), 90.0)

    def test_objective_does_not_gate_on_win_rate(self):
        value = {"months": 2, "dd_pct": 9,
                 "trades_per_calendar_day": 3,
                 "months_ge_30pct": 2, "wr": 35}
        self.assertTrue(audit.objective_ok(value))


if __name__ == "__main__":
    unittest.main()
