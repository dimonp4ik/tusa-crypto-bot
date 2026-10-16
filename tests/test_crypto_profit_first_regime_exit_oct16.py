import unittest

import crypto_profit_first_regime_exit_oct16_audit as audit


class ProfitFirstRegimeExitAuditTests(unittest.TestCase):
    def test_profile_score_does_not_depend_on_win_rate(self):
        base = {
            "equivalent_monthly_return_pct": 3.0,
            "dd_pct": 5.0,
            "negative_months": 1,
        }
        self.assertEqual(
            audit.profile_score({**base, "wr": 5.0}),
            audit.profile_score({**base, "wr": 95.0}))

    def test_objective_requires_every_month_above_thirty(self):
        passed = {"months": 2, "months_ge_30pct": 2, "dd_pct": 10.0}
        failed_month = {"months": 2, "months_ge_30pct": 1, "dd_pct": 9.0}
        failed_drawdown = {"months": 2, "months_ge_30pct": 2, "dd_pct": 10.1}
        self.assertTrue(audit.objective_ok(passed))
        self.assertFalse(audit.objective_ok(failed_month))
        self.assertFalse(audit.objective_ok(failed_drawdown))


if __name__ == "__main__":
    unittest.main()
