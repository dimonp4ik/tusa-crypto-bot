import unittest

import crypto_profit_first_runner_oct16_audit as audit


def metrics(**updates):
    value = {
        "n": 800,
        "wr": 10.0,
        "return_pct": 20.0,
        "dd_pct": 8.0,
        "trades_per_calendar_day": 2.2,
        "months": 12,
        "monthly_return_pct": {"2022-01": 1.0},
    }
    value.update(updates)
    return value


class ProfitFirstRunnerAuditTests(unittest.TestCase):
    def test_monthly_equivalent_compounds_total_return(self):
        value = {
            "return_pct": 21.0,
            "monthly_return_pct": {"2024-01": 10.0, "2024-02": 10.0},
        }
        self.assertAlmostEqual(audit.monthly_equivalent(value), 10.0)

    def test_development_gate_does_not_require_high_win_rate(self):
        value = metrics(wr=5.0)
        self.assertTrue(audit.development_ok(value, {"2022": value}))

    def test_profit_objective_requires_every_month_at_target(self):
        passed = audit.enrich(metrics(
            dd_pct=9.0,
            monthly_return_pct={"2024-01": 31.0, "2024-02": 30.0}))
        failed_month = audit.enrich(metrics(
            dd_pct=9.0,
            monthly_return_pct={"2024-01": 31.0, "2024-02": 29.9}))
        failed_drawdown = audit.enrich(metrics(
            dd_pct=10.1,
            monthly_return_pct={"2024-01": 31.0, "2024-02": 30.0}))
        self.assertTrue(passed["profit_first_objective_pass"])
        self.assertFalse(failed_month["profit_first_objective_pass"])
        self.assertFalse(failed_drawdown["profit_first_objective_pass"])


if __name__ == "__main__":
    unittest.main()
