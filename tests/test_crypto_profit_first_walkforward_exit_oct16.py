import unittest

import pandas as pd

import crypto_profit_first_walkforward_exit_oct16_audit as audit


class ProfitFirstWalkforwardExitAuditTests(unittest.TestCase):
    def test_alignment_rejects_missing_or_duplicate_entry_keys(self):
        fixed = pd.DataFrame({
            "symbol": ["BTCUSDT"], "entry_time": [100], "direction": [1],
        })
        missing = pd.DataFrame({
            "symbol": ["ETHUSDT"], "entry_time": [100], "direction": [1],
        })
        duplicate = pd.concat([fixed, fixed], ignore_index=True)
        with self.assertRaises(ValueError):
            audit.align(fixed, missing)
        with self.assertRaises(ValueError):
            audit.align(duplicate, duplicate)

    def test_objective_does_not_require_win_rate(self):
        value = {
            "months": 3, "months_ge_30pct": 3,
            "dd_pct": 9.0, "wr": 1.0,
        }
        self.assertTrue(audit.objective_ok(value))


if __name__ == "__main__":
    unittest.main()
