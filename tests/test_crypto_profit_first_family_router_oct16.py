import unittest

import numpy as np
import pandas as pd

import crypto_profit_first_family_router_oct16_audit as audit


class ProfitFirstFamilyRouterAuditTest(unittest.TestCase):
    def test_family_mask_treats_missing_flags_as_disabled(self):
        frame = pd.DataFrame({
            "family_trend": [1, np.nan, 0],
            "family_breakout": [np.nan, 1, 0],
        })
        self.assertEqual(
            audit.family_mask(frame, ("trend",)).tolist(),
            [True, False, False])
        self.assertEqual(
            audit.family_mask(frame, ("trend", "breakout")).tolist(),
            [True, True, False])

    def test_selection_requires_each_year_to_pass(self):
        good = {"return_pct": 1, "dd_pct": 14,
                "trades_per_calendar_day": 3}
        bad = {**good, "return_pct": -1}
        self.assertTrue(audit.selection_ok({"2022": good, "2023": good}))
        self.assertFalse(audit.selection_ok({"2022": good, "2023": bad}))


if __name__ == "__main__":
    unittest.main()
