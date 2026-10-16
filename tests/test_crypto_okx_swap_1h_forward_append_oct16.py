import unittest

import pandas as pd

import crypto_okx_swap_1h_forward_append_oct16_audit as audit


class OkxSwapForwardAppendOct16Tests(unittest.TestCase):
    def test_pending_high_priority_emission_reserves_hourly_slot(self):
        rows = pd.DataFrame([
            {
                "entry_time": 1000, "exit_time": 1100, "symbol": "A",
                "direction": 1, "quality": .9, "outcome_state": "RESOLVED",
                "signed_return": .005, "target_hit": True, "exit_reason": 1,
                "selected_target_pct": .5, "source": "x", "module": "x",
            },
            {
                "entry_time": 1000, "exit_time": 10000, "symbol": "B",
                "direction": 1, "quality": .8, "outcome_state": "PENDING",
                "signed_return": float("nan"), "target_hit": False,
                "exit_reason": -1, "selected_target_pct": 1.5,
                "source": "x", "module": "x",
            },
            {
                "entry_time": 1000, "exit_time": 1100, "symbol": "C",
                "direction": 1, "quality": .7, "outcome_state": "RESOLVED",
                "signed_return": .005, "target_hit": True, "exit_reason": 1,
                "selected_target_pct": .5, "source": "x", "module": "x",
            },
        ])
        emitted = audit.emission_rows(rows)
        accepted = audit.runtime_gate_pending(emitted)
        self.assertEqual([row["symbol"] for row in accepted], ["A", "B"])
        self.assertEqual(accepted[1]["outcome"], "PENDING")

    def test_objective_requires_frequency_and_economic_return(self):
        rows = [{
            "entry_time": 1000 + index * 3600,
            "exit_time": 1100 + index * 3600,
            "symbol": f"S{index}", "direction": "LONG",
            "source": "x", "module": "x", "selected_target_pct": .5,
            "source_priority": index, "module_priority": 0,
            "target_hit": True, "exit_reason": 1, "outcome": "TP",
            "signed_price_return": .005,
        } for index in range(3)]
        value = audit.stats(rows)
        self.assertFalse(value["objective_checks"]["economic_return_gate"])
        self.assertFalse(value["objective_pass"])


if __name__ == "__main__":
    unittest.main()
