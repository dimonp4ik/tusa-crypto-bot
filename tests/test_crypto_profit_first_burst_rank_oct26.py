import json
import unittest

import crypto_profit_first_burst_rank_oct26_audit as audit


class ProfitFirstBurstRankOct26Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(audit.OUT.read_text(encoding="utf-8"))

    def test_grid_was_selected_before_forward_and_rejected(self):
        self.assertTrue(self.report["baseline_replay_parity"])
        self.assertFalse(self.report["robust_improvement"])
        self.assertEqual(self.report["status"],
                         "REJECTED_NO_ROBUST_IMPROVEMENT")
        self.assertFalse(self.report["protocol"]["retune_on_forward"])
        self.assertEqual(len(self.report["development_frontier"]), 12)
        self.assertEqual(self.report["selected_before_2024"]["policy"],
                         self.report["baseline"]["policy"])

    def test_same_timestamp_direction_cap_blocks_the_second_entry(self):
        base = {
            "entry_time": 100_000.0, "exit_time": 100_900.0,
            "source_priority": 3, "module_priority": 999,
            "direction": "SHORT", "family": "breakout",
            "model_probability": .4, "risk_pct": .012,
            "abs_z": 2.0, "eff_ratio": .5, "outcome": "TP",
        }
        rows = [
            {**base, "symbol": "BTCUSDT"},
            {**base, "symbol": "ETHUSDT", "model_probability": .5},
        ]
        capped = audit.gate(rows, "probability_desc", 1)
        uncapped = audit.gate(rows, "probability_desc", 2)
        self.assertEqual([row["symbol"] for row in capped], ["ETHUSDT"])
        self.assertEqual(len(uncapped), 2)


if __name__ == "__main__":
    unittest.main()
