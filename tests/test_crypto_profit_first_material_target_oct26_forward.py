import json
import unittest

import pandas as pd

import crypto_profit_first_material_target_oct26_forward as audit


class ProfitFirstMaterialOct26ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(audit.OUT.read_text(encoding="utf-8"))

    def test_previous_snapshot_is_an_exact_prefix(self):
        parity = self.report["integrity"]["previous_snapshot_prefix"]
        self.assertTrue(parity["passes"])
        self.assertEqual(parity["old_rows"], parity["refresh_prefix_rows"])
        self.assertEqual(parity["old_rows"], parity["matched_rows"])
        self.assertEqual(parity["missing_from_refresh"], 0)
        self.assertEqual(parity["unexpected_in_refresh_prefix"], 0)
        self.assertEqual(parity["maximum_signed_return_delta"], 0.0)

    def test_saved_cumulative_rows_reproduce_reported_money_metrics(self):
        rows = pd.read_pickle(audit.CUMULATIVE_ROWS_OUT)
        expected = self.report[
            "cumulative_unseen_after_original_oct16_cutoff"]
        for margin in (.03, .05):
            key = f"margin_{int(margin * 100)}pct"
            observed = audit.metrics(rows, margin)
            for field in (
                    "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
                    "negative_months", "trades_per_calendar_day",
                    "observed_calendar_days", "projected_monthly_return_pct"):
                self.assertEqual(observed[field], expected[key][field])

    def test_activation_gate_uses_cumulative_unseen_rows_and_elapsed_days(self):
        gate = self.report["activation_gate"]
        rows = pd.read_pickle(audit.CUMULATIVE_ROWS_OUT)
        self.assertEqual(gate["observed_new_completed_trades"], len(rows))
        self.assertEqual(gate["required_new_completed_trades"], 60)
        self.assertEqual(gate["required_calendar_days"], 14)
        self.assertFalse(gate["met"])
        self.assertFalse(self.report["ready_for_live_orders"])
        self.assertFalse(self.report["policy"]["retuned_on_october"])


if __name__ == "__main__":
    unittest.main()
