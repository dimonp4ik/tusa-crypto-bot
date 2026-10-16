import unittest

import pandas as pd

import crypto_5m_cross_section_confirmation_xperp_oct16_audit as audit


class CrossSectionConfirmationXperpTests(unittest.TestCase):
    def test_remap_requires_exact_timestamp_and_contiguous_future(self):
        frame = pd.DataFrame({
            "symbol": ["A", "A", "A"],
            "entry_time": [300, 600, 999],
            "entry_index": [99, 99, 99],
        })
        candles = {"A": {
            "time": [0, 300, 600, 900, 1200],
            "open": [1] * 5, "high": [1] * 5, "low": [1] * 5,
            "close": [1] * 5, "volume": [1] * 5,
        }}
        found = audit.remap_entries(frame, candles, min_future_bars=3)
        self.assertEqual(found.entry_time.tolist(), [300, 600])
        self.assertEqual(found.entry_index.tolist(), [1, 2])

    def test_remap_rejects_gap_inside_outcome_path(self):
        frame = pd.DataFrame({"symbol": ["A"], "entry_time": [300]})
        candles = {"A": {
            "time": [0, 300, 600, 1200],
            "open": [1] * 4, "high": [1] * 4, "low": [1] * 4,
            "close": [1] * 4, "volume": [1] * 4,
        }}
        self.assertTrue(audit.remap_entries(
            frame, candles, min_future_bars=4).empty)

    def test_paired_comparison_matches_by_symbol_and_time(self):
        source = pd.DataFrame({
            "symbol": ["B", "A"], "entry_time": [2, 1],
            "signed_return": [-.01, .005], "outcome_code": [0, 2],
        })
        venue = pd.DataFrame({
            "symbol": ["A", "B"], "entry_time": [1, 2],
            "signed_return": [.005, .005], "outcome_code": [2, 2],
        })
        value = audit.paired_comparison(source, venue)
        self.assertEqual(value["n"], 2)
        self.assertEqual(value["positive_sign_agreement_pct"], 50.0)
        self.assertEqual(value["literal_target_agreement_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
