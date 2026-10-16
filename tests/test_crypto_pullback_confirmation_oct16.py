import unittest

import pandas as pd

import crypto_pullback_confirmation_oct16_audit as audit


class PullbackConfirmationTests(unittest.TestCase):
    def test_existing_entry_wins_delayed_collision(self):
        frame = pd.DataFrame({
            "symbol": ["TEST", "TEST"],
            "entry_time": [100, 200],
            "family_pullback": [1.0, float("nan")],
            "quality": [.9, .8],
        })
        delayed = frame.iloc[[0]].copy()
        delayed["entry_time"] = 200
        delayed["confirmation_delayed"] = True
        result = audit.candidate_frame(
            frame, delayed, pd.Series([True, False]))
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0].entry_time), 200)
        self.assertFalse(bool(result.iloc[0].confirmation_delayed))


if __name__ == "__main__":
    unittest.main()
