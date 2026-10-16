import unittest

import pandas as pd

import crypto_pullback_direction_switch_oct16_audit as audit


class PullbackDirectionSwitchTests(unittest.TestCase):
    def test_switch_changes_only_flagged_pullback_path(self):
        original = pd.DataFrame({
            "pullback_entry": [True, False, True],
            "direction": [1, 1, -1],
            "signed_return": [-.01, .005, -.01],
            "exit_time": [10, 20, 30],
            "target_hit": [False, True, False],
            "exit_reason": [0, 1, 0],
        })
        reverse = original.copy()
        reverse["direction"] = [-1, -1, 1]
        reverse["signed_return"] = [.005, -.01, .005]
        reverse["exit_time"] = [11, 21, 31]
        reverse["target_hit"] = [True, False, True]
        reverse["exit_reason"] = [1, 0, 1]
        result = audit.switched(
            original, reverse, pd.Series([True, True, False]))
        self.assertEqual(result.direction.tolist(), [-1, 1, -1])
        self.assertEqual(result.signed_return.tolist(), [.005, .005, -.01])
        self.assertEqual(result.direction_switched.tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
