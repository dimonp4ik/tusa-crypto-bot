import tempfile
import unittest
from pathlib import Path

import crypto_5m_cross_section_confirmation_holdout_oct16_audit as audit


class CrossSectionConfirmationHoldoutTests(unittest.TestCase):
    def test_merge_is_sorted_and_tail_overwrites_overlap(self):
        base = {"time": [0, 300], "x": [1, 2]}
        tail = {"time": [300, 600], "x": [20, 3]}
        self.assertEqual(audit.merge_time_series(base, tail), {
            "time": [0, 300, 600], "x": [1, 20, 3],
        })

    def test_merge_refuses_schema_change(self):
        with self.assertRaises(ValueError):
            audit.merge_time_series(
                {"time": [0], "x": [1]}, {"time": [300], "y": [2]})


if __name__ == "__main__":
    unittest.main()
