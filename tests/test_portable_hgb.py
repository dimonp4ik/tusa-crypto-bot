import math
import unittest

from src.portable_hgb import predict_probability, score_named


class PortableHgbTests(unittest.TestCase):
    def setUp(self):
        self.model = {
            "features": ["x", "y"], "baseline": 0.0,
            "trees": [[
                [0, 1.0, 1, 1, 2, 0.0, 0],
                [0, 0.0, 0, 0, 0, -1.0, 1],
                [0, 0.0, 0, 0, 0, 1.0, 1],
            ]],
        }

    def test_tree_branches_and_sigmoid(self):
        self.assertAlmostEqual(predict_probability(self.model, [0.5, 0]),
                               1 / (1 + math.exp(1)))
        self.assertAlmostEqual(predict_probability(self.model, [2.0, 0]),
                               1 / (1 + math.exp(-1)))

    def test_missing_branch_and_named_features(self):
        expected = 1 / (1 + math.exp(1))
        self.assertAlmostEqual(score_named(self.model, {"y": 4}), expected)

    def test_feature_count_is_strict(self):
        with self.assertRaises(ValueError):
            predict_probability(self.model, [1.0])


if __name__ == "__main__":
    unittest.main()
