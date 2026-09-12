"""The filter must read closed bars only, and must never block trading by failing."""
import unittest

import numpy as np

from src import bank_filter as BF

HOUR = 3600
BAR = 900


def series(n=2600, seed=0):
    """Synthetic 15m candles: t aligned to the bar grid, a = open/high/low/close/volume."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.int64) * BAR
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    v = rng.uniform(10, 1000, n)
    return t, np.c_[o, hi, lo, c, v]


class BarsTest(unittest.TestCase):
    def test_incomplete_bar_is_dropped(self):
        t, a = series(400)
        full_t, full_a = BF.build_bars(t, a, HOUR)
        self.assertEqual(len(full_t), 100)
        cut_t, cut_a = BF.build_bars(t[:-1], a[:-1], HOUR)      # last hour has 3 of 4 candles
        self.assertEqual(len(cut_t), 99)
        np.testing.assert_array_equal(cut_t, full_t[:-1])
        np.testing.assert_allclose(cut_a, full_a[:-1])

    def test_bar_is_the_real_aggregate(self):
        t, a = series(8)
        bt, ba = BF.build_bars(t, a, HOUR)
        self.assertEqual(list(bt), [0, HOUR])
        self.assertAlmostEqual(ba[0, 0], a[0, 0])               # open of the first candle
        self.assertAlmostEqual(ba[0, 1], a[:4, 1].max())        # high of the hour
        self.assertAlmostEqual(ba[0, 2], a[:4, 2].min())        # low of the hour
        self.assertAlmostEqual(ba[0, 3], a[3, 3])               # close of the last candle
        self.assertAlmostEqual(ba[0, 4], a[:4, 4].sum())

    def test_a_gap_in_the_feed_drops_that_bar_only(self):
        t, a = series(12)
        keep = np.ones(12, bool); keep[5] = False               # one missing candle in hour 2
        bt, _ = BF.build_bars(t[keep], a[keep], HOUR)
        self.assertEqual(list(bt), [0, 2 * HOUR])


class LookAheadTest(unittest.TestCase):
    """Candles of the hour in progress must not move a single feature of the closed hour."""

    def test_features_ignore_the_unfinished_hour(self):
        t, a = series()
        base, names = BF.features(t, a, t, a)
        self.assertIsNotNone(base)
        self.assertEqual(len(names), len(base))
        for extra in (1, 2, 3):
            t2 = np.r_[t, t[-1] + BAR * np.arange(1, extra + 1)]
            a2 = np.vstack([a, np.tile(a[-1] * 1.5, (extra, 1))])   # loud future candles
            row, nm2 = BF.features(t2, a2, t2, a2)
            self.assertEqual(names, nm2)
            bad = [names[i] for i in range(len(names))
                   if not (np.isnan(base[i]) and np.isnan(row[i])) and base[i] != row[i]]
            self.assertEqual(bad, [], f"{len(bad)} features moved with {extra} future candles")

    def test_short_history_scores_nothing(self):
        t, a = series(800)                                       # under 300 closed hours
        row, _ = BF.features(t, a, t, a)
        self.assertIsNone(row)


class FailOpenTest(unittest.TestCase):
    def setUp(self):
        self._cache = dict(BF._cache)

    def tearDown(self):
        BF._cache.clear(); BF._cache.update(self._cache)

    def test_missing_model_takes_the_signal(self):
        BF._cache.update(model=None, spec=None, tried=True)
        t, a = series()
        self.assertEqual(BF.allow(t, a, t, a), (True, None))

    def test_broken_model_takes_the_signal(self):
        class Boom:
            def predict(self, X):
                raise RuntimeError("model file corrupt")
        BF._cache.update(model=Boom(), spec={"features": [], "threshold": 0.0}, tried=True)
        t, a = series()
        self.assertEqual(BF.allow(t, a, t, a), (True, None))

    def test_too_little_history_takes_the_signal(self):
        BF._cache.update(model=object(), spec={"features": [], "threshold": 0.0}, tried=True)
        t, a = series(800)
        self.assertEqual(BF.allow(t, a, t, a), (True, None))

    def test_threshold_decides(self):
        t, a = series()
        _, names = BF.features(t, a, t, a)

        class Fixed:
            def __init__(self, v):
                self.v = v

            def predict(self, X):
                return np.array([self.v])
        for v, want in ((0.20, True), (0.05, False)):
            BF._cache.update(model=Fixed(v), spec={"features": names, "threshold": 0.1}, tried=True)
            take, sc = BF.allow(t, a, t, a)
            self.assertEqual(take, want)
            self.assertAlmostEqual(sc, v)

    def test_features_are_reordered_to_the_training_order(self):
        t, a = series()
        row, names = BF.features(t, a, t, a)
        seen = {}

        class Spy:
            def predict(self, X):
                seen["row"] = X[0]
                return np.array([1.0])
        want = list(reversed(names))
        BF._cache.update(model=Spy(), spec={"features": want, "threshold": 0.0}, tried=True)
        BF.allow(t, a, t, a)
        np.testing.assert_array_equal(seen["row"], row[::-1])


class RealModelTest(unittest.TestCase):
    def test_the_shipped_model_scores_a_signal(self):
        BF._cache.update(model=None, spec=None, tried=False)
        model, spec = BF._load()
        if model is None:
            self.skipTest("models/bank_filter.joblib is not installed")
        self.assertEqual(len(spec["features"]), 140)
        t, a = series()
        take, sc = BF.allow(t, a, t, a)
        self.assertIsNotNone(sc)
        self.assertEqual(take, sc >= spec["threshold"])


if __name__ == "__main__":
    unittest.main()
