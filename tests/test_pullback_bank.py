import time
import unittest

import numpy as np

from src import pullback_bank as PB


def _walk(n, seed=0, start=1_700_000_000, drift=0.0):
    rng = np.random.default_rng(seed)
    t = start - start % 86400 + np.arange(n, dtype=np.int64) * PB.BAR
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.003, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.003, n))
    return t, np.c_[o, h, l, c, np.ones(n)]


class BarsTest(unittest.TestCase):
    def test_only_complete_contiguous_hours(self):
        t, a = _walk(40)
        t = np.delete(t, 9); a = np.delete(a, 9, axis=0)          # hole inside hour 2
        T1, B1 = PB.build_bars(t, a, PB.HOUR)
        self.assertEqual(len(T1), 9)                               # 10 hours, one broken
        self.assertTrue(np.all(T1 % PB.HOUR == 0))


class ExecutionTest(unittest.TestCase):
    """advance() with hand-made signals and bars."""

    def _bars(self, rows, start=1_700_006_400):
        t = start + np.arange(len(rows), dtype=np.int64) * PB.BAR
        return t, np.array(rows, dtype=float)

    def test_passive_fill_gets_no_take_profit_in_fill_bar(self):
        # order at close of bar 3 (hour close), limit 100; bar 4 opens above, dips to 99.9
        # and its high 102 would hit a 0.5-ATR take of 100.5 - must NOT count in that bar.
        rows = [[100, 100.1, 99.9, 100, 1]] * 4 + [[100.2, 102, 99.9, 100.2, 1],
                                                   [100.2, 100.3, 100.1, 100.2, 1],
                                                   [100.2, 101.0, 100.1, 100.9, 1]]
        t, a = self._bars(rows)
        sig = {int(t[3]) + PB.BAR: (1, 100.0, 1.0)}                 # rule 1: tp 0.5, sl 3
        ev = PB.advance({}, t, a, sig)
        kinds = [e[0] for e in ev]
        self.assertEqual(kinds, ["ORDER", "OPEN", "EXIT"])
        opened = ev[1]; exited = ev[2]
        self.assertEqual(opened[6], "maker")
        self.assertEqual(exited[1], int(t[6]) + PB.BAR)             # took profit in bar 6, not 4
        self.assertAlmostEqual(exited[4], 0.5 / 100 - PB.PASSIVE_COST)

    def test_missed_limit_places_nothing(self):
        rows = [[100, 100.1, 99.9, 100, 1]] * 4 + [[100.5, 101, 100.2, 100.8, 1]] * 3
        t, a = self._bars(rows)
        sig = {int(t[3]) + PB.BAR: (1, 100.0, 1.0)}
        ev = PB.advance({}, t, a, sig)
        self.assertEqual([e[0] for e in ev], ["ORDER", "MISS"])

    def test_stop_gap_fills_at_open(self):
        rows = [[100, 100.1, 99.9, 100, 1]] * 4 + [[99.8, 99.9, 99.7, 99.8, 1],
                                                   [95.0, 95.5, 94.0, 95.0, 1]]
        t, a = self._bars(rows)
        sig = {int(t[3]) + PB.BAR: (1, 100.0, 1.0)}
        ev = PB.advance({}, t, a, sig)
        ex = ev[-1]
        self.assertEqual(ex[0], "EXIT"); self.assertEqual(ex[3], "stop")
        e = 99.8 * (1 + PB.TAKER_SLIP)
        self.assertAlmostEqual(ex[4], 95.0 / e - 1 - PB.TAKER_COST)

    def test_one_position_per_symbol(self):
        rows = [[100, 100.1, 99.9, 100, 1]] * 4 + [[99.9, 100.0, 99.8, 99.9, 1]] * 12
        t, a = self._bars(rows)
        sig = {int(x) + PB.BAR: (1, 100.0, 1.0) for x in t if (int(x) + PB.BAR) % PB.HOUR == 0}
        ev = PB.advance({}, t, a, sig)
        self.assertEqual(sum(e[0] == "OPEN" for e in ev), 1)


class LivePathTest(unittest.TestCase):
    def setUp(self):
        n = 96 * 70
        self.t, self.a = _walk(n, seed=3, drift=0.00012)
        self.bt, self.ba = _walk(n, seed=4, drift=0.00005)
        self.sig = PB.hourly_signals(self.t, self.a, self.bt, self.ba)
        self.assertGreater(len(self.sig), 5)          # the tests below must see real signals

    def test_incremental_equals_batch(self):
        batch = PB.advance({}, self.t, self.a, self.sig)
        st, inc = {}, []
        for k in range(0, len(self.t), 37):
            inc += PB.advance(st, self.t[:k + 37], self.a[:k + 37], self.sig)
        self.assertEqual(batch, inc)

    def test_future_bars_do_not_change_past_signals(self):
        cut = len(self.t) - 96 * 5
        early = PB.hourly_signals(self.t[:cut], self.a[:cut], self.bt[:cut], self.ba[:cut])
        last_close = int(self.t[cut - 1]) + PB.BAR
        full = {k: v for k, v in self.sig.items() if k <= last_close}
        self.assertEqual(early, full)

    def test_regime_incremental_equals_batch_and_survives_window(self):
        bt, bc = PB.btc_daily(self.bt, self.ba)
        batch = PB.trend_regime(self.t, self.a, bt, bc)
        st = {}
        for k in range(96 * 30, len(self.t) + 1, 96):
            lo = max(0, k - 96 * 40)                       # the live bot sees a 40-day window
            bm = (self.bt >= self.t[lo]) & (self.bt <= self.t[k - 1])
            d, c = PB.btc_daily(self.bt[bm], self.ba[bm])
            iv = PB.regime_update(st, self.t[lo:k], self.a[lo:k], d, c)
        warm = int(self.t[96 * 30])
        self.assertEqual([x for x in iv if x[1] > warm + 86400 * 25],
                         [x for x in batch if x[1] > warm + 86400 * 25])

    def test_regime_known_at_each_hour_equals_batch(self):
        # The live bot calls regime_update() every hour with only the bars closed so far.
        # The side it sees at that hour must equal the backtest's side at that hour -
        # including the hours right after a trend entry was signalled (pending fill).
        bt, bc = PB.btc_daily(self.bt, self.ba)
        batch = PB.trend_regime(self.t, self.a, bt, bc)
        side_at = lambda iv, c: [s for x0, x1, s in iv if x0 <= c < x1]
        st, checked, flips = {}, 0, 0
        for k in range(96 * 20, len(self.t)):
            c = int(self.t[k]) + PB.BAR
            if c % PB.HOUR:
                continue
            bm = self.bt <= self.t[k]
            d, cl = PB.btc_daily(self.bt[bm], self.ba[bm])
            live = side_at(PB.regime_update(st, self.t[:k + 1], self.a[:k + 1], d, cl), c)
            self.assertEqual(live, side_at(batch, c), time.strftime("%d.%m %H:%M", time.gmtime(c)))
            checked += 1; flips += bool(live)
        self.assertGreater(flips, 50)

    def test_signals_only_inside_matching_regime(self):
        rules = PB.RULE_SETS["bank9+short2"]
        seen = set()
        for seed, drift in ((3, 0.00012), (5, -0.00012), (7, -0.0002)):
            t, a = _walk(96 * 70, seed=seed, drift=drift)
            bt, ba = _walk(96 * 70, seed=seed + 1, drift=0.00005)
            d, c = PB.btc_daily(bt, ba)
            iv = PB.trend_regime(t, a, d, c)
            for close, (k, _, _) in PB.hourly_signals(t, a, bt, ba, rules).items():
                side = [s for x0, x1, s in iv if x0 <= close < x1]
                self.assertEqual(side, [rules[k].get("side", "LONG")])
                seen.add(side[0])
        self.assertEqual(seen, {"LONG", "SHORT"})              # both sides really exercised


class ShortMirrorTest(unittest.TestCase):
    def _run(self, rows):
        t = 1_700_006_400 + np.arange(len(rows), dtype=np.int64) * PB.BAR
        a = np.array(rows, dtype=float)
        return t, PB.advance({}, t, a, {int(t[3]) + PB.BAR: (0, 100.0, 1.0)}, PB.SHORT_RULES)

    def test_short_passive_fill_ignores_take_in_fill_bar(self):
        t, ev = self._run([[100, 100.1, 99.9, 100, 1]] * 4 + [[99.8, 100.1, 98.0, 99.8, 1],
                                                              [99.8, 99.9, 99.7, 99.8, 1],
                                                              [99.8, 99.9, 99.0, 99.2, 1]])
        self.assertEqual([e[0] for e in ev], ["ORDER", "OPEN", "EXIT"])
        self.assertEqual(ev[1][6:], ("maker", "SHORT"))
        self.assertEqual(ev[2][1], int(t[6]) + PB.BAR)         # the 98.0 low in the fill bar is ignored
        self.assertAlmostEqual(ev[2][4], 0.5 / 100 - PB.PASSIVE_COST)

    def test_short_stop_gap_fills_at_open(self):
        t, ev = self._run([[100, 100.1, 99.9, 100, 1]] * 4 + [[100.2, 100.3, 100.1, 100.2, 1],
                                                              [104.0, 104.5, 103.8, 104.0, 1]])
        ex = ev[-1]; e = 100.2 * (1 - PB.TAKER_SLIP)
        self.assertEqual(ex[3], "stop")
        self.assertAlmostEqual(ex[4], 1 - 104.0 / e - PB.TAKER_COST)


if __name__ == "__main__":
    unittest.main()
