import unittest

import numpy as np

from src import trend4h as M


def _synthetic(seed=7, n_bars=900, drift=0.0):
    """Random-walk 15m candles (seconds), aligned to 4h boundaries."""
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000 - 1_700_000_000 % M.BAR_SEC
    n = n_bars * 16
    ret = rng.normal(drift, 0.004, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[100.0, close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
    times = t0 + 900 * np.arange(n)
    return times, open_, high, low, close


def _calm_btc(times):
    days = np.arange(times[0] - 90 * 86400, times[-1] + 86400, 86400)
    return list(days), np.full(len(days), 50_000.0)


class Trend4hTest(unittest.TestCase):
    def setUp(self):
        t, o, h, l, c = _synthetic()
        self.T, self.B = M.build_4h(t, o, h, l, c)
        self.BT, self.BC = _calm_btc(t)

    def test_complete_bars_only(self):
        t, o, h, l, c = _synthetic(n_bars=10)
        T, B = M.build_4h(t[:-3], o[:-3], h[:-3], l[:-3], c[:-3])
        self.assertEqual(len(T), 9)   # the truncated last bar is dropped

    def test_incremental_equals_batch(self):
        batch = M.advance({}, self.T, self.B, self.BT, self.BC)
        st, inc = {}, []
        for k in range(1, len(self.T) + 1):
            inc += M.advance(st, self.T[:k], self.B[:k], self.BT, self.BC)
        self.assertEqual(batch, inc)
        self.assertTrue(any(e[0] == "OPEN" for e in batch))

    def test_future_bars_do_not_change_the_past(self):
        k = len(self.T) // 2
        head = M.advance({}, self.T[:k], self.B[:k], self.BT, self.BC)
        full = M.advance({}, self.T, self.B, self.BT, self.BC)
        self.assertEqual(head, full[:len(head)])

    def test_one_position_per_symbol(self):
        open_now = False
        for e in M.advance({}, self.T, self.B, self.BT, self.BC):
            if e[0] == "OPEN":
                self.assertFalse(open_now)
                open_now = True
            elif e[0] == "EXIT":
                self.assertTrue(open_now)
                open_now = False

    def test_trailing_stop_never_loosens(self):
        st = {}
        stops = {}
        for e in M.advance(st, self.T, self.B, self.BT, self.BC):
            if e[0] == "OPEN":
                side, last = e[2], e[4]
            elif e[0] == "STOP":
                if side == "LONG":
                    self.assertGreaterEqual(e[2], last)
                else:
                    self.assertLessEqual(e[2], last)
                last = e[2]

    def test_btc_gate_blocks_entries_when_btc_runs(self):
        days, _ = _calm_btc(self.T)
        trending = 50_000 * np.exp(np.linspace(0, 3, len(days)))   # ~+12% per 20d
        ev = M.advance({}, self.T, self.B, days, trending)
        self.assertFalse(any(e[0] == "OPEN" for e in ev))
        ev_open = M.advance({}, self.T, self.B, days, trending, gate=False)
        self.assertTrue(any(e[0] == "OPEN" for e in ev_open))

    def test_gate_reads_only_closed_daily_bars(self):
        days, closes = _calm_btc(self.T)
        ts = int(self.T[-1])
        i = int(np.searchsorted(days, ts - 86400, "right")) - 1
        spiked = closes.copy()
        spiked[i + 1:] *= 2.0          # a move that has not closed yet by ts
        self.assertEqual(M.btc_momentum(days, closes, ts), M.btc_momentum(days, spiked, ts))


if __name__ == "__main__":
    unittest.main()
