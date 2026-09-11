import json
import os
import tempfile
import unittest

import numpy as np

from src import pullback_bank as PB
from src import pullback_paper as P
from tests.test_pullback_bank import _walk


def _candles(t, a):
    return dict(time=t, open=a[:, 0], high=a[:, 1], low=a[:, 2], close=a[:, 3], volume=a[:, 4])


class PaperTest(unittest.TestCase):
    def setUp(self):
        n = 96 * 70
        self.t, self.a = _walk(n, seed=3, drift=0.00012)
        self.bt, self.ba = _walk(n, seed=4, drift=0.00005)
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "state.json")
        self.msgs = []

    def _fetch(self, upto, fail=()):
        def f(sym):
            if sym in fail:
                raise RuntimeError("boom")
            t, a = (self.bt, self.ba) if sym == "BTCUSDT" else (self.t, self.a)
            k = np.searchsorted(t, upto, "right")
            return _candles(t[:k], a[:k])
        return f

    def test_btc_failure_sends_nothing(self):
        now = int(self.t[-1]) + PB.BAR
        n = P.run_once(["ETHUSDT"], self._fetch(now, fail=("BTCUSDT",)), self.path,
                       self.msgs.append, now=now)
        self.assertEqual(n, 0); self.assertEqual(self.msgs, [])
        self.assertFalse(os.path.exists(self.path))

    def test_symbol_failure_is_isolated(self):
        now = int(self.t[-1]) + PB.BAR
        P.run_once(["BADUSDT", "ETHUSDT"], self._fetch(now, fail=("BADUSDT",)), self.path,
                   self.msgs.append, now=now)
        with open(self.path, encoding="utf-8") as f:
            st = json.load(f)
        self.assertIn("ETHUSDT", st); self.assertNotIn("BADUSDT", st)

    def test_first_run_silent_then_hourly_runs_equal_batch(self):
        # Run hour by hour over the last 10 days; the reported OPEN/EXIT count must equal
        # the batch replay over the same bars (nothing lost, nothing reported twice).
        start = len(self.t) - 96 * 10
        now0 = int(self.t[start]) + PB.BAR
        P.run_once(["ETHUSDT"], self._fetch(now0), self.path, self.msgs.append, now=now0)
        warm = list(self.msgs)
        total = 0
        for k in range(start + 4, len(self.t), 4):
            now = int(self.t[k]) + PB.BAR
            total += P.run_once(["ETHUSDT"], self._fetch(now), self.path, self.msgs.append, now=now)
        sig = PB.hourly_signals(self.t, self.a, self.bt, self.ba)
        st = {}
        PB.advance(st, self.t[:start], self.a[:start], sig)
        ev = PB.advance(st, self.t[:k + 1], self.a[:k + 1], sig)
        batch = sum(e[0] in ("OPEN", "EXIT") for e in ev)
        self.assertEqual(total, batch)
        self.assertTrue(all("🟢" not in m and "✅" not in m for m in warm))


if __name__ == "__main__":
    unittest.main()
