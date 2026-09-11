import os
import tempfile
import unittest

import numpy as np

from src import trend4h_paper as P
from tests.test_trend4h import _synthetic, _calm_btc


class PaperRunnerTest(unittest.TestCase):
    def setUp(self):
        self.t, o, h, l, c = _synthetic(n_bars=700)
        self.candles = dict(time=list(self.t), open=list(o), high=list(h), low=list(l), close=list(c))
        self.days, self.closes = _calm_btc(self.t)
        self.path = os.path.join(tempfile.mkdtemp(), "state.json")

    def _fetch_upto(self, k):
        return lambda sym: {key: v[:k] for key, v in self.candles.items()}

    def test_btc_failure_sends_nothing(self):
        sent = []
        def boom():
            raise RuntimeError("network down")
        n = P.run_once(["X"], self._fetch_upto(len(self.t)), boom, self.path, sent.append)
        self.assertEqual(n, 0); self.assertEqual(sent, [])

    def test_symbol_failure_is_isolated(self):
        sent = []
        def fetch(sym):
            if sym == "BAD":
                raise RuntimeError("bad symbol")
            return self._fetch_upto(len(self.t))(sym)
        P.run_once(["BAD", "GOOD"], fetch, lambda: (self.days, self.closes), self.path, sent.append,
                   now=self.t[-1] + 900)
        state = P._load_state(self.path)
        self.assertIn("GOOD", state); self.assertNotIn("BAD", state)

    def test_first_run_is_silent_then_reports_only_new_bars(self):
        sent = []
        half = len(self.t) // 2
        P.run_once(["X"], self._fetch_upto(half), lambda: (self.days, self.closes), self.path,
                   sent.append, now=self.t[half - 1] + 900)
        first = len(sent)
        total = 0
        for k in range(half + 16, len(self.t) + 1, 16):
            total += P.run_once(["X"], self._fetch_upto(k), lambda: (self.days, self.closes),
                                self.path, sent.append, now=self.t[k - 1] + 900)
        self.assertLessEqual(first, 1)
        self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main()
