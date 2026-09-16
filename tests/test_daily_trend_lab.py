import unittest

import numpy as np
import pandas as pd

from daily_trend_lab import donchian_score, portfolio_returns


class DailyTrendLabTests(unittest.TestCase):
    def test_donchian_breakout_excludes_current_bar_high(self):
        index = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
        frame = pd.DataFrame({
            "open": [9, 9, 9, 11],
            "high": [10, 10, 100, 12],
            "low": [8, 8, 8, 10],
            "close": [9, 9, 11, 11],
            "volume": [1, 1, 1, 1],
        }, index=index)
        signal = donchian_score(frame, lookbacks=(2,))
        self.assertEqual(signal.iloc[2], 1.0)

    def test_signal_executes_at_next_open(self):
        index = pd.date_range("2026-01-01", periods=25, freq="D", tz="UTC")
        close = 100 + np.arange(25, dtype=float) * 0.1
        open_ = np.full(25, 100.0)
        open_[22] = 110.0
        frame = pd.DataFrame({
            "open": open_, "high": close + 1, "low": close - 1,
            "close": close, "volume": np.ones(25),
        }, index=index)
        signal = pd.Series(0.0, index=index)
        signal.iloc[20] = 1.0
        result = portfolio_returns(
            {"A": frame}, {"A": signal}, one_way_cost_bps=0, crash_overlay=False,
        )
        self.assertAlmostEqual(result.loc[index[21], "net_return"], 0.1)
        self.assertEqual(result.loc[index[20], "net_return"], 0.0)


if __name__ == "__main__":
    unittest.main()
