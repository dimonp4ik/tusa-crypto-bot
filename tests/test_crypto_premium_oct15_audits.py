import unittest

import numpy as np
import pandas as pd

import crypto_premium_dislocation_oct15_audit as premium
import crypto_premium_exit_geometry_oct15_audit as geometry
import crypto_futures_funding_regime_oct6_audit as funding


class PremiumAuditCausalityTests(unittest.TestCase):
    def test_attach_waits_for_premium_bar_close(self):
        count = 700
        times = np.arange(count, dtype=np.int64)*900
        close = 0.001 + np.sin(np.arange(count)/13)*0.0002
        premium_data = {}
        for offset, symbol in enumerate(funding.SYMBOLS):
            symbol_close = close*(1+offset/20)
            premium_data[symbol] = {
                "time": times,
                "open": symbol_close-0.00001,
                "high": symbol_close+0.00003,
                "low": symbol_close-0.00003,
                "close": symbol_close,
            }
        bar = 650
        entries = pd.DataFrame([
            {"symbol": symbol, "entry_time": entry_time, "marker": marker}
            for entry_time, marker in (
                (int(times[bar]+899), "before_close"),
                (int(times[bar]+900), "at_close"))
            for symbol in funding.SYMBOLS
        ])

        joined = premium.attach(entries, premium_data)

        before = joined.loc[
            joined.marker.eq("before_close") & joined.symbol.eq("BTCUSDT")].iloc[0]
        at_close = joined.loc[
            joined.marker.eq("at_close") & joined.symbol.eq("BTCUSDT")].iloc[0]
        self.assertEqual(before.premium_available_time, times[bar-1]+900)
        self.assertEqual(at_close.premium_available_time, times[bar]+900)

    def test_geometry_resolves_same_bar_collision_as_stop(self):
        signed, exits, hits, reasons = geometry.resolve(
            np.array([100.0]), np.array([101.0]), np.array([99.0]),
            np.array([100.0]), np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64), np.array([1], dtype=np.int8),
            0.5, 0.5, 24)

        self.assertAlmostEqual(signed[0], -0.005)
        self.assertEqual(exits[0], 300)
        self.assertEqual(hits[0], 0)
        self.assertEqual(reasons[0], 0)


if __name__ == "__main__":
    unittest.main()
