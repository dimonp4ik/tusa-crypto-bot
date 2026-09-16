import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.regime_filters import crypto_pullback_short
from src.telegram_notifier import bracket_for_analysis


class CryptoRegimeFilterTests(unittest.TestCase):
    def test_regime_paper_row_is_logged_once_with_exact_bracket(self):
        from src import db
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                db, "DB_PATH", str(Path(tmp) / "test.db")):
            db.init_db()
            analysis = {"symbol": "TEST", "direction": "SHORT", "decision": "SHORT",
                        "current_price": 100, "atr": 2, "fixed_stop_atr": 2,
                        "fixed_target_r": .25, "source": "regime", "signal_bar_ts": 123}
            first = db.log_setup_candidate_once(analysis)
            second = db.log_setup_candidate_once(analysis)
            self.assertIsNotNone(first)
            self.assertIsNone(second)
            with db._conn() as connection:
                row = connection.execute("SELECT tp1,tp2,sl FROM setup_log").fetchone()
            self.assertEqual(tuple(row), (99.0, 99.0, 104.0))

    def test_fixed_family_bracket_closes_full_trade_at_target(self):
        analysis = {"direction": "SHORT", "atr": 2,
                    "fixed_stop_atr": 2, "fixed_target_r": .25}
        tp1, tp2, sl = bracket_for_analysis(analysis, 100)
        self.assertEqual((tp1, tp2, sl), (99.0, 99.0, 104.0))

    def test_frozen_short_pullback_rule_and_client_market_entry(self):
        closes = [210 - i for i in range(100)]
        closes += [110 - i * (20 / 17) for i in range(18)] + [101.5, 101.25]
        candles = {
            "close": closes,
            "high": [value + .05 for value in closes],
            "low": [value - .05 for value in closes],
        }
        btc_closes = [200 - i for i in range(120)]
        btc = {"close": btc_closes}
        setup = crypto_pullback_short(candles, btc, market_price=101.1, target_r=.5)
        self.assertIsNotNone(setup)
        self.assertEqual(setup["direction"], "SHORT")
        self.assertEqual(setup["entry"], 101.1)
        self.assertGreaterEqual(setup["eff_ratio"], .25)
        self.assertLessEqual(abs(setup["z"]), 1)
        self.assertAlmostEqual(setup["entry"] - setup["tp"],
                               .5 * (setup["sl"] - setup["entry"]))

    def test_rejects_long_side_shape(self):
        closes = [100 + i for i in range(120)]
        candles = {"close": closes, "high": [x + .1 for x in closes],
                   "low": [x - .1 for x in closes]}
        btc = {"close": [200 - i for i in range(120)]}
        self.assertIsNone(crypto_pullback_short(candles, btc))

    def test_rejects_pullback_outside_bearish_btc_regime(self):
        closes = [210 - i for i in range(100)]
        closes += [110 - i * (20 / 17) for i in range(18)] + [101.5, 101.25]
        candles = {"close": closes, "high": [x + .05 for x in closes],
                   "low": [x - .05 for x in closes]}
        btc = {"close": [100 + i for i in range(120)]}
        self.assertIsNone(crypto_pullback_short(candles, btc))

    def test_strong_btc_regime_uses_the_frozen_half_r_target(self):
        closes = [210 - i for i in range(100)]
        closes += [110 - i * (20 / 17) for i in range(18)] + [101.5, 101.25]
        candles = {"close": closes, "high": [x + .05 for x in closes],
                   "low": [x - .05 for x in closes]}
        btc = {"close": [200 - i for i in range(120)]}
        self.assertEqual(crypto_pullback_short(candles, btc)["target_r"], .5)


if __name__ == "__main__":
    unittest.main()
