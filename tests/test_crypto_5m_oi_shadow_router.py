import unittest
from unittest.mock import patch

import src.crypto_5m_oi_shadow_router as router

from src.crypto_5m_oi_shadow_router import (
    SHADOW_ONLY_SYMBOLS, MODULE_NAME, candidate_context, shadow_setup,
    symbol_context_allowed,
    symbol_context_variant,
)


class ShadowUniverseSafetyTests(unittest.TestCase):
    def test_ondo_is_shadow_transfer_only(self):
        from src.crypto_venue_router import ROBUST_SYMBOLS

        self.assertIn("ONDOUSDT", SHADOW_ONLY_SYMBOLS)
        self.assertNotIn("ONDOUSDT", ROBUST_SYMBOLS)

    def test_ondo_transfer_is_refused_by_absolute_move_route(self):
        self.assertIsNone(symbol_context_variant(
            "ONDOUSDT", {"direction": "SHORT"}, 12 * 3600))
        self.assertFalse(symbol_context_allowed(
            "ONDOUSDT", {"direction": "SHORT"}, 8 * 3600))
        self.assertFalse(symbol_context_allowed(
            "ONDOUSDT", {"direction": "LONG"}, 12 * 3600))
        self.assertFalse(symbol_context_allowed(
            "BTCUSDT", {"direction": "LONG"}, 8 * 3600))

    def test_invalidated_candidate_fails_closed(self):
        self.assertFalse(router.SHADOW_CANDIDATE_ENABLED)
        self.assertIsNone(symbol_context_variant(
            "BTCUSDT", {"direction": "LONG"}, 8 * 3600))

    def test_hype_is_refused_only_by_the_five_minute_route(self):
        self.assertIsNone(symbol_context_variant(
            "HYPEUSDT", {"direction": "SHORT", "btc_eff20": 0.0},
            12 * 3600))


def candles(values):
    return {
        "time": [index * 300 for index in range(len(values))],
        "open": [value + .1 for value in values],
        "high": [value + .3 for value in values],
        "low": [value - .3 for value in values],
        "close": list(values),
        "volume": [1.0] * len(values),
    }


def qualifying_candles():
    count = 110
    coin = [200 - .2 * index for index in range(count)]
    coin[-2], coin[-1] = 183.0, 182.0
    btc = [1000 - 2 * index for index in range(count)]
    for index in range(count - 13, count):
        btc[index] = 808 + index - (count - 13)
    return candles(coin), candles(btc)


def oi_history(end_value=90.0):
    count = 170
    return {
        "time": [index * 3600 for index in range(count)],
        "oi": [100 + (end_value - 100) * index / (count - 1)
               for index in range(count)],
    }


class FiveMinuteOiShadowRouterTests(unittest.TestCase):
    def test_frozen_short_rule_produces_market_shadow_setup(self):
        coin, btc = qualifying_candles()
        context = candidate_context(coin, btc)
        self.assertEqual(context["direction"], "SHORT")
        coin["high"] = [value + 1.0 for value in coin["close"]]
        coin["low"] = [value - 1.0 for value in coin["close"]]
        with patch.object(router, "SHADOW_CANDIDATE_ENABLED", True):
            setup = shadow_setup(
                coin, btc, oi_history(110.0), entry_time=170 * 3600,
                market_price=182.0, symbol="TESTUSDT")
        self.assertEqual(setup["module"], MODULE_NAME)
        self.assertEqual(setup["entry"], 182.0)
        self.assertEqual(setup["target_r"], .5)
        self.assertEqual(setup["runner_target_r"], 2.0)
        self.assertEqual(setup["first_target_move_pct"], .5)
        self.assertGreaterEqual(setup["observed_atr_pct"], .2)
        self.assertTrue(setup["_shadow_experiment"])
        self.assertTrue(setup["_five_minute_oi_shadow"])

    def test_crowded_oi_drop_fails_closed(self):
        coin, btc = qualifying_candles()
        with patch.object(router, "SHADOW_CANDIDATE_ENABLED", True):
            self.assertIsNone(shadow_setup(
                coin, btc, oi_history(75.0), entry_time=170 * 3600,
                market_price=182.0, symbol="TESTUSDT"))

    def test_missing_or_short_oi_fails_closed(self):
        coin, btc = qualifying_candles()
        with patch.object(router, "SHADOW_CANDIDATE_ENABLED", True):
            for history in ({}, {"time": [0, 3600], "oi": [100, 101]}):
                self.assertIsNone(shadow_setup(
                    coin, btc, history, entry_time=170 * 3600,
                    market_price=182.0, symbol="TESTUSDT"))

    def test_micro_target_is_not_published_even_in_shadow(self):
        coin, btc = qualifying_candles()
        with patch.object(router, "SHADOW_CANDIDATE_ENABLED", True):
            self.assertIsNone(shadow_setup(
                coin, btc, oi_history(110.0), entry_time=170 * 3600,
                market_price=1000.0, symbol="TESTUSDT"))

    def test_repeated_pullback_flag_is_not_a_new_setup(self):
        coin, btc = qualifying_candles()
        # Make the prior bar another valid short instead of a fresh transition.
        coin["close"][-3] = 184.0
        coin["close"][-2] = 183.0
        self.assertIsNone(candidate_context(coin, btc))

    def test_unaligned_btc_bar_fails_closed(self):
        coin, btc = qualifying_candles()
        btc["time"][-1] += 1
        self.assertIsNone(candidate_context(coin, btc))


if __name__ == "__main__":
    unittest.main()
