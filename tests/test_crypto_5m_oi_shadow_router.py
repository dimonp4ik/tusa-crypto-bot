import unittest

from src.crypto_5m_oi_shadow_router import (
    MODULE_NAME, candidate_context, shadow_setup,
)


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
        setup = shadow_setup(
            coin, btc, oi_history(), entry_time=170 * 3600,
            market_price=182.0, symbol="TESTUSDT")
        self.assertEqual(setup["module"], MODULE_NAME)
        self.assertEqual(setup["entry"], 182.0)
        self.assertEqual(setup["target_r"], .25)
        self.assertTrue(setup["_shadow_experiment"])
        self.assertTrue(setup["_five_minute_oi_shadow"])

    def test_crowded_oi_drop_fails_closed(self):
        coin, btc = qualifying_candles()
        self.assertIsNone(shadow_setup(
            coin, btc, oi_history(75.0), entry_time=170 * 3600,
            market_price=182.0, symbol="TESTUSDT"))

    def test_missing_or_short_oi_fails_closed(self):
        coin, btc = qualifying_candles()
        for history in ({}, {"time": [0, 3600], "oi": [100, 101]}):
            self.assertIsNone(shadow_setup(
                coin, btc, history, entry_time=170 * 3600,
                market_price=182.0, symbol="TESTUSDT"))

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
