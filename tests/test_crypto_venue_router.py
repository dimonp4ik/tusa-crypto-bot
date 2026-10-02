import unittest
from unittest.mock import patch

from src.crypto_venue_router import (
    MODULES, ROBUST_EXCLUDED_SYMBOLS, ROBUST_SYMBOLS,
    SHADOW_EXPERIMENTAL_MODULES, btc_context, shadow_experimental_setups,
    venue_setups,
)
from src import binance_client


class CryptoVenueRouterTests(unittest.TestCase):
    def test_frozen_module_count_and_market_bracket(self):
        context = {
            "btc_regime": "bear_rally", "btc_atr_pct": .004,
            "btc_return20_atr": -2.0, "btc_fast_slow_atr": 0.0,
            "btc_slow_long_atr": 0.0, "btc_eff20": .3,
            "btc_eff96": .2, "btc_return96_atr": 0.0,
        }
        signal = {"family": "range_reversion", "direction": "LONG",
                  "atr": 2.0, "eff_ratio": .2, "z": -2.2, "abs_z": 2.2}
        with patch("src.crypto_venue_router.btc_context", return_value=context), \
                patch("src.crypto_venue_router.family_signals", return_value=[signal]):
            setups = venue_setups(
                {}, {}, entry_time=8 * 3600, market_price=100.0,
                symbol="AAVEUSDT",
            )
        self.assertEqual(len(MODULES), 7)
        self.assertEqual(
            [(module.name, module.target_r) for module in MODULES],
            [
                ("pullback_long_bull_europe", .5),
                ("rr_long_bear_rally_europe", .5),
                ("rr_long_bear_day", .5),
                ("rr_short_bull_pullback_asia", .25),
                ("rr_short_bear_asia", .25),
                ("rr_short_bull_evening", .25),
                ("breakout_long_bull_pullback_evening", .25),
            ],
        )
        self.assertEqual(len(setups), 1)
        self.assertEqual(setups[0]["module"], "rr_long_bear_rally_europe")
        self.assertEqual(setups[0]["entry_source"], "MARKET")
        self.assertEqual((setups[0]["entry"], setups[0]["sl"], setups[0]["tp"]),
                         (100.0, 96.0, 102.0))
        self.assertNotIn("size", setups[0])
        self.assertNotIn("size_mult", setups[0])

    def test_session_is_part_of_module_condition(self):
        context = {
            "btc_regime": "bear_rally", "btc_atr_pct": .004,
            "btc_return20_atr": -2.0, "btc_fast_slow_atr": 0.0,
            "btc_slow_long_atr": -2.0, "btc_eff20": .3,
            "btc_eff96": .2, "btc_return96_atr": 0.0,
        }
        signal = {"family": "range_reversion", "direction": "LONG",
                  "atr": 2.0, "eff_ratio": .2, "z": -2.2, "abs_z": 2.2}
        with patch("src.crypto_venue_router.btc_context", return_value=context), \
                patch("src.crypto_venue_router.family_signals", return_value=[signal]):
            self.assertEqual(venue_setups({}, {}, entry_time=14 * 3600,
                                          market_price=100.0,
                                          symbol="AAVEUSDT"), [])

    def test_asia_pullback_short_uses_conditional_half_r_target(self):
        base_context = {
            "btc_regime": "bull_pullback", "btc_atr_pct": .004,
            "btc_return20_atr": -2.0, "btc_fast_slow_atr": -1.5,
            "btc_slow_long_atr": 1.0, "btc_eff96": .2,
            "btc_return96_atr": 0.0,
        }
        signal = {"family": "range_reversion", "direction": "SHORT",
                  "atr": 2.0, "eff_ratio": .2, "z": 2.2, "abs_z": 2.2}

        def setup(efficiency):
            context = {**base_context, "btc_eff20": efficiency}
            with patch("src.crypto_venue_router.btc_context",
                       return_value=context), patch(
                       "src.crypto_venue_router.family_signals",
                       return_value=[signal]):
                return venue_setups({}, {}, entry_time=3 * 3600,
                                    market_price=100.0, symbol="AAVEUSDT")[0]

        choppy = setup(.099999)
        directional = setup(.1)
        self.assertEqual((choppy["target_r"], choppy["tp"]), (.25, 99.0))
        self.assertEqual((directional["target_r"], directional["tp"]),
                         (.5, 98.0))

    def test_evening_short_rejects_extreme_cross_sectional_extension(self):
        context = {
            "btc_regime": "bull", "btc_atr_pct": .004,
            "btc_return20_atr": -1.8, "btc_fast_slow_atr": 1.0,
            "btc_slow_long_atr": 4.5, "btc_eff20": .2,
            "btc_eff96": .2, "btc_return96_atr": 2.0,
        }
        base = {"family": "range_reversion", "direction": "SHORT",
                "atr": 2.0, "eff_ratio": .2}
        with patch("src.crypto_venue_router.btc_context", return_value=context), \
                patch("src.crypto_venue_router.family_signals",
                      return_value=[{**base, "z": 3.0, "abs_z": 3.0}]):
            accepted = venue_setups({}, {}, entry_time=20 * 3600,
                                    market_price=100.0, symbol="AAVEUSDT")
        with patch("src.crypto_venue_router.btc_context", return_value=context), \
                patch("src.crypto_venue_router.family_signals",
                      return_value=[{**base, "z": 3.000001,
                                     "abs_z": 3.000001}]):
            rejected = venue_setups({}, {}, entry_time=20 * 3600,
                                    market_price=100.0, symbol="AAVEUSDT")
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["module"], "rr_short_bull_evening")
        self.assertEqual(rejected, [])

    def test_frozen_shadow_supplement_is_separate_from_production(self):
        context = {
            "btc_regime": "bull_pullback", "btc_atr_pct": .004,
            "btc_return20_atr": 2.0, "btc_fast_slow_atr": -1.0,
            "btc_slow_long_atr": 4.0, "btc_eff20": .2,
            "btc_eff96": .2, "btc_return96_atr": 2.0,
        }
        signal = {"family": "trend_pullback", "direction": "SHORT",
                  "atr": 2.0, "eff_ratio": .2, "z": 1.25,
                  "abs_z": 1.25}
        with patch("src.crypto_venue_router.btc_context", return_value=context), \
                patch("src.crypto_venue_router.family_signals",
                      return_value=[signal]):
            production = venue_setups(
                {}, {}, entry_time=20 * 3600, market_price=100.0,
                symbol="AAVEUSDT",
            )
            shadow = shadow_experimental_setups(
                {}, {}, entry_time=20 * 3600, market_price=100.0,
                symbol="AAVEUSDT",
            )
        self.assertEqual(len(SHADOW_EXPERIMENTAL_MODULES), 2)
        self.assertEqual(production, [])
        self.assertEqual(len(shadow), 1)
        self.assertEqual(
            shadow[0]["module"],
            "shadow_pullback_short_bull_pullback_evening",
        )
        self.assertTrue(shadow[0]["_shadow_experiment"])
        self.assertEqual((shadow[0]["target_r"], shadow[0]["entry_source"]),
                         (.25, "MARKET"))

    def test_monthly_complement_is_shadow_only(self):
        context = {
            "btc_regime": "bull", "btc_atr_pct": .008,
            "btc_return20_atr": 0.0, "btc_fast_slow_atr": 1.0,
            "btc_slow_long_atr": 2.0, "btc_eff20": .2,
            "btc_eff96": .2, "btc_return96_atr": 1.0,
        }
        signal = {"family": "range_reversion", "direction": "LONG",
                  "atr": 2.0, "eff_ratio": .2, "z": -1.25,
                  "abs_z": 1.25}
        with patch("src.crypto_venue_router.btc_context", return_value=context), \
                patch("src.crypto_venue_router.family_signals",
                      return_value=[signal]):
            production = venue_setups(
                {}, {}, entry_time=2 * 3600, market_price=100.0,
                symbol="AAVEUSDT",
            )
            shadow = shadow_experimental_setups(
                {}, {}, entry_time=2 * 3600, market_price=100.0,
                symbol="AAVEUSDT",
            )
        self.assertEqual(production, [])
        self.assertEqual(len(shadow), 1)
        self.assertEqual(shadow[0]["module"],
                         "shadow_rr_long_bull_asia_high_vol")
        self.assertTrue(shadow[0]["_shadow_experiment"])
        self.assertEqual((shadow[0]["target_r"], shadow[0]["entry_source"]),
                         (.25, "MARKET"))

    def test_robust_symbol_gate_fails_closed(self):
        self.assertEqual(ROBUST_EXCLUDED_SYMBOLS,
                         {"ADAUSDT", "DOTUSDT", "NEARUSDT", "TAOUSDT"})
        self.assertNotIn("ADAUSDT", ROBUST_SYMBOLS)
        for symbol in (None, "ADAUSDT", "DOTUSDT", "NEARUSDT", "TAOUSDT",
                       "UNKNOWNUSDT"):
            self.assertEqual(
                venue_setups({}, {}, entry_time=20 * 3600,
                             market_price=100.0, symbol=symbol),
                [],
            )

    def test_btc_context_requires_full_closed_history(self):
        values = list(range(300))
        candles = {"close": values, "high": [x + 1 for x in values],
                   "low": [x - 1 for x in values]}
        self.assertIsNone(btc_context(candles))

    def test_xperp_fetch_paginates_for_btc_context(self):
        def row(timestamp):
            value = float(timestamp)
            return [str(timestamp * 1000), str(value), str(value + 1),
                    str(value - 1), str(value), "0", "1", "0", "1"]

        first = [row(timestamp) for timestamp in range(500, 200, -1)]
        second = [row(timestamp) for timestamp in range(200, 98, -1)]
        with binance_client._kl_lock:
            binance_client._kl_cache.clear()
        with patch.object(binance_client, "get_xperp_instruments",
                          return_value={"BTC": "BTC-USDT-SWAP"}), \
                patch.object(binance_client, "_okx_get",
                             side_effect=[{"data": first}, {"data": second}]) as request:
            candles = binance_client.get_klines_xperp("BTCUSDT", limit=400)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(len(candles["time"]), 400)
        self.assertEqual(candles["time"], sorted(candles["time"]))
        self.assertEqual(request.call_args_list[1].args[1]["after"], first[-1][0])


if __name__ == "__main__":
    unittest.main()
