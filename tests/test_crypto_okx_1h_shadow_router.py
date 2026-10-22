import json
import math
import unittest

from src.crypto_okx_1h_shadow_router import (
    MODEL_PATH, POLICY_PATH, QUALITY_THRESHOLD, SUPPORTED_SYMBOLS,
    _float32, _route_allowed, _symbol_features, shadow_setups,
)


class CryptoOkxOneHourShadowRouterTests(unittest.TestCase):
    def test_frozen_artifact_has_material_move_contract(self):
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(model["bar"], "1H")
        self.assertEqual(policy["target_pct"], 2.0)
        self.assertEqual(policy["stop_pct"], 1.0)
        self.assertEqual(policy["hold_bars"], 12)
        self.assertEqual(policy["exit_profile"], "fixed_2pct_12h")
        self.assertEqual(policy["quality_threshold"], QUALITY_THRESHOLD)
        self.assertTrue(policy["shadow_only"])
        self.assertEqual(len(model["features"]), 52)

    def test_frozen_direction_route(self):
        self.assertTrue(_route_allowed("BTCUSDT", -1, {"trend"}))
        self.assertTrue(_route_allowed("ETHUSDT", 1, {"pullback"}))
        self.assertTrue(_route_allowed("AVAXUSDT", 1, {"trend"}))
        self.assertFalse(_route_allowed("BTCUSDT", 1, {"trend"}))

    def test_model_inputs_use_training_float32_precision(self):
        self.assertNotEqual(_float32(0.7227249797903944), 0.7227249797903944)
        self.assertEqual(_float32(1.0), 1.0)

    def test_incomplete_cross_market_input_fails_closed(self):
        self.assertEqual(shadow_setups({}, {}), {})
        partial = {next(iter(SUPPORTED_SYMBOLS)): {"time": [1]}}
        self.assertEqual(shadow_setups(partial, {}), {})

    def test_latest_feature_row_is_finite_and_uses_closed_history(self):
        count = 260
        close = [100.0]
        for index in range(1, count):
            change = .0008 + .0015 * math.sin(index / 3.7)
            close.append(close[-1] * (1 + change))
        candle = {
            "time": [1_700_000_000 + 3600 * index for index in range(count)],
            "open": [value * (1 - .0003 * math.sin(index))
                     for index, value in enumerate(close)],
            "high": [value * (1.003 + .0005 * math.sin(index / 2))
                     for index, value in enumerate(close)],
            "low": [value * (0.997 - .0004 * math.cos(index / 2))
                    for index, value in enumerate(close)],
            "close": close,
            "volume_quote": [1_000_000 * (1.1 + .2 * math.sin(index / 5))
                             for index in range(count)],
        }
        features = _symbol_features(candle)
        self.assertIsNotNone(features)
        self.assertTrue(all(math.isfinite(value) for value in features.values()))
        self.assertAlmostEqual(features["ret1"], close[-1] / close[-2] - 1)
        self.assertAlmostEqual(features["ret168"], close[-1] / close[-169] - 1)


if __name__ == "__main__":
    unittest.main()
