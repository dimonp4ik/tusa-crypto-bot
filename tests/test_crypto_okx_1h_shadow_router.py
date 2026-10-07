import json
import math
import unittest

from src.crypto_okx_1h_shadow_router import (
    CASCADE_PATH, MODEL_PATH, QUALITY_THRESHOLD, SUPPORTED_SYMBOLS,
    _symbol_features, shadow_setups,
)


class CryptoOkxOneHourShadowRouterTests(unittest.TestCase):
    def test_frozen_artifact_has_material_move_contract(self):
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(model["bar"], "1H")
        self.assertEqual(model["target_pct"], .5)
        self.assertEqual(model["stop_pct"], 1.0)
        self.assertEqual(model["hold_bars"], 12)
        self.assertEqual(model["quality_threshold"], QUALITY_THRESHOLD)
        self.assertEqual(len(model["features"]), 52)
        cascade = json.loads(CASCADE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(sorted(map(float, cascade["models"])), [1.0, 1.5, 2.0])
        self.assertEqual([cascade["models"][key]["hold_bars"]
                          for key in ("1.0", "1.5", "2.0")], [24, 36, 48])

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
