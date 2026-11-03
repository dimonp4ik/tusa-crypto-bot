import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from src import crypto_profit_first_material_shadow_router as router


SAMPLE = {
    "symbol": "BILLUSDT",
    "entry_time": 1792814400.0,
    "entry": .006117,
    "risk_pct": .0177724841775847,
    "probability": .3570134432603048,
    "signal": {
        "direction": "SHORT",
        "family": "trend_pullback",
        "eff_ratio": .2968299711815555,
        "z": .2533373367204087,
    },
    "context": {
        "btc_fast_slow_atr": -4.827231871258964,
        "btc_eff20": .2033491270201879,
        "btc_return20_atr": 2.582396231969475,
        "btc_z20": .6762829954575805,
        "btc_atr_pct": .0014991631695018,
        "btc_slow_long_atr": -7.216483294083128,
        "btc_return96_atr": -6.411068589932362,
        "btc_eff96": .0869691616163865,
        "btc_regime": "bear",
    },
}


class ProfitFirstMaterialShadowRouterTests(unittest.TestCase):
    def test_portable_score_matches_frozen_research_row(self):
        probability = router.model_probability(
            symbol=SAMPLE["symbol"], signal=SAMPLE["signal"],
            context=SAMPLE["context"], entry_time=SAMPLE["entry_time"],
            risk_pct=SAMPLE["risk_pct"])
        self.assertAlmostEqual(probability, SAMPLE["probability"], places=15)

    def test_artifact_is_bound_to_audited_source_report(self):
        source = Path(
            "reports/audit_2026_09_08/crypto_profit_first_material_target_oct18.json")
        observed = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(router.MODEL["source_report_sha256"], observed)
        self.assertEqual(router.MODEL["near_target_r"], .5)
        self.assertEqual(router.MODEL["near_close_fraction"], .05)
        self.assertEqual(router.MODEL["runner_target_r"], 2.0)
        self.assertEqual(router.MODEL["hold_bars"], 48)

    def test_setup_is_market_entry_and_permanently_shadow_only(self):
        signal = {
            **SAMPLE["signal"],
            "atr": SAMPLE["entry"] * SAMPLE["risk_pct"] / 2.0,
            "abs_z": abs(SAMPLE["signal"]["z"]),
        }
        candles = {"close": [SAMPLE["entry"]] * router.REQUIRED_SYMBOL_HISTORY_BARS}
        btc_candles = {"close": [1.0] * router.REQUIRED_BTC_HISTORY_BARS}
        with patch.object(router, "family_signals", return_value=[signal]), \
             patch.object(router, "btc_context", return_value=SAMPLE["context"]):
            rows = router.shadow_setups(
                candles, btc_candles,
                entry_time=SAMPLE["entry_time"],
                market_price=SAMPLE["entry"], symbol=SAMPLE["symbol"])
        self.assertEqual(len(rows), 1)
        setup = rows[0]
        self.assertEqual(setup["entry_source"], "MARKET")
        self.assertTrue(setup["_shadow_experiment"])
        self.assertTrue(setup["_profit_first_material_shadow"])
        self.assertAlmostEqual(setup["model_quality"], SAMPLE["probability"])
        self.assertAlmostEqual(setup["target_r"], .5)
        self.assertAlmostEqual(setup["runner_target_r"], 2.0)
        self.assertAlmostEqual(setup["tp1_close_fraction"], .05)
        self.assertEqual(setup["hold_bars"], 48)
        self.assertGreater(setup["sl"], setup["entry"])
        self.assertLess(setup["tp"], setup["entry"])

    def test_history_requirements_and_filter_reason_are_explainable(self):
        audit = {}
        rows = router.shadow_setups(
            {"close": [1.0] * (router.REQUIRED_SYMBOL_HISTORY_BARS - 1)},
            {"close": [1.0] * router.REQUIRED_BTC_HISTORY_BARS},
            entry_time=SAMPLE["entry_time"], market_price=1.0,
            symbol="BTCUSDT", diagnostics=audit)
        self.assertEqual(rows, [])
        self.assertEqual(audit["result"], "insufficient_symbol_history")
        self.assertEqual(audit["required_symbol_history_bars"], 100)
        self.assertEqual(audit["required_btc_history_bars"], 385)

        signal = {
            **SAMPLE["signal"], "atr": .001, "abs_z": 1.0,
        }
        with patch.object(router, "btc_context", return_value=SAMPLE["context"]), \
             patch.object(router, "family_signals", return_value=[signal]):
            rows = router.shadow_setups(
                {"close": [1.0] * router.REQUIRED_SYMBOL_HISTORY_BARS},
                {"close": [1.0] * router.REQUIRED_BTC_HISTORY_BARS},
                entry_time=SAMPLE["entry_time"], market_price=1.0,
                symbol="BTCUSDT", diagnostics=audit)
        self.assertEqual(rows, [])
        self.assertEqual(audit["result"], "all_family_signals_filtered")
        self.assertEqual(
            audit["rejected"], {"model_risk_outside_1_to_2pct": 1})

    def test_model_uses_immutable_bar_open_while_bracket_uses_market(self):
        signal = {
            **SAMPLE["signal"], "atr": .75,
            "abs_z": abs(SAMPLE["signal"]["z"]),
        }
        with patch.object(router, "btc_context", return_value=SAMPLE["context"]), \
             patch.object(router, "family_signals", return_value=[signal]), \
             patch.object(router, "model_probability",
                          return_value=router.MODEL["quality_threshold"] + .01) as score:
            rows = router.shadow_setups(
                {"close": [100.0] * router.REQUIRED_SYMBOL_HISTORY_BARS},
                {"close": [1.0] * router.REQUIRED_BTC_HISTORY_BARS},
                entry_time=SAMPLE["entry_time"], market_price=101.0,
                model_price=100.0, symbol="BTCUSDT")
        self.assertEqual(len(rows), 1)
        setup = rows[0]
        self.assertAlmostEqual(score.call_args.kwargs["risk_pct"], .015)
        self.assertAlmostEqual(setup["model_risk_pct"], .015)
        self.assertAlmostEqual(setup["risk_pct"], 1.5 / 101.0)
        self.assertAlmostEqual(setup["entry_reference_price"], 100.0)
        self.assertAlmostEqual(setup["entry_drift_pct"], 1.0)
        self.assertAlmostEqual(setup["entry"], 101.0)
        self.assertAlmostEqual(setup["sl"], 102.5)

    def test_market_drift_can_only_remove_out_of_band_execution(self):
        signal = {
            **SAMPLE["signal"], "atr": .75,
            "abs_z": abs(SAMPLE["signal"]["z"]),
        }
        audit = {}
        with patch.object(router, "btc_context", return_value=SAMPLE["context"]), \
             patch.object(router, "family_signals", return_value=[signal]), \
             patch.object(router, "model_probability") as score:
            rows = router.shadow_setups(
                {"close": [100.0] * router.REQUIRED_SYMBOL_HISTORY_BARS},
                {"close": [1.0] * router.REQUIRED_BTC_HISTORY_BARS},
                entry_time=SAMPLE["entry_time"], market_price=200.0,
                model_price=100.0, symbol="BTCUSDT", diagnostics=audit)
        self.assertEqual(rows, [])
        score.assert_not_called()
        self.assertEqual(
            audit["rejected"], {"execution_risk_outside_1_to_2pct": 1})


if __name__ == "__main__":
    unittest.main()
