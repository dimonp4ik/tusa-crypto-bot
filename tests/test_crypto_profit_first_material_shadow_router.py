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
        with patch.object(router, "family_signals", return_value=[signal]), \
             patch.object(router, "btc_context", return_value=SAMPLE["context"]):
            rows = router.shadow_setups(
                {"close": [SAMPLE["entry"]]}, {"close": []},
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


if __name__ == "__main__":
    unittest.main()
