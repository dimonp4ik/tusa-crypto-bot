import concurrent.futures
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["BOT_STARTUP_ENABLED"] = "0"

from src import db


class SetupDedupTests(unittest.TestCase):
    def test_concurrent_scans_claim_closed_bar_once(self):
        analysis = {
            "symbol": "TESTUSDT", "direction": "LONG", "decision": "LONG",
            "current_price": 100.0, "atr": 1.0, "fixed_stop_atr": 2.0,
            "fixed_target_r": 0.5, "signal_bar_ts": 1_700_000_000.0,
            "source": "venue_regime",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "dedup.db")
            with patch.object(db, "DB_PATH", path), \
                    patch("src.telegram_notifier.bracket_for_analysis",
                          return_value=(101.0, 102.0, 98.0)):
                db.init_db()
                with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                    ids = list(pool.map(
                        lambda _: db.log_setup_candidate_once(analysis), range(40)))
                with db._conn() as conn:
                    rows = conn.execute(
                        "SELECT COUNT(*) FROM setup_log WHERE symbol='TESTUSDT'"
                    ).fetchone()[0]

        self.assertEqual(sum(item is not None for item in ids), 1)
        self.assertEqual(rows, 1)

    def test_blocked_candidate_keeps_exact_execution_contract_and_realised_r(self):
        analysis = {
            "symbol": "TESTUSDT", "direction": "LONG", "decision": "LONG",
            "current_price": 100.0, "atr": 1.0, "fixed_stop_atr": 2.0,
            "fixed_target_r": .5, "fixed_runner_target_r": 2.0,
            "signal_bar_ts": 1_700_000_000.0,
            "source": "venue_shadow_profit_first_2r",
            "strategy_tp1_close_frac": .05,
            "strategy_runner_mode": "fixed_be",
            "strategy_bar_seconds": 900,
            "strategy_entry_bar_ts": 1_700_000_900.0,
            "strategy_stop_on_close": False,
            "strategy_max_bars": 48,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "blocked.db")
            with patch.object(db, "DB_PATH", path), \
                    patch("src.telegram_notifier.bracket_for_analysis",
                          return_value=(101.0, 104.0, 98.0)):
                db.init_db()
                setup_id = db.log_setup_candidate_once(analysis)
                db.mark_setup_blocked(setup_id, "hour_cap")
                rows = db.get_unresolved_venue_setups()
                db.mark_setup_resolved(
                    setup_id, "TP2", 1, 1, net_r=1.925)
                stats = db.get_cap_impact_stats(0)["hour_cap"]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["strategy_bar_seconds"], 900)
        self.assertEqual(rows[0]["strategy_max_bars"], 48)
        self.assertEqual(rows[0]["strategy_runner_mode"], "fixed_be")
        self.assertAlmostEqual(rows[0]["strategy_tp1_close_frac"], .05)
        self.assertEqual(stats["n"], 1)
        self.assertAlmostEqual(stats["saved_r"], -1.925)


if __name__ == "__main__":
    unittest.main()
