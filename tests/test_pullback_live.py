import unittest

import numpy as np

from src import okx_trader
from src import pullback_bank as PB
from src import pullback_live as L

NOW = 1_800_000_000 - 1_800_000_000 % 3600       # an hour close


class FakeEx:
    """Exchange double: records every call, flat until an entry fills."""

    def __init__(self, fill_px=100.0, book_spread=0.0002, oco_ok=True, close_ok=True):
        self.calls = []
        self.size = 0.0
        self.fill_px = fill_px
        self.book_spread = book_spread
        self.oco_ok, self.close_ok = oco_ok, close_ok
        self.balance = 1000.0
        self.exit_px = None

    def _c(self, name, *a):
        self.calls.append((name,) + a)

    def get_position_size(self, creds, inst):
        return True, self.size

    def get_balance(self, creds):
        return True, self.balance

    def get_xperp_spec(self, inst):
        return {"ctVal": 0.01, "lotSz": 1.0, "minSz": 1.0, "tickSz": 0.01, "lever": 10}

    def get_last_price(self, inst):
        return self.fill_px

    def get_order_book(self, inst):
        import time
        h = self.fill_px * self.book_spread / 2
        return {"ts": str(int(time.time() * 1000)),
                "bids": [[str(self.fill_px - h), "100000"]], "asks": [[str(self.fill_px + h), "100000"]]}

    def ensure_leverage(self, creds, inst, lev):
        return True, "ok"

    def place_market_entry(self, creds, inst, direction, sz):
        self._c("market", inst, direction, sz)
        self.size = sz
        return True, "oid"

    def get_position_avg_px(self, creds, inst):
        return self.fill_px if self.size else None

    def place_protection_oco(self, creds, inst, direction, sl, tp):
        self._c("oco", inst, direction, sl, tp)
        return (True, "algo") if self.oco_ok else (False, "rejected")

    def close_position_market(self, creds, inst):
        self._c("close", inst)
        if self.close_ok:
            self.size = 0.0
            return True, "flat"
        return False, "pending"

    def cancel_protection(self, creds, inst, algo):
        self._c("cancel", inst, algo)
        return True, "ok"

    def get_last_fill_px(self, creds, inst, since_ts=None, side=None):
        return self.exit_px

    calc_contracts = staticmethod(okx_trader.calc_contracts)
    round_to_tick = staticmethod(okx_trader.round_to_tick)
    fmt_px_display = staticmethod(okx_trader.fmt_px_display)


def make(ex, state=None, margin=lambda bal: bal * 0.05, rules="strict3+short2"):
    msgs, saved = [], {}
    live = L.Live(symbols=["ETHUSDT"], rules_name=rules, ex=ex, inst_id_of=lambda s: "ETH-X",
                  candles=None, feed_prices=lambda: {}, dm=lambda uid, t: msgs.append(t),
                  users=lambda: [{"user_id": 7, "creds": {"k": 1}, "margin": margin}],
                  load_state=lambda: state or {}, save_state=lambda st: saved.update(st),
                  max_spread=0.0005, max_slip=0.0003, slip_min_n=3, leverage=10,
                  max_daily_loss=0.03, max_drawdown=0.15, sleep=lambda s: None)
    return live, msgs


def watch(side="LONG", level=100.0, atr=1.0, rule=0):
    return dict(sym="ETHUSDT", close=NOW, until=NOW + L.WATCH_SEC, rule=rule, level=level,
                atr=atr, side=side)


class EntryTest(unittest.TestCase):
    def test_no_order_until_price_touches_level(self):
        ex = FakeEx(); live, _ = make(ex)
        live.state["watch"].append(watch())
        live.tick(NOW + 20, {"ETHUSDT": 100.5})
        self.assertEqual(ex.calls, [])
        live.tick(NOW + 21, {"ETHUSDT": 99.99})
        self.assertEqual([c[0] for c in ex.calls], ["market", "oco"])
        self.assertEqual(live.state["watch"], [])

    def test_watch_expires_after_15_minutes(self):
        ex = FakeEx(); live, _ = make(ex)
        live.state["watch"].append(watch())
        live.tick(NOW + L.WATCH_SEC, {"ETHUSDT": 90.0})
        self.assertEqual(ex.calls, [])
        self.assertEqual(live.state["watch"], [])

    def test_levels_from_real_fill_and_user_margin(self):
        ex = FakeEx(fill_px=100.0); live, msgs = make(ex, margin=lambda bal: 50.0)
        live.state["watch"].append(watch(level=100.0, atr=1.0, rule=0))   # tp 0.5, sl 3 ATR
        live.tick(NOW + 5, {"ETHUSDT": 100.0})
        _, _, _, sz = ex.calls[0]
        self.assertEqual(sz, okx_trader.calc_contracts(50.0, 10, 100.0, ex.get_xperp_spec("x")))
        _, _, _, sl, tp = ex.calls[1]
        self.assertAlmostEqual(sl, 97.0); self.assertAlmostEqual(tp, 100.5)
        p = live.state["pos"][0]
        self.assertEqual((p["side"], p["entry"], p["algo"]), ("LONG", 100.0, "algo"))
        self.assertIn("вход ETHUSDT LONG", msgs[-1])

    def test_short_mirror(self):
        ex = FakeEx(fill_px=100.0); live, _ = make(ex)
        k = next(i for i, r in enumerate(live.rules) if r.get("side") == "SHORT")
        live.state["watch"].append(watch(side="SHORT", level=100.0, rule=k))
        live.tick(NOW + 5, {"ETHUSDT": 99.9})
        self.assertEqual(ex.calls, [])                          # below the level: no short
        live.tick(NOW + 6, {"ETHUSDT": 100.0})
        _, _, direction, sl, tp = ex.calls[1]
        self.assertEqual(direction, "SHORT")
        self.assertAlmostEqual(sl, 103.0); self.assertAlmostEqual(tp, 99.5)

    def test_wide_spread_blocks_entry(self):
        ex = FakeEx(book_spread=0.003); live, _ = make(ex)
        live.state["watch"].append(watch())
        live.tick(NOW + 5, {"ETHUSDT": 99.0})
        self.assertEqual(ex.calls, [])

    def test_not_flat_on_exchange_blocks_entry(self):
        ex = FakeEx(); ex.size = 3.0; live, _ = make(ex)
        live.state["watch"].append(watch())
        live.tick(NOW + 5, {"ETHUSDT": 99.0})
        self.assertEqual(ex.calls, [])

    def test_failed_protection_closes_position(self):
        ex = FakeEx(oco_ok=False); live, msgs = make(ex)
        live.state["watch"].append(watch())
        live.tick(NOW + 5, {"ETHUSDT": 99.0})
        self.assertEqual([c[0] for c in ex.calls], ["market", "oco", "close"])
        self.assertEqual(live.state["pos"], [])
        self.assertIn("закрыта для безопасности", msgs[-1])

    def test_naked_position_is_tracked_and_urgent(self):
        ex = FakeEx(oco_ok=False, close_ok=False); live, msgs = make(ex)
        live.state["watch"].append(watch())
        live.tick(NOW + 5, {"ETHUSDT": 99.0})
        self.assertIn("БЕЗ СТОПА", msgs[-1])
        self.assertEqual(len(live.state["pos"]), 1)             # manager keeps trying to close

    def test_equity_guard_stops_new_entries(self):
        ex = FakeEx(); live, msgs = make(ex)
        live.state["guard"]["7"] = {"peak": 2000.0, "day": "x", "daily_start": 2000.0}
        live.state["watch"].append(watch())
        live.tick(NOW + 5, {"ETHUSDT": 99.0})                    # 1000 vs peak 2000 = -50%
        self.assertEqual(ex.calls, [])
        self.assertIn("остановлены", msgs[-1])


class ManageTest(unittest.TestCase):
    def _open(self, ex):
        live, msgs = make(ex)
        live.state["watch"].append(watch())
        live.tick(NOW + 5, {"ETHUSDT": 99.0})
        return live, msgs

    def test_exchange_close_is_reported(self):
        ex = FakeEx(); live, msgs = self._open(ex)
        ex.size = 0.0; ex.exit_px = 100.5
        live.manage(NOW + 5 + 120)
        self.assertEqual(live.state["pos"], [])
        self.assertEqual(live.state["hist"][-1]["reason"], "стоп/тейк")
        self.assertIn("✅", msgs[-1])

    def test_time_exit_after_48h(self):
        ex = FakeEx(); live, msgs = self._open(ex)
        ex.exit_px = 99.5
        live.manage(NOW + 5 + L.HOLD_SEC + 1)
        self.assertEqual([c[0] for c in ex.calls][-2:], ["close", "cancel"])
        self.assertEqual(live.state["pos"], [])
        self.assertIn("48 часов", msgs[-1])


class SlippageGuardTest(unittest.TestCase):
    def test_bad_slippage_excludes_coin_then_pauses(self):
        ex = FakeEx(); live, msgs = make(ex)
        for i in range(3):
            live._record_slip("ETHUSDT", 0.001, NOW + i)
        self.assertIn("ETHUSDT", live.state["excluded"])
        self.assertIn("исключена", msgs[-1])
        for s in ("AUSDT", "BUSDT", "CUSDT"):
            for i in range(2):
                live._record_slip(s, 0.001, NOW + i)
        self.assertTrue(live.state["paused"])


class ScanTest(unittest.TestCase):
    def test_signal_becomes_watch_once(self):
        t = NOW - PB.BAR * (4 * (PB.MIN_HOURS + 60)) + np.arange(4 * (PB.MIN_HOURS + 60)) * PB.BAR
        a = np.c_[np.full((len(t), 4), 100.0), np.ones(len(t))]

        class Store:
            def get(self, sym, now):
                return t, a
        ex = FakeEx(); live, _ = make(ex)
        live.candles = Store()
        orig = PB.hourly_signals
        PB.hourly_signals = lambda *a_, **k: {NOW: (0, 100.0, 1.0)}
        try:
            live.maybe_scan(NOW + 15)
            live.maybe_scan(NOW + 20)
        finally:
            PB.hourly_signals = orig
        self.assertEqual(len(live.state["watch"]), 1)
        self.assertEqual(live.state["watch"][0]["until"], NOW + L.WATCH_SEC)


if __name__ == "__main__":
    unittest.main()
