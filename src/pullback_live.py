"""LIVE trading for the pullback bank (src/pullback_bank.py) - real OKX orders.

Trades for every onboarded autotrade user (same list and same keys as the old autotrader).
Position size is the user's own setting, exactly as before: a percent of the deposit or a
fixed dollar margin per trade (autotrader._margin_for), at the usual fixed leverage.

No resting limit orders (owner's choice, 11.09.2026):
  * Every hour, a few seconds after the close, signals are computed from CLOSED 15m bars by
    the same functions the backtest and the paper runner use (pullback_bank.hourly_signals
    with the persisted trend regime). A signal becomes a WATCH that lives 15 minutes.
  * A loop polls the feed price every second; when price touches the watch level the bot
    sends a MARKET order. Backtest of exactly this ("touch-market"): profitable up to about
    0.02-0.03% slippage per side, negative at 0.10% (reports/NIGHT_2026_09_11.md).
  * Right after the fill ONE reduce-only OCO goes on the exchange. Its stop and take are
    trigger orders that execute at MARKET - nothing sits in the order book - and they fire
    even while the bot is down. Both levels are measured from the real fill.
  * 48h after entry the bot market-closes whatever is still open.
  * Slippage of every entry is measured against the X-Perp best price seen just before the
    order. A coin whose average adverse slippage exceeds PULLBACK_LIVE_MAX_SLIP is excluded
    for 7 days; if the average over all coins exceeds it, new entries stop. Both are
    reported to the traders.
Everything lives in bot_state, so a restart or redeploy resumes watches and positions.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import numpy as np

from src import pullback_bank as PB
from src.execution_guard import execution_quote
from src.risk_limits import equity_guard

log = logging.getLogger(__name__)

STATE_KEY = "pullback_live"
WATCH_SEC = 900              # a signal may be entered during the first 15 minutes only
SCAN_DELAY = 10              # seconds after the hour before candles are read
SCAN_GIVE_UP = 300           # symbols whose closing candle is still missing are skipped
MANAGE_EVERY = 15            # seconds between position checks (private API budget)
HOLD_SEC = PB.HOLD_BARS * PB.BAR
EXCLUDE_SEC = 7 * 86400
KEEP_15M = 6000
FILL_READ_TRIES = 5


# ---------------------------------------------------------------- candles
def _as_arrays(d):
    t = np.asarray(d["time"], dtype=np.int64)
    vol = d.get("volume")
    a = np.c_[d["open"], d["high"], d["low"], d["close"],
              vol if vol is not None else np.zeros(len(t))].astype(float)
    return t, a


class CandleStore:
    """Closed 15m candles per symbol: one deep load, then a short refresh per hour."""

    def __init__(self, fetch_deep, fetch_recent):
        self.fetch_deep = fetch_deep        # sym -> dict(time, open, high, low, close[, volume])
        self.fetch_recent = fetch_recent    # sym -> same shape, the latest ~100 candles
        self.data = {}

    def get(self, sym, now):
        if sym not in self.data:
            t, a = _as_arrays(self.fetch_deep(sym))
        else:
            t, a = self.data[sym]
            rt, ra = _as_arrays(self.fetch_recent(sym))
            rows = {int(x): r for x, r in zip(t, a)}
            rows.update({int(x): r for x, r in zip(rt, ra)})
            ks = sorted(rows)
            t = np.asarray(ks, dtype=np.int64)
            a = np.asarray([rows[k] for k in ks], dtype=float).reshape(-1, 5)
        t, a = t[-KEEP_15M:], a[-KEEP_15M:]
        self.data[sym] = (t, a)
        keep = t + PB.BAR <= now
        return t[keep], a[keep]


# ---------------------------------------------------------------- engine
class Live:
    """users() -> [{"user_id", "creds", "margin": fn(balance)->$ margin per trade}]."""

    def __init__(self, *, symbols, rules_name, ex, inst_id_of, candles, feed_prices, users,
                 dm, load_state, save_state, max_spread, max_slip, slip_min_n, leverage,
                 max_daily_loss, max_drawdown, peak_floor=0.0, sleep=time.sleep):
        self.symbols = list(symbols)
        self.rules = PB.RULE_SETS[rules_name]
        self.ex, self.inst_id_of, self.candles = ex, inst_id_of, candles
        self.feed_prices, self.users, self.dm = feed_prices, users, dm
        self._save = save_state
        self.max_spread, self.max_slip, self.slip_min_n = max_spread, max_slip, slip_min_n
        self.leverage = leverage
        self.max_daily_loss, self.max_drawdown, self.peak_floor = max_daily_loss, max_drawdown, peak_floor
        self.sleep = sleep
        st = load_state() or {}
        for k, v in (("regime", {}), ("watch", []), ("pos", []), ("slip", {}), ("slip_all", []), ("hist", []),
                     ("excluded", {}), ("guard", {}), ("scan", {}), ("paused", False)):
            st.setdefault(k, v)
        self.state = st

    def save(self):
        try:
            self._save(self.state)
        except Exception as e:                      # state loss must be loud
            log.error("pullback live: state save failed: %s", e)

    def _tell_all(self, text):
        for u in self.users():
            try:
                self.dm(u["user_id"], text)
            except Exception as e:
                log.warning("pullback live DM failed: %s", e)

    # ------------------------------------------------------------ hourly signals
    def maybe_scan(self, now):
        close = int(now) - int(now) % PB.HOUR
        if now - close < SCAN_DELAY:
            return
        sc = self.state["scan"]
        if sc.get("close") != close:
            sc.clear(); sc.update(close=close, done=[])
        todo = [s for s in self.symbols if s not in sc["done"]]
        if not todo:
            return
        if now - close > SCAN_GIVE_UP:
            log.warning("pullback live: no closing candle for %s at %s", todo, close)
            sc["done"] = list(self.symbols); self.save()
            return
        try:
            bt, ba = self.candles.get("BTCUSDT", now)
        except Exception as e:
            log.warning("pullback live: BTC candles failed: %s", e)
            return
        if not len(bt) or int(bt[-1]) + PB.BAR != close:
            return                                      # BTC closing candle not published yet
        bd, bc = PB.btc_daily(bt, ba)
        for sym in todo:
            try:
                t, a = self.candles.get(sym, now)
                if not len(t) or int(t[-1]) + PB.BAR != close:
                    continue                            # retry on the next loop pass
                sc["done"].append(sym)
                if len(t) < 4 * (PB.MIN_HOURS + 50):
                    continue
                reg = self.state["regime"].setdefault(sym, {})
                horizon = int(t[0]) - 30 * 86400
                iv = [x for x in PB.regime_update(reg, t, a, bd, bc) if x[1] > horizon]
                reg["iv"] = [x for x in reg["iv"] if x[1] > horizon]
                s = PB.hourly_signals(t, a, bt, ba, self.rules, regime=iv).get(close)
                if s is None or self._busy(sym):
                    continue
                ex_ts = self.state["excluded"].get(sym)
                if ex_ts and now - ex_ts < EXCLUDE_SEC:
                    log.info("pullback live: %s excluded (slippage), signal skipped", sym)
                    continue
                k, level, atr = s
                self.state["watch"].append(dict(sym=sym, close=close, until=close + WATCH_SEC,
                                                rule=k, level=level, atr=atr,
                                                side=self.rules[k].get("side", "LONG")))
                log.info("pullback live: watch %s %s level %s", sym, self.rules[k]["name"], level)
            except Exception as e:
                log.warning("pullback live: scan %s failed: %s", sym, e)
        self.save()

    def _busy(self, sym):
        return any(w["sym"] == sym for w in self.state["watch"]) or \
            any(p["sym"] == sym for p in self.state["pos"])

    # ------------------------------------------------------------ every second
    def tick(self, now, prices):
        changed = False
        for w in list(self.state["watch"]):
            if now >= w["until"]:
                self.state["watch"].remove(w); changed = True
                continue
            p = prices.get(w["sym"])
            if p is None:
                continue
            if (p <= w["level"]) if w["side"] == "LONG" else (p >= w["level"]):
                self.state["watch"].remove(w); changed = True
                self.save()                             # never enter twice after a crash
                if self.state["paused"]:
                    log.warning("pullback live: paused by slippage guard, %s skipped", w["sym"])
                    continue
                for u in self.users():
                    try:
                        self._enter(u, w, now)
                    except Exception as e:
                        log.error("pullback live: entry %s for %s crashed: %s", w["sym"], u["user_id"], e)
        if changed:
            self.save()

    def _enter(self, u, w, now):
        uid, creds, ex = u["user_id"], u["creds"], self.ex
        sym, direction = w["sym"], w["side"]
        lg = direction == "LONG"
        r = self.rules[w["rule"]]
        inst = self.inst_id_of(sym)
        if not inst:
            return
        if any(p["uid"] == uid and p["sym"] == sym for p in self.state["pos"]):
            return
        ok, held = ex.get_position_size(creds, inst)
        if not ok or held != 0:
            log.warning("pullback live: %s %s not flat on the exchange (%s) - skipped", uid, inst, held)
            return
        ok, bal = ex.get_balance(creds)
        if not ok:
            log.warning("pullback live: balance failed for %s: %s", uid, bal)
            return
        g = self.state["guard"].get(str(uid), {})
        was = g.get("last_block")
        g, blocked = equity_guard(g, equity=bal, day=datetime.now(timezone.utc).date().isoformat(),
                                  max_daily_loss=self.max_daily_loss, max_drawdown=self.max_drawdown,
                                  peak_floor=self.peak_floor or None)
        g["last_block"] = blocked
        self.state["guard"][str(uid)] = g
        if blocked:
            if blocked != was:
                self.dm(uid, f"⏸ Банк откатов: новые входы остановлены ({blocked}). "
                             f"Открытые позиции сопровождаются.")
            return
        margin = float(u["margin"](bal))            # the user's own size setting
        if margin <= 0:
            return
        if margin > bal:
            log.info("pullback live: %s margin %.2f above balance %.2f - skipped", uid, margin, bal)
            return
        spec = ex.get_xperp_spec(inst)
        px = ex.get_last_price(inst)
        if not spec or not px:
            return
        sz = ex.calc_contracts(margin, self.leverage, px, spec)
        if sz <= 0:
            log.info("pullback live: %s margin %.2f below the exchange minimum for %s", uid, margin, sym)
            return
        try:
            quote = execution_quote(ex.get_order_book(inst), direction, sz, time.time() * 1000,
                                    max_spread=self.max_spread,
                                    max_slippage=max(2 * self.max_slip, 0.0005))
        except (TypeError, ValueError, KeyError) as e:
            log.info("pullback live: %s execution refused: %s", sym, e)
            return
        ok, lerr = ex.ensure_leverage(creds, inst, self.leverage)
        if not ok:
            log.warning("pullback live: leverage failed %s %s: %s", uid, inst, lerr)
            return
        ref = quote.best
        ok, oid = ex.place_market_entry(creds, inst, direction, sz)
        if not ok:
            self.dm(uid, f"❌ Банк откатов: не смог открыть {sym} {direction}.\n`{oid}`")
            return
        fill = None
        for _ in range(FILL_READ_TRIES):
            fill = ex.get_position_avg_px(creds, inst)
            if fill:
                break
            self.sleep(0.4)
        ok, pos_sz = ex.get_position_size(creds, inst)
        pos_sz = pos_sz if ok and pos_sz and pos_sz > 0 else sz
        fill = fill or ref
        slip = (fill / ref - 1) if lg else (1 - fill / ref)
        tp_frac = r["tp"] * w["atr"] / w["level"]
        sl_frac = r["sl"] * w["atr"] / w["level"]
        tick = spec.get("tickSz", 0)
        tp_px = ex.round_to_tick(fill * (1 + tp_frac) if lg else fill * (1 - tp_frac), tick)
        sl_px = ex.round_to_tick(fill * (1 - sl_frac) if lg else fill * (1 + sl_frac), tick)
        pos = dict(uid=uid, sym=sym, inst=inst, side=direction, rule=r["name"], sz=pos_sz,
                   entry=fill, ref=ref, sl=sl_px, tp=tp_px, algo="", open_ts=now,
                   deadline=now + HOLD_SEC, risk_frac=sl_frac, slip=slip, margin=margin,
                   next_check=now + MANAGE_EVERY)
        ok, algo = ex.place_protection_oco(creds, inst, direction, sl_px, tp_px)
        if not ok:
            closed, cerr = ex.close_position_market(creds, inst)
            if not closed:
                closed, cerr = ex.close_position_market(creds, inst)
            if closed:
                self.dm(uid, f"❌ Банк откатов: не смог поставить стоп по {sym} — позиция закрыта "
                             f"для безопасности.\n`{algo}`")
            else:
                log.error("pullback live NAKED POSITION %s %s: %s / %s", uid, inst, algo, cerr)
                self.dm(uid, f"🚨 *СРОЧНО: {sym} открыта БЕЗ СТОПА*\nНе удалось ни поставить защиту, "
                             f"ни закрыть позицию. Закрой её на бирже вручную.\n`{algo}`\n`{cerr}`")
                pos["deadline"] = now                   # the manager keeps trying to close it
                self.state["pos"].append(pos)
                self.save()
            return
        pos["algo"] = algo
        self.state["pos"].append(pos)
        self.save()
        self.dm(uid, f"🤖 Банк откатов: вход {sym} {direction} [{r['name']}]\n"
                     f"Цена: {ex.fmt_px_display(fill, tick)} (проскальзывание {100 * slip:+.3f}%)\n"
                     f"Стоп: {ex.fmt_px_display(sl_px, tick)}  Тейк: {ex.fmt_px_display(tp_px, tick)}\n"
                     f"Маржа ${margin:.2f} x{self.leverage}, выход не позже чем через 48ч")
        self._record_slip(sym, slip, now)

    def _record_slip(self, sym, slip, now):
        vals = self.state["slip"].setdefault(sym, [])
        vals.append(float(slip)); del vals[:-50]
        if len(vals) >= self.slip_min_n and np.mean(vals) > self.max_slip:
            self.state["excluded"][sym] = now
            self.state["slip"][sym] = []
            self._tell_all(f"⚠️ Банк откатов: {sym} исключена на 7 дней — среднее проскальзывание "
                           f"{100 * np.mean(vals):.3f}% больше допустимого {100 * self.max_slip:.3f}%.")
        # Portfolio guard keeps its own history: excluding a coin resets that coin's list,
        # but its bad fills must still count towards the average over all coins.
        allv = self.state["slip_all"]
        allv.append(float(slip)); del allv[:-100]
        if len(allv) >= 3 * self.slip_min_n and np.mean(allv) > self.max_slip and not self.state["paused"]:
            self.state["paused"] = True
            self._tell_all(f"⏸ Банк откатов: новые входы остановлены — общее среднее проскальзывание "
                           f"{100 * np.mean(allv):.3f}% больше допустимого {100 * self.max_slip:.3f}%. "
                           f"Открытые позиции сопровождаются.")
        self.save()

    # ------------------------------------------------------------ open positions
    def manage(self, now):
        if not self.state["pos"]:
            return
        creds = {u["user_id"]: u["creds"] for u in self.users()}
        for p in list(self.state["pos"]):
            if now < p["next_check"]:
                continue
            p["next_check"] = now + MANAGE_EVERY
            c = creds.get(p["uid"])
            if c is None:
                continue
            lg = p["side"] == "LONG"
            ok, size = self.ex.get_position_size(c, p["inst"])
            if not ok:
                continue
            if size == 0:
                if now - p["open_ts"] < 60:
                    continue                            # a fresh fill can read as flat for a moment
                px = self.ex.get_last_fill_px(c, p["inst"], since_ts=p["open_ts"],
                                              side="sell" if lg else "buy")
                self._finish(p, px, "стоп/тейк", now)
            elif now >= p["deadline"]:
                closed, err = self.ex.close_position_market(c, p["inst"])
                if closed:
                    if p.get("algo"):
                        self.ex.cancel_protection(c, p["inst"], p["algo"])
                    px = self.ex.get_last_fill_px(c, p["inst"], since_ts=p["open_ts"],
                                                  side="sell" if lg else "buy")
                    self._finish(p, px, "48 часов" if p.get("algo") else "аварийное закрытие", now)
                else:
                    log.warning("pullback live: time exit %s failed: %s", p["inst"], err)
        self.save()

    def _finish(self, p, exit_px, reason, now):
        lg = p["side"] == "LONG"
        ret = None
        if exit_px:
            ret = (exit_px / p["entry"] - 1) if lg else (1 - exit_px / p["entry"])
        self.state["pos"].remove(p)
        self.state["hist"].append(dict(sym=p["sym"], side=p["side"], rule=p["rule"], uid=p["uid"],
                                       open_ts=p["open_ts"], close_ts=now, entry=p["entry"],
                                       exit=exit_px, ret=ret, slip=p.get("slip"), reason=reason))
        del self.state["hist"][:-500]
        self.save()
        if ret is None:
            self.dm(p["uid"], f"ℹ️ Банк откатов: {p['sym']} закрыта ({reason}), биржа не отдала цену выхода.")
        else:
            usd = ret * p.get("margin", 0) * self.leverage
            self.dm(p["uid"], f"{'✅' if ret > 0 else '🔴'} Банк откатов: {p['sym']} {p['side']} закрыта "
                              f"({reason}): {100 * ret:+.2f}% цены, ≈ ${usd:+.2f} до комиссии")

    # ------------------------------------------------------------ loop
    def step(self, now):
        try:
            self.maybe_scan(now)
        except Exception as e:
            log.error("pullback live: scan crashed: %s", e)
        if self.state["watch"]:
            try:
                self.tick(now, self.feed_prices())
            except Exception as e:
                log.warning("pullback live: price tick failed: %s", e)
        try:
            self.manage(now)
        except Exception as e:
            log.error("pullback live: manage crashed: %s", e)


def run_forever(live, stop=None):
    while stop is None or not stop.is_set():
        live.step(time.time())
        time.sleep(1)


# ---------------------------------------------------------------- production wiring
def _feed_prices():
    import requests
    from backtest import OKX_HOSTS
    for host in OKX_HOSTS:
        try:
            r = requests.get(host + "/api/v5/market/tickers", params={"instType": "SWAP"}, timeout=3)
            out = {}
            for x in r.json().get("data", []):
                inst = str(x.get("instId", ""))
                if inst.endswith("-USDT-SWAP") and x.get("last"):
                    out[inst.split("-")[0] + "USDT"] = float(x["last"])
            if out:
                return out
        except Exception:
            continue
    return {}


def _fetch_recent(sym):
    from backtest import _okx_get_bt, _inst_id
    raw = _okx_get_bt("/api/v5/market/candles",
                      {"instId": _inst_id(sym), "bar": "15m", "limit": "100"}).json().get("data", [])
    rows = sorted((c for c in raw if len(c) <= 8 or c[8] == "1"), key=lambda c: int(c[0]))
    return dict(time=[int(c[0]) // 1000 for c in rows], open=[float(c[1]) for c in rows],
                high=[float(c[2]) for c in rows], low=[float(c[3]) for c in rows],
                close=[float(c[4]) for c in rows], volume=[float(c[6]) for c in rows])


def build_default():
    import config as C
    from backtest import fetch_history
    from src import okx_trader
    from src.autotrader import _creds_of, _dm, _inst_id_of, _margin_for
    from src.db import at_get_active_traders, get_bot_state, set_bot_state

    cache = {"at": 0.0, "users": []}

    def users():
        if time.time() - cache["at"] > 300:
            out = []
            for u in at_get_active_traders():
                c = _creds_of(u)
                if c:
                    out.append({"user_id": int(u["user_id"]), "creds": c,
                                "margin": (lambda bal, _u=dict(u): _margin_for(_u, bal))})
            cache.update(at=time.time(), users=out)
        return cache["users"]

    def load_state():
        raw = get_bot_state(STATE_KEY)
        return json.loads(raw) if raw else {}

    def save_state(st):
        set_bot_state(STATE_KEY, json.dumps(st))

    candles = CandleStore(
        lambda s: fetch_history(s, "15min", 900, C.PULLBACK_FETCH_15M, refresh_cache=True),
        _fetch_recent)
    return Live(symbols=C.PULLBACK_SYMBOLS, rules_name=C.PULLBACK_RULES, ex=okx_trader,
                inst_id_of=_inst_id_of, candles=candles, feed_prices=_feed_prices, users=users,
                dm=_dm, load_state=load_state, save_state=save_state,
                max_spread=C.PULLBACK_LIVE_MAX_SPREAD, max_slip=C.PULLBACK_LIVE_MAX_SLIP,
                slip_min_n=C.PULLBACK_LIVE_SLIP_MIN_N, leverage=C.AUTOTRADE_LEVERAGE,
                max_daily_loss=C.PULLBACK_LIVE_MAX_DAILY_LOSS,
                max_drawdown=C.PULLBACK_LIVE_MAX_DRAWDOWN)


def start_default():
    try:
        live = build_default()
        now = time.time()
        for s in ["BTCUSDT"] + list(live.symbols):     # deep load once, before the first scan
            try:
                live.candles.get(s, now)
            except Exception as e:
                log.warning("pullback live: warm-up %s failed: %s", s, e)
    except Exception as e:
        log.error("pullback live: start failed: %s", e)
        return
    log.info("pullback live: started, %d symbols, %d rules", len(live.symbols), len(live.rules))
    run_forever(live)
