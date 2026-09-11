"""Paper runner for the high win-rate pullback bank (src/pullback_bank.py). Never places orders.

Once per hour: fetch closed 15m candles for each symbol and for BTC, rebuild the hourly
signals from closed bars only, and feed every unseen 15m bar through
pullback_bank.advance() - the exact code the backtest replays. State (pending limit order,
open position, last processed bar) lives in a JSON file. Reports OPEN / EXIT events.
A failure on one symbol is logged and skipped; a BTC failure skips the whole run (every
rule and the trend regime need BTC).
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

from src import pullback_bank as M

log = logging.getLogger(__name__)


def _load_state(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def _closed(c, now):
    t = np.asarray(c["time"], dtype=np.int64)
    vol = c.get("volume")
    a = np.c_[c["open"], c["high"], c["low"], c["close"],
              vol if vol is not None else np.zeros(len(t))].astype(float)
    keep = t + M.BAR <= now                      # only fully closed 15m bars
    return t[keep], a[keep]


def run_once(symbols, fetch_15m, state_path, notify, *, rules_name="bank9", now=None):
    """fetch_15m(symbol) -> dict(time, open, high, low, close[, volume]) of 15m candles,
    seconds, oldest first. notify(text) delivers a message. Returns event count."""
    now = time.time() if now is None else now
    rules = M.RULE_SETS[rules_name]
    state = _load_state(state_path)
    try:
        bt, ba = _closed(fetch_15m("BTCUSDT"), now)
    except Exception as e:
        log.warning("pullback paper: BTC fetch failed: %s", e)
        return 0
    lines, n = [], 0
    for sym in symbols:
        try:
            t, a = _closed(fetch_15m(sym), now)
            if len(t) < 4 * (M.MIN_HOURS + 50):
                continue
            st = state.setdefault(sym, {})
            # The trend regime is persisted and only moves forward, so a trend position
            # opened before the fetched window is not forgotten.
            bd, bc = M.btc_daily(bt, ba)
            reg = st.setdefault("regime", {})
            horizon = int(t[0]) - 30 * 86400
            iv = [x for x in M.regime_update(reg, t, a, bd, bc) if x[1] > horizon]
            reg["iv"] = [x for x in reg["iv"] if x[1] > horizon]
            sig = M.hourly_signals(t, a, bt, ba, rules, regime=iv)
            if st.get("last") is None:
                # First run: warm up silently so history is not reported as new.
                M.advance(st, t[:-1], a[:-1], sig, rules)
                pos = st.get("pos")
                if pos:
                    lines.append(f"📌 {sym} уже в позиции ({rules[pos['rule']]['name']}) с "
                                 f"{time.strftime('%d.%m %H:%M', time.gmtime(pos['open_ts']))} UTC")
            for e in M.advance(st, t, a, sig, rules):
                if e[0] == "OPEN":
                    n += 1
                    lines.append(f"🟢 {sym} {e[7]} [{e[2]}] вход {e[3]:.6g} ({e[6]}), "
                                 f"тейк {e[4]:.6g}, стоп {e[5]:.6g}")
                elif e[0] == "EXIT":
                    n += 1
                    lines.append(f"{'✅' if e[4] > 0 else '🔴'} {sym} [{e[2]}] {e[3]}: "
                                 f"{100 * e[4]:+.2f}% ({e[5]:+.2f}R)")
        except Exception as e:
            log.warning("pullback paper: %s failed: %s", sym, e)
    _save_state(state_path, state)
    if lines:
        notify("📄 Банк откатов (бумага, без ордеров)\n" + "\n".join(lines))
    return n
