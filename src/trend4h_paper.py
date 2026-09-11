"""Paper runner for the 4h breakout strategy. Never places orders.

Once per closed 4h bar: fetch closed 15m candles for each symbol and BTC daily
closes, feed any new closed 4h bars through trend4h.advance() (the exact code
the backtest replays), persist state to a JSON file, and report OPEN / STOP /
EXIT events. A failure on one symbol is logged and skipped; state only moves
forward for symbols that were processed.
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

from src import trend4h as M

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


def run_once(symbols, fetch_15m, fetch_btc_daily, state_path, notify, *, now=None):
    """fetch_15m(symbol) -> dict(time, open, high, low, close) of CLOSED 15m
    candles, seconds, oldest first. fetch_btc_daily() -> (times, closes) of
    CLOSED daily bars. notify(text) delivers a message. Returns event count."""
    now = time.time() if now is None else now
    state = _load_state(state_path)
    try:
        bt, bc = fetch_btc_daily()
        bt = [int(t) for t in bt]; bc = np.asarray(bc, dtype=float)
    except Exception as e:                       # no BTC gate -> no entries at all
        log.warning("trend4h paper: BTC daily fetch failed: %s", e)
        return 0
    lines, n = [], 0
    for sym in symbols:
        try:
            c = fetch_15m(sym)
            T, B = M.build_4h(c["time"], c["open"], c["high"], c["low"], c["close"])
            T = T[T + M.BAR_SEC <= now]           # only bars that have fully closed
            B = B[:len(T)]
            if len(T) < 80:
                continue
            st = state.setdefault(sym, {})
            if st.get("last") is None:
                # First run: warm up silently on history so we do not report
                # stale entries as new; then say what is open right now.
                M.advance(st, T[:-1], B[:-1], bt, bc)
                pos = st.get("pos")
                if pos:
                    lines.append(f"📌 {sym} уже в позиции {pos['side']} с "
                                 f"{time.strftime('%d.%m %H:%M', time.gmtime(pos['open_ts']))} UTC, "
                                 f"стоп {pos['stop']:.6g}")
            for e in M.advance(st, T, B, bt, bc):
                n += 1
                if e[0] == "OPEN":
                    lines.append(f"🟢 {sym} {e[2]} вход ~{e[3]:.6g}, стоп {e[4]:.6g}")
                elif e[0] == "EXIT":
                    lines.append(f"🔴 {sym} {e[2]} выход {e[3]:+.2f}R")
                elif e[0] == "STOP":
                    lines.append(f"↕️ {sym} стоп → {e[2]:.6g}")
        except Exception as e:
            log.warning("trend4h paper: %s failed: %s", sym, e)
    _save_state(state_path, state)
    if lines:
        notify("📄 4ч тренд (бумага, без ордеров)\n" + "\n".join(lines))
    return n
