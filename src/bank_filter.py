"""Judge each bank signal before it becomes a trade.

The bank decides WHERE to trade; this module decides WHETHER this particular signal is worth
taking. It rebuilds the same 140 numbers the research used - market structure on 1h, 4h and 1d
(swings, HH/HL/LH/LL, breaks, zones, order blocks, imbalances, stop runs, candle geometry) plus
indicators across the scale, sessions, volume profile and the link to BTC - and runs the model
trained on the bank's own history.

Measured on 2025-26, which the model never saw (reports/NIGHT_2026_09_11.md):
  bank alone      2.0 trades/day, 84.7% win rate, +0.056R per trade, drawdown -11.1R
  bank + filter   1.2 trades/day, 89.2% win rate, +0.110R per trade, drawdown  -5.0R

Everything is read from CLOSED bars only; the audit that proves it is tests/test_bank_filter.py.
Off unless PULLBACK_FILTER_ENABLED is set: with no model file or on any error the signal is
taken, so the filter can never block trading by failing.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

from src import features_extra as FX
from src import features_struct as FS

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'models', 'bank_filter.joblib')
SPEC_PATH = MODEL_PATH.replace('.joblib', '.json')
_cache = {"model": None, "spec": None, "tried": False}


def _load():
    if _cache["tried"]:
        return _cache["model"], _cache["spec"]
    _cache["tried"] = True
    try:
        import joblib
        _cache["model"] = joblib.load(MODEL_PATH)
        with open(SPEC_PATH, encoding="utf-8") as f:
            _cache["spec"] = json.load(f)
        log.info("bank filter: model loaded, %d features, threshold %.4f",
                 len(_cache["spec"]["features"]), _cache["spec"]["threshold"])
    except Exception as e:                      # no model, no numpy build, broken file - trade on
        log.warning("bank filter: model not loaded (%s); every signal will be taken", e)
    return _cache["model"], _cache["spec"]


def build_bars(t15, a15, sec):
    """Aggregate 15m candles into complete `sec` bars (an unfinished bar is dropped)."""
    need = sec // 900
    key = (t15 // sec) * sec
    out_t, rows = [], []
    i = 0
    n = len(t15)
    while i < n:
        j = i
        while j < n and key[j] == key[i]:
            j += 1
        if j - i == need and t15[j - 1] - t15[i] == (need - 1) * 900:
            block = a15[i:j]
            rows.append([block[0, 0], block[:, 1].max(), block[:, 2].min(), block[-1, 3],
                         block[:, 4].sum()])
            out_t.append(key[i])
        i = j
    return np.asarray(out_t, dtype=np.int64), np.asarray(rows, dtype=float).reshape(-1, 5)


def atr(A, n=14):
    h, l, c = A[:, 1], A[:, 2], A[:, 3]
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = np.full(len(c), np.nan)
    cs = np.cumsum(np.r_[0, tr])
    out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def features(t15, a15, btc_t15, btc_a15):
    """The 140 numbers for the LAST fully closed hour, in the order the model expects."""
    t1, A1 = build_bars(t15, a15, 3600)
    if len(t1) < 300:
        return None, None
    atr1 = atr(A1, 14)
    cols, names = [], []
    base = FS.structure(t1, *A1.T[:5], atr1)
    for k in FS.NAMES:
        cols.append(base[k]); names.append(f'h1_{k}')
    for sec, tag in ((14400, 'h4'), (86400, 'd1')):
        t, A = build_bars(t15, a15, sec)
        if len(t) < 60:
            for k in FS.NAMES:
                cols.append(np.full(len(t1), np.nan)); names.append(f'{tag}_{k}')
            continue
        st = FS.structure(t, *A.T[:5], atr(A, 14))
        for k in FS.NAMES:
            cols.append(FX.map_htf(t1, t, st[k], sec)); names.append(f'{tag}_{k}')
    btc_t1, btc_A1 = build_bars(btc_t15, btc_a15, 3600)
    ex = FX.extras('', t1, A1, atr1, btc_t1, btc_A1)
    for k in FX.NAMES:
        cols.append(ex[k]); names.append(f'x_{k}')
    X = np.vstack(cols).T.astype(np.float32)
    return X[-1], names


def score(t15, a15, btc_t15, btc_a15):
    """Model score for the signal of the last closed hour, or None when unavailable."""
    model, spec = _load()
    if model is None:
        return None
    try:
        row, names = features(t15, a15, btc_t15, btc_a15)
        if row is None:
            return None
        want = spec["features"]
        if names != want:                        # order must match the training set exactly
            idx = {n: i for i, n in enumerate(names)}
            row = np.array([row[idx[n]] if n in idx else np.nan for n in want], dtype=np.float32)
        return float(model.predict(row.reshape(1, -1))[0])
    except Exception as e:
        log.warning("bank filter: scoring failed (%s); signal taken", e)
        return None


def allow(t15, a15, btc_t15, btc_a15):
    """(take?, score). Anything unavailable means take the signal - never block on a failure."""
    s = score(t15, a15, btc_t15, btc_a15)
    if s is None:
        return True, None
    _, spec = _load()
    return s >= float(spec["threshold"]), s
