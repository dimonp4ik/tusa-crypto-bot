"""Frozen 15-minute Profit-First X-Perp model for paper signals only.

The model proposes market entries and fixed bracket geometry.  It never
chooses a client's position size and it cannot place an exchange order.
Callers must pass candles whose last bar is closed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from src.crypto_venue_router import (
    BTC_CONTEXT_HISTORY_BARS, FAMILY_SIGNAL_HISTORY_BARS, ROBUST_SYMBOLS,
    btc_context, family_signals,
)
from src.portable_hgb import score_named
from src.tp1_volume_bank import (
    EXTRA_CLOSE_FRACTION as TP1_VOLUME_EXTRA_CLOSE_FRACTION,
    VOLUME_RATIO_THRESHOLD as TP1_VOLUME_RATIO_THRESHOLD,
)


ARTIFACT_PATH = Path(__file__).with_name(
    "frozen_crypto_profit_first_material_2r.json")
MODEL = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
MODULE = "profit_first_material_2r"
REQUIRED_BTC_HISTORY_BARS = BTC_CONTEXT_HISTORY_BARS
REQUIRED_SYMBOL_HISTORY_BARS = FAMILY_SIGNAL_HISTORY_BARS


def _increment(values: dict, key: str) -> None:
    values[key] = int(values.get(key, 0)) + 1


def _session(timestamp: float) -> str:
    hour = datetime.fromtimestamp(timestamp, timezone.utc).hour
    if hour <= 5:
        return "00_05"
    if hour <= 11:
        return "06_11"
    if hour <= 17:
        return "12_17"
    return "18_23"


def feature_values(*, symbol: str, signal: dict, context: dict,
                   entry_time: float, risk_pct: float) -> dict[str, float]:
    """Build the exact one-row design matrix used by the frozen research."""
    direction = str(signal["direction"])
    sign = 1.0 if direction == "LONG" else -1.0
    hour = datetime.fromtimestamp(entry_time, timezone.utc)
    fractional_hour = hour.hour + hour.minute / 60.0
    values = {
        "risk_pct": float(risk_pct),
        "eff_ratio": float(signal["eff_ratio"]),
        "dir_z": float(signal["z"]) * sign,
        "abs_z": abs(float(signal["z"])),
        "hour_sin": math.sin(2.0 * math.pi * fractional_hour / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * fractional_hour / 24.0),
        "btc_fast_slow_atr": float(context["btc_fast_slow_atr"]),
        "btc_eff20": float(context["btc_eff20"]),
        "btc_return20_atr": float(context["btc_return20_atr"]),
        "btc_z20": float(context["btc_z20"]),
        "btc_atr_pct": float(context["btc_atr_pct"]),
        "btc_slow_long_atr": float(context["btc_slow_long_atr"]),
        "btc_return96_atr": float(context["btc_return96_atr"]),
        "btc_eff96": float(context["btc_eff96"]),
        "btc_trend_dir_atr": float(context["btc_fast_slow_atr"]) * sign,
        "btc_return_dir_atr": float(context["btc_return20_atr"]) * sign,
    }
    categories = {
        "symbol": symbol,
        "direction": direction,
        "family": str(signal["family"]),
        "utc_session": _session(entry_time),
        "btc_regime": str(context["btc_regime"]),
    }
    for field, known in MODEL["categories"].items():
        selected = categories[field]
        for category in known:
            values[f"{field}={category}"] = float(selected == category)
    return values


def model_probability(*, symbol: str, signal: dict, context: dict,
                      entry_time: float, risk_pct: float) -> float:
    return score_named(MODEL, feature_values(
        symbol=symbol, signal=signal, context=context,
        entry_time=entry_time, risk_pct=risk_pct))


def shadow_setups(candles: dict, btc_candles: dict, *, entry_time: float,
                  market_price: float | None = None,
                  symbol: str | None = None,
                  diagnostics: dict | None = None) -> list[dict]:
    """Return at most one frozen-model setup, permanently marked shadow-only."""
    audit = diagnostics if diagnostics is not None else {}
    audit.clear()
    symbol_bars = len(candles.get("close") or [])
    btc_bars = len(btc_candles.get("close") or [])
    audit.update({
        "symbol": symbol,
        "symbol_history_bars": symbol_bars,
        "btc_history_bars": btc_bars,
        "required_symbol_history_bars": REQUIRED_SYMBOL_HISTORY_BARS,
        "required_btc_history_bars": REQUIRED_BTC_HISTORY_BARS,
        "family_signals": 0,
        "rejected": {},
    })
    if not symbol or symbol not in ROBUST_SYMBOLS:
        audit["result"] = "unsupported_symbol"
        return []
    if not math.isfinite(entry_time):
        audit["result"] = "invalid_entry_time"
        return []
    if symbol_bars < REQUIRED_SYMBOL_HISTORY_BARS:
        audit["result"] = "insufficient_symbol_history"
        return []
    if btc_bars < REQUIRED_BTC_HISTORY_BARS:
        audit["result"] = "insufficient_btc_history"
        return []
    context = btc_context(btc_candles)
    if context is None:
        audit["result"] = "invalid_btc_context"
        return []
    try:
        entry = float(market_price if market_price is not None
                      else candles["close"][-1])
    except (KeyError, IndexError, TypeError, ValueError):
        audit["result"] = "invalid_market_price"
        return []
    if not math.isfinite(entry) or entry <= 0:
        audit["result"] = "invalid_market_price"
        return []

    signals = family_signals(candles)
    audit["family_signals"] = len(signals)
    if not signals:
        audit["result"] = "no_family_signal"
        return []
    for signal in signals:
        risk = float(MODEL["stop_atr"]) * float(signal["atr"])
        risk_pct = risk / entry
        if (not math.isfinite(risk_pct)
                or risk_pct < float(MODEL["risk_pct_min"])
                or risk_pct > float(MODEL["risk_pct_max"])):
            _increment(audit["rejected"], "risk_outside_1_to_2pct")
            continue
        probability = model_probability(
            symbol=symbol, signal=signal, context=context,
            entry_time=entry_time, risk_pct=risk_pct)
        if probability < float(MODEL["quality_threshold"]):
            _increment(audit["rejected"], "model_quality_below_threshold")
            continue
        sign = 1.0 if signal["direction"] == "LONG" else -1.0
        near_r = float(MODEL["near_target_r"])
        runner_r = float(MODEL["runner_target_r"])
        stop = entry - sign * risk
        near_target = entry + sign * risk * near_r
        if risk >= entry or min(stop, near_target) <= 0:
            _increment(audit["rejected"], "invalid_geometry")
            continue
        audit["result"] = "candidate"
        audit["model_quality"] = probability
        return [{
            **signal,
            **context,
            "module": MODULE,
            "direction": signal["direction"],
            "entry_source": "MARKET",
            "entry": entry,
            "sl": stop,
            "tp": near_target,
            "target_r": near_r,
            "runner_target_r": runner_r,
            "tp1_close_fraction": float(MODEL["near_close_fraction"]),
            "tp1_volume_ratio_threshold": TP1_VOLUME_RATIO_THRESHOLD,
            "tp1_volume_extra_close_fraction": TP1_VOLUME_EXTRA_CLOSE_FRACTION,
            "hold_bars": int(MODEL["hold_bars"]),
            "model_quality": probability,
            "risk_pct": risk_pct,
            "first_target_move_pct": 100.0 * risk_pct * near_r,
            "runner_target_move_pct": 100.0 * risk_pct * runner_r,
            "utc_session": _session(entry_time),
            "_shadow_experiment": True,
            "_profit_first_material_shadow": True,
        }]
    audit["result"] = "all_family_signals_filtered"
    return []
