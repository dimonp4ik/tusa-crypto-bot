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

from src.crypto_venue_router import ROBUST_SYMBOLS, btc_context, family_signals
from src.portable_hgb import score_named


ARTIFACT_PATH = Path(__file__).with_name(
    "frozen_crypto_profit_first_material_2r.json")
MODEL = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
MODULE = "profit_first_material_2r"


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
                  symbol: str | None = None) -> list[dict]:
    """Return at most one frozen-model setup, permanently marked shadow-only."""
    if (not symbol or symbol not in ROBUST_SYMBOLS
            or not math.isfinite(entry_time)):
        return []
    context = btc_context(btc_candles)
    if context is None:
        return []
    entry = float(market_price if market_price is not None
                  else candles["close"][-1])
    if not math.isfinite(entry) or entry <= 0:
        return []

    for signal in family_signals(candles):
        risk = float(MODEL["stop_atr"]) * float(signal["atr"])
        risk_pct = risk / entry
        if (not math.isfinite(risk_pct)
                or risk_pct < float(MODEL["risk_pct_min"])
                or risk_pct > float(MODEL["risk_pct_max"])):
            continue
        probability = model_probability(
            symbol=symbol, signal=signal, context=context,
            entry_time=entry_time, risk_pct=risk_pct)
        if probability < float(MODEL["quality_threshold"]):
            continue
        sign = 1.0 if signal["direction"] == "LONG" else -1.0
        near_r = float(MODEL["near_target_r"])
        runner_r = float(MODEL["runner_target_r"])
        stop = entry - sign * risk
        near_target = entry + sign * risk * near_r
        if risk >= entry or min(stop, near_target) <= 0:
            continue
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
            "hold_bars": int(MODEL["hold_bars"]),
            "model_quality": probability,
            "risk_pct": risk_pct,
            "first_target_move_pct": 100.0 * risk_pct * near_r,
            "runner_target_move_pct": 100.0 * risk_pct * runner_r,
            "utc_session": _session(entry_time),
            "_shadow_experiment": True,
            "_profit_first_material_shadow": True,
        }]
    return []
