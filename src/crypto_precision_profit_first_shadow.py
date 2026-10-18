"""Frozen 15-minute profit-first precision model for shadow observation.

The module proposes a market-entry 0.25R/2R barbell bracket.  It never sizes
or places an order.  Its portable HGB and cutoff were frozen before the 2024+
and direct X-Perp reveals recorded in the source audit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from src.crypto_venue_router import (
    ROBUST_SYMBOLS, _session, btc_context, family_signals,
)
from src.portable_hgb import score_named


MODEL_PATH = Path(__file__).with_name(
    "frozen_crypto_precision_profit_first_hgb.json")
_MODEL = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
QUALITY_THRESHOLD = float(_MODEL["quality_threshold"])
STOP_ATR = float(_MODEL["stop_atr"])
NEAR_TARGET_R = float(_MODEL["near_target_r"])
RUNNER_TARGET_R = float(_MODEL["runner_target_r"])
NEAR_CLOSE_FRACTION = float(_MODEL["near_close_fraction"])
HOLD_BARS = int(_MODEL["hold_bars"])
LOSS_PAUSE = int(_MODEL["loss_pause"])
HOURLY_CAP = int(_MODEL["hourly_cap"])


def _feature_values(symbol: str, signal: dict, context: dict,
                    entry_time: float, entry: float) -> dict[str, float] | None:
    direction = str(signal["direction"])
    sign = 1.0 if direction == "LONG" else -1.0
    risk = STOP_ATR * float(signal["atr"])
    stamp = datetime.fromtimestamp(float(entry_time), timezone.utc)
    hour = stamp.hour + stamp.minute / 60.0
    raw = {
        "risk_pct": risk / entry,
        "eff_ratio": float(signal["eff_ratio"]),
        "dir_z": float(signal["z"]) * sign,
        "abs_z": abs(float(signal["z"])),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        **{name: float(context[name]) for name in (
            "btc_fast_slow_atr", "btc_eff20", "btc_return20_atr",
            "btc_z20", "btc_atr_pct", "btc_slow_long_atr",
            "btc_return96_atr", "btc_eff96")},
    }
    raw["btc_trend_dir_atr"] = raw["btc_fast_slow_atr"] * sign
    raw["btc_return_dir_atr"] = raw["btc_return20_atr"] * sign
    categories = {
        "symbol": symbol, "direction": direction,
        "family": str(signal["family"]),
        "utc_session": _session(entry_time),
        "btc_regime": str(context["btc_regime"]),
    }
    values = {}
    for name in _MODEL["features"]:
        if "=" in name:
            field, expected = name.split("=", 1)
            values[name] = float(categories.get(field) == expected)
        else:
            value = raw.get(name)
            if value is None or not math.isfinite(float(value)):
                return None
            values[name] = float(value)
    return values


def shadow_setups(candles: dict, btc_candles: dict, *, entry_time: float,
                  market_price: float | None = None,
                  symbol: str | None = None) -> list[dict]:
    """Return at most one exact frozen-model setup for the next market entry."""
    if not symbol or symbol not in ROBUST_SYMBOLS or not math.isfinite(entry_time):
        return []
    context = btc_context(btc_candles)
    if context is None:
        return []
    entry = float(market_price if market_price is not None
                  else candles.get("close", [0])[-1])
    if not math.isfinite(entry) or entry <= 0:
        return []
    # Keep the historical family order.  The portfolio gate admitted the first
    # same-symbol candidate at one timestamp rather than ranking on outcomes.
    for signal in family_signals(candles):
        values = _feature_values(symbol, signal, context, entry_time, entry)
        if values is None:
            continue
        quality = score_named(_MODEL, values)
        if quality < QUALITY_THRESHOLD:
            continue
        risk = STOP_ATR * float(signal["atr"])
        sign = 1.0 if signal["direction"] == "LONG" else -1.0
        stop = entry - sign * risk
        near = entry + sign * risk * NEAR_TARGET_R
        runner = entry + sign * risk * RUNNER_TARGET_R
        if risk >= entry or min(stop, near, runner) <= 0:
            continue
        return [{
            **signal, **context,
            "module": "shadow_precision_profit_first_025_2r",
            "direction": signal["direction"],
            "entry_source": "MARKET", "entry": entry,
            "sl": stop, "tp": near, "target_r": NEAR_TARGET_R,
            "runner_target_r": RUNNER_TARGET_R,
            "near_close_fraction": NEAR_CLOSE_FRACTION,
            "hold_bars": HOLD_BARS,
            "first_target_move_pct": 100.0 * risk * NEAR_TARGET_R / entry,
            "runner_move_pct": 100.0 * risk * RUNNER_TARGET_R / entry,
            "utc_session": _session(entry_time),
            "model_quality": quality,
            "signal_bar_ts": float(entry_time) - 900.0,
            "_shadow_experiment": True,
            "_precision_profit_first_shadow": True,
        }]
    return []
