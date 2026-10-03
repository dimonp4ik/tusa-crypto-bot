"""Causal X-Perp regime modules frozen by the September 2026 audit.

The router only proposes market-entry brackets.  It never chooses client size
and it does not place orders.  Callers must pass closed 15-minute candles.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math


@dataclass(frozen=True)
class Module:
    name: str
    family: str
    direction: str
    btc_regime: str
    utc_session: str
    target_r: float
    conditions: tuple[tuple[str, str, float], ...]


# Order is the frozen venue-calibration priority. The base rules were discovered
# on 2020-24, validated on 2025 and base-market 2026, then calibrated using
# X-Perp outcomes strictly before 2026-08-27. The evening-short abs-z ceiling
# was selected on 2020-23, validated on 2024, and then survived 2025-26 plus
# direct and later paper-forward venue replays.
MODULES = (
    Module("pullback_long_bull_europe", "trend_pullback", "LONG", "bull", "06_11", .5,
           (("btc_eff96", "ge", .2), ("btc_eff20", "le", .1))),
    Module("rr_long_bear_rally_europe", "range_reversion", "LONG", "bear_rally", "06_11", .5,
           (("btc_slow_long_atr", "ge", -1.0),)),
    Module("rr_long_bear_day", "range_reversion", "LONG", "bear", "12_17", .5,
           (("btc_slow_long_atr", "ge", -1.0), ("eff_ratio", "ge", .2))),
    Module("rr_short_bull_pullback_asia", "range_reversion", "SHORT", "bull_pullback",
           "00_05", .25,
           (("btc_eff96", "ge", .1), ("btc_fast_slow_atr", "le", -1.0))),
    Module("rr_short_bear_asia", "range_reversion", "SHORT", "bear", "00_05", .25,
           (("btc_fast_slow_atr", "le", -1.0), ("btc_slow_long_atr", "ge", -2.0))),
    Module("rr_short_bull_evening", "range_reversion", "SHORT", "bull", "18_23", .25,
           (("btc_fast_slow_atr", "le", 2.0), ("btc_slow_long_atr", "ge", 4.0),
            ("abs_z", "le", 3.0))),
    Module("breakout_long_bull_pullback_evening", "breakout", "LONG", "bull_pullback",
           "18_23", .25,
           (("btc_slow_long_atr", "le", 2.0), ("btc_fast_slow_atr", "ge", -1.0))),
)

# These modules passed chronological transfer checks but have not reached the
# predeclared 20-trade/two-month unseen activation gate.  Callers must keep
# them shadow-only; they are intentionally absent from MODULES.
SHADOW_EXPERIMENTAL_MODULES = (
    Module(
        "shadow_pullback_short_bull_pullback_evening",
        "trend_pullback", "SHORT", "bull_pullback", "18_23", .25,
        (("btc_return20_atr", "ge", 2.0),
         ("btc_slow_long_atr", "ge", 4.0),
         ("abs_z", "ge", 1.25)),
    ),
    Module(
        "shadow_rr_long_bull_asia_high_vol",
        "range_reversion", "LONG", "bull", "00_05", .25,
        (("btc_atr_pct", "ge", .008),),
    ),
    Module(
        "shadow_pullback_short_bull_pullback_asia_trend",
        "trend_pullback", "SHORT", "bull_pullback", "00_05", .25,
        (("btc_return96_atr", "ge", 0.0),
         ("btc_slow_long_atr", "ge", 2.0)),
    ),
    Module(
        "shadow_pullback_short_bull_pullback_evening_slow_btc",
        "trend_pullback", "SHORT", "bull_pullback", "18_23", .25,
        (("btc_eff20", "ge", .2),
         ("btc_return96_atr", "le", -4.0)),
    ),
)

AUDITED_SYMBOLS = frozenset({
    "AAVEUSDT", "ADAUSDT", "AVAXUSDT", "BILLUSDT", "BTCUSDT", "DOTUSDT",
    "ETHUSDT", "HYPEUSDT", "LINKUSDT", "NEARUSDT", "SOLUSDT", "SUIUSDT",
    "TAOUSDT", "XLMUSDT", "XRPUSDT", "ZECUSDT",
})
ROBUST_EXCLUDED_SYMBOLS = frozenset({
    "ADAUSDT", "DOTUSDT", "NEARUSDT", "TAOUSDT",
})
ROBUST_SYMBOLS = AUDITED_SYMBOLS - ROBUST_EXCLUDED_SYMBOLS

# Shadow candidate selected on 2020-23, validated on 2024, then checked on
# 2025-26 and public X-Perp through 2026-09-30.  The base module stays at
# 0.25R when BTC is choppy; a directional BTC path gives it room to 0.50R.
CONDITIONAL_TARGET_MODULE = "rr_short_bull_pullback_asia"
CONDITIONAL_TARGET_BTC_EFF20_MIN = .1
CONDITIONAL_TARGET_R = .5


def _session(timestamp: float) -> str:
    hour = datetime.fromtimestamp(timestamp, timezone.utc).hour
    if hour <= 5:
        return "00_05"
    if hour <= 11:
        return "06_11"
    if hour <= 17:
        return "12_17"
    return "18_23"


def _atr(candles: dict, start: int, end: int) -> float:
    close = candles["close"]
    values = [max(
        float(candles["high"][index]) - float(candles["low"][index]),
        abs(float(candles["high"][index]) - float(close[index - 1])),
        abs(float(candles["low"][index]) - float(close[index - 1])),
    ) for index in range(start, end)]
    return sum(values) / len(values)


def btc_context(candles: dict) -> dict | None:
    """Return the exact closed-bar BTC features used by the research replay."""
    close = list(map(float, candles.get("close", [])))
    high = candles.get("high", [])
    low = candles.get("low", [])
    if len(close) < 385 or len(high) != len(close) or len(low) != len(close):
        return None
    history = close[-385:]
    recent, slow, long = history[-20:], history[-96:], history[-384:]
    atr = _atr({"close": close, "high": high, "low": low}, len(close) - 14, len(close))
    movement20 = sum(abs(recent[i] - recent[i - 1]) for i in range(1, 20))
    movement96 = sum(abs(slow[i] - slow[i - 1]) for i in range(1, 96))
    deviation = math.sqrt(sum((value - sum(recent) / 20) ** 2 for value in recent) / 20)
    if not atr or not movement20 or not movement96 or not deviation or recent[-1] <= 0:
        return None
    fast_mean, slow_mean, long_mean = sum(recent) / 20, sum(slow) / 96, sum(long) / 384
    if fast_mean > slow_mean > long_mean:
        regime = "bull"
    elif fast_mean < slow_mean < long_mean:
        regime = "bear"
    elif slow_mean > long_mean:
        regime = "bull_pullback"
    else:
        regime = "bear_rally"
    result = {
        "btc_regime": regime,
        "btc_fast_slow_atr": (fast_mean - slow_mean) / atr,
        "btc_slow_long_atr": (slow_mean - long_mean) / atr,
        "btc_eff20": abs(recent[-1] - recent[0]) / movement20,
        "btc_eff96": abs(slow[-1] - slow[0]) / movement96,
        "btc_return20_atr": (recent[-1] - recent[0]) / atr,
        "btc_return96_atr": (slow[-1] - slow[0]) / atr,
        "btc_atr_pct": atr / recent[-1],
    }
    return result if all(math.isfinite(value) for key, value in result.items()
                         if key != "btc_regime") else None


def family_signals(candles: dict) -> list[dict]:
    """Generate entry families from the latest closed bar for the next market entry."""
    close = list(map(float, candles.get("close", [])))
    high = candles.get("high", [])
    low = candles.get("low", [])
    if len(close) < 100 or len(high) != len(close) or len(low) != len(close):
        return []
    last = close[-1]
    recent = close[-20:]
    mean = sum(recent) / 20
    std = math.sqrt(sum((value - mean) ** 2 for value in recent) / 20)
    atr = _atr({"close": close, "high": high, "low": low}, len(close) - 14, len(close))
    path = sum(abs(close[index] - close[index - 1])
               for index in range(len(close) - 19, len(close)))
    slow = sum(close[-96:]) / 96
    if last <= 0 or not atr or not std or not path:
        return []
    efficiency = abs(last - close[-20]) / path
    z = (last - mean) / std
    common = {"atr": atr, "eff_ratio": efficiency, "z": z, "abs_z": abs(z)}
    result = []
    if efficiency < .3:
        if z <= -2:
            result.append({**common, "family": "range_reversion", "direction": "LONG"})
        elif z >= 2:
            result.append({**common, "family": "range_reversion", "direction": "SHORT"})
    if mean > slow and last < mean - .5 * atr and last > slow and last > close[-2]:
        result.append({**common, "family": "trend_pullback", "direction": "LONG"})
    elif mean < slow and last > mean + .5 * atr and last < slow and last < close[-2]:
        result.append({**common, "family": "trend_pullback", "direction": "SHORT"})
    if last > max(map(float, high[-21:-1])) and last > slow:
        result.append({**common, "family": "breakout", "direction": "LONG"})
    elif last < min(map(float, low[-21:-1])) and last < slow:
        result.append({**common, "family": "breakout", "direction": "SHORT"})
    return result


def _matches(module: Module, signal: dict, context: dict, session: str) -> bool:
    if (module.family != signal["family"] or module.direction != signal["direction"]
            or module.btc_regime != context["btc_regime"] or module.utc_session != session):
        return False
    values = {**signal, **context}
    for field, operator, threshold in module.conditions:
        actual = values.get(field)
        if actual is None or (operator == "ge" and actual < threshold) \
                or (operator == "le" and actual > threshold):
            return False
    return True


def target_r_for(module: Module, context: dict) -> float:
    """Choose the audited target from information known before market entry."""
    if (module.name == CONDITIONAL_TARGET_MODULE
            and float(context.get("btc_eff20", -math.inf))
            >= CONDITIONAL_TARGET_BTC_EFF20_MIN):
        return CONDITIONAL_TARGET_R
    return module.target_r


def _setups_for_modules(candles: dict, btc_candles: dict,
                        modules: tuple[Module, ...], *, entry_time: float,
                        market_price: float | None = None,
                        symbol: str | None = None,
                        shadow_experiment: bool = False) -> list[dict]:
    if not symbol or symbol not in ROBUST_SYMBOLS:
        return []
    context = btc_context(btc_candles)
    if context is None or not math.isfinite(entry_time):
        return []
    session = _session(entry_time)
    entry = float(market_price if market_price is not None else candles["close"][-1])
    if not math.isfinite(entry) or entry <= 0:
        return []
    signals = family_signals(candles)
    for module in modules:
        for signal in signals:
            if not _matches(module, signal, context, session):
                continue
            risk = 2 * signal["atr"]
            sign = 1 if signal["direction"] == "LONG" else -1
            stop = entry - sign * risk
            target_r = target_r_for(module, context)
            target = entry + sign * risk * target_r
            if risk >= entry or min(stop, target) <= 0:
                continue
            return [{
                **signal, **context,
                "module": module.name,
                "_shadow_experiment": shadow_experiment,
                "direction": signal["direction"],
                "entry_source": "MARKET",
                "entry": entry,
                "sl": stop,
                "tp": target,
                "target_r": target_r,
                "utc_session": session,
            }]
    return []


def venue_setups(candles: dict, btc_candles: dict, *, entry_time: float,
                 market_price: float | None = None,
                 symbol: str | None = None) -> list[dict]:
    """Return frozen production matches with fixed-R market brackets."""
    return _setups_for_modules(
        candles, btc_candles, MODULES, entry_time=entry_time,
        market_price=market_price, symbol=symbol,
    )


def shadow_experimental_setups(
        candles: dict, btc_candles: dict, *, entry_time: float,
        market_price: float | None = None,
        symbol: str | None = None) -> list[dict]:
    """Return forward-test setups which are permanently ineligible for orders."""
    return _setups_for_modules(
        candles, btc_candles, SHADOW_EXPERIMENTAL_MODULES,
        entry_time=entry_time, market_price=market_price, symbol=symbol,
        shadow_experiment=True,
    )
