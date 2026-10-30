"""Frozen shadow challenger that banks weak-volume TP1 candles.

The challenger never changes an entry.  It observes a completed TP1-hit bar,
compares its volume with the preceding 20 completed bars, and records a market
close at that bar's close for paper comparison with the baseline exit.
"""
from __future__ import annotations

import math
import statistics


VOLUME_LOOKBACK = 20
VOLUME_RATIO_THRESHOLD = 1.5
EXTRA_CLOSE_FRACTION = .75
BASE_TP1_CLOSE_FRACTION = .05


def completed_observation(candles: dict, *, activation_ts: float,
                          bar_seconds: int, now: float) -> dict | None:
    """Return a causal TP1-bar observation once that exact bar is complete."""
    times = candles.get("time") or []
    try:
        index = next(i for i, value in enumerate(times)
                     if float(value) == float(activation_ts))
    except StopIteration:
        return None
    flags = candles.get("confirmed") or []
    confirmed = (bool(flags[index]) if index < len(flags)
                 else float(times[index]) + int(bar_seconds) <= float(now))
    if not confirmed or index < VOLUME_LOOKBACK:
        return None
    volumes = candles.get("volume") or []
    closes = candles.get("close") or []
    if index >= len(volumes) or index >= len(closes):
        return None
    prior = [float(value) for value in
             volumes[index - VOLUME_LOOKBACK:index]]
    current = float(volumes[index])
    close = float(closes[index])
    median = float(statistics.median(prior))
    if (not all(math.isfinite(value) and value >= 0 for value in prior)
            or not math.isfinite(current) or current < 0
            or not math.isfinite(close) or close <= 0
            or median <= 0):
        return None
    ratio = current / median
    return {
        "volume_ratio": ratio,
        "close": close,
        "extra_close_fraction": (
            EXTRA_CLOSE_FRACTION
            if ratio < VOLUME_RATIO_THRESHOLD else 0.0),
        "observed_at": float(times[index]) + int(bar_seconds),
    }


def candidate_r(*, tp1_fraction: float, tp1_r: float,
                extra_fraction: float, extra_r: float,
                runner_r: float) -> float:
    """Three-leg paper R: TP1 bank, optional close bank, final runner."""
    values = (tp1_fraction, tp1_r, extra_fraction, extra_r, runner_r)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("volume-bank R inputs must be finite")
    tp1_fraction = float(tp1_fraction)
    extra_fraction = float(extra_fraction)
    if (tp1_fraction < 0 or extra_fraction < 0
            or tp1_fraction + extra_fraction > 1):
        raise ValueError("invalid close fractions")
    runner_fraction = 1.0 - tp1_fraction - extra_fraction
    return round(
        tp1_fraction * float(tp1_r)
        + extra_fraction * float(extra_r)
        + runner_fraction * float(runner_r), 4)
