"""Causal daily trend hypotheses on cached OKX candles.

Signals are formed after a daily close and executed at the next daily open.
Portfolio weights are inverse-volatility scaled to one unit of gross exposure.
The output is research-only and does not place orders.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


FIELDS = ("open", "high", "low", "close", "volume")


def load_symbol(cache_dir: Path, symbol: str, pattern: str) -> tuple[pd.DataFrame, list[Path]]:
    paths = [Path(item) for item in sorted(glob.glob(str(cache_dir / pattern.format(symbol=symbol))))]
    if not paths:
        raise FileNotFoundError(f"no cached candles for {symbol}: {pattern}")
    records: dict[int, dict] = {}
    for path in paths:
        payload = pd.read_pickle(path)
        if not isinstance(payload, dict) or "time" not in payload:
            raise ValueError(f"unexpected cache payload: {path}")
        for index, timestamp in enumerate(payload["time"]):
            records[int(timestamp)] = {field: float(payload[field][index]) for field in FIELDS}
    frame = pd.DataFrame.from_dict(records, orient="index").sort_index()
    frame.index = pd.to_datetime(frame.index, unit="s", utc=True)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"invalid time index for {symbol}")
    return frame, paths


def donchian_score(frame: pd.DataFrame, lookbacks: tuple[int, ...] = (20, 60, 120)) -> pd.Series:
    states = []
    for lookback in lookbacks:
        prior_high = frame["high"].shift(1).rolling(lookback, min_periods=lookback).max()
        prior_low = frame["low"].shift(1).rolling(lookback, min_periods=lookback).min()
        changes = pd.Series(np.nan, index=frame.index)
        changes.loc[frame["close"] > prior_high] = 1.0
        changes.loc[frame["close"] < prior_low] = -1.0
        states.append(changes.ffill().fillna(0.0))
    return pd.concat(states, axis=1).mean(axis=1)


def momentum_long_cash(frame: pd.DataFrame, lookback: int = 30) -> pd.Series:
    momentum = np.log(frame["close"] / frame["close"].shift(lookback))
    signal = (momentum > 0).astype(float)
    signal[momentum.isna()] = 0.0
    return signal


def market_risk_multiplier(frames: dict[str, pd.DataFrame]) -> pd.Series:
    closes = pd.concat({symbol: frame["close"] for symbol, frame in frames.items()}, axis=1).dropna()
    market_return = closes.pct_change().mean(axis=1).fillna(0.0)
    market_index = (1.0 + market_return).cumprod()
    peak = market_index.rolling(90, min_periods=30).max()
    multiplier = pd.Series(1.0, index=market_index.index)
    multiplier.loc[market_index < 0.85 * peak] = 0.5
    return multiplier


def portfolio_returns(
    frames: dict[str, pd.DataFrame],
    signals: dict[str, pd.Series],
    *,
    one_way_cost_bps: float,
    crash_overlay: bool,
) -> pd.DataFrame:
    opens = pd.concat({symbol: frame["open"] for symbol, frame in frames.items()}, axis=1).dropna()
    closes = pd.concat({symbol: frame["close"] for symbol, frame in frames.items()}, axis=1).reindex(opens.index)
    signal_at_open = pd.concat(signals, axis=1).reindex(opens.index).shift(1).fillna(0.0)
    volatility = closes.pct_change().rolling(20, min_periods=20).std().shift(1)
    raw_weight = signal_at_open.div(volatility.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = raw_weight.abs().sum(axis=1).replace(0.0, np.nan)
    weights = raw_weight.div(gross, axis=0).fillna(0.0)
    if crash_overlay:
        # The state is known only after the prior close, like the trading signal.
        overlay = market_risk_multiplier(frames).reindex(weights.index).shift(1).fillna(1.0)
        weights = weights.mul(overlay, axis=0)
    forward_open_return = opens.shift(-1).div(opens).sub(1.0)
    gross_return = (weights * forward_open_return).sum(axis=1, min_count=1)
    turnover = weights.sub(weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (one_way_cost_bps / 10_000.0)
    result = pd.DataFrame({
        "gross_return": gross_return,
        "cost": cost,
        "net_return": gross_return - cost,
        "gross_exposure": weights.abs().sum(axis=1),
        "turnover": turnover,
    }).dropna(subset=["net_return"])
    return result


def metrics(result: pd.DataFrame) -> dict:
    returns = result["net_return"]
    equity = (1.0 + returns).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1.0)
    deviation = returns.std(ddof=1)
    return {
        "days": int(len(result)),
        "active_days": int((result["gross_exposure"] > 0).sum()),
        "net_return_pct": float((equity.iloc[-1] - 1.0) * 100) if len(equity) else 0.0,
        "max_drawdown_pct": float(drawdown.min() * 100) if len(drawdown) else 0.0,
        "annualized_sharpe": float(math.sqrt(365) * returns.mean() / deviation) if deviation and math.isfinite(deviation) else None,
        "daily_win_rate": float((returns > 0).mean() * 100) if len(returns) else 0.0,
        "total_cost_pct": float(result["cost"].sum() * 100),
        "turnover": float(result["turnover"].sum()),
    }


def split_metrics(result: pd.DataFrame) -> dict:
    output = {"full": metrics(result)}
    for year in sorted(set(result.index.year)):
        output[str(year)] = metrics(result[result.index.year == year])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="{symbol}_1d_192*.pkl")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--one-way-cost-bps", type=float, default=6.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.one_way_cost_bps < 0:
        parser.error("one-way-cost-bps must be nonnegative")

    frames: dict[str, pd.DataFrame] = {}
    source_paths: list[Path] = []
    for symbol in args.symbols:
        frame, paths = load_symbol(args.cache_dir, symbol, args.pattern)
        frames[symbol] = frame
        source_paths.extend(paths)

    common_index = frames[args.symbols[0]].index
    for frame in frames.values():
        common_index = common_index.intersection(frame.index)
    frames = {symbol: frame.reindex(common_index) for symbol, frame in frames.items()}

    hypotheses = {
        "donchian_20_60_120": {symbol: donchian_score(frame) for symbol, frame in frames.items()},
        "momentum_30d_long_cash": {symbol: momentum_long_cash(frame) for symbol, frame in frames.items()},
    }
    results = {}
    for name, signals in hypotheses.items():
        for overlay in (False, True):
            label = name + ("_crash_half" if overlay else "")
            series = portfolio_returns(
                frames,
                signals,
                one_way_cost_bps=args.one_way_cost_bps,
                crash_overlay=overlay,
            )
            results[label] = split_metrics(series)

    report = {
        "status": "RESEARCH_ONLY",
        "coverage": {
            "first": str(common_index.min()),
            "last": str(common_index.max()),
            "days": len(common_index),
            "symbols": args.symbols,
        },
        "cost_model": {"one_way_bps": args.one_way_cost_bps, "round_trip_bps": 2 * args.one_way_cost_bps},
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "results": results,
        "limitations": [
            "Daily SWAP candles are a signal proxy; they do not prove X-Perp execution quality.",
            "Funding, tax, borrow constraints and liquidation are excluded.",
            "The strategy uses no leverage; leveraged deployment would multiply both returns and drawdowns nonlinearly.",
            "These prespecified hypotheses still require forward paper validation before live use.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"coverage": report["coverage"], "results": report["results"]}))


if __name__ == "__main__":
    main()
