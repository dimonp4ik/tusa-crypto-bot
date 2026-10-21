"""Chronological audit of material multi-day crypto trend entries.

This is a new entry generator rather than a filter over legacy setups.  It
uses closed five-minute candles sampled hourly to identify persistent trends,
channel expansion, or a pullback/reclaim inside a longer trend.  Entries occur
at the next five-minute market open and seek literal 2-5% price moves.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_precision_be_runner_oct18_audit as rails
from crypto_runtime_shadow_portfolio_oct3_audit import money
from src import research_crypto_replay as replay


ROOT = Path("reports/audit_2026_09_08")
HISTORY = ROOT / "crypto_5m_history"
MANIFEST = HISTORY / "manifest.json"
OUT = ROOT / "crypto_multiday_material_trend_oct21.json"
ROWS_OUT = ROOT / "crypto_multiday_material_trend_oct21.pkl"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "XLMUSDT",
           "LINKUSDT", "SOLUSDT", "AVAXUSDT", "AAVEUSDT")
DEVELOPMENT = (2020, 2021, 2022)
VALIDATION = (2023, 2024)
HIDDEN = (2025, 2026)
TARGETS = (2.0, 3.0, 5.0)
STOPS = (.75, 1.0, 1.5)
HOLDS = (144, 288, 576)  # 12h, 24h, 48h in five-minute bars
TOP_EXACT = 80
MARGIN = .03
LEVERAGE = 10.0
MONTH_DAYS = 30
MONTH_TARGET = .30


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict[str, dict], dict[str, str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candles, sources = {}, {str(MANIFEST): digest(MANIFEST)}
    for symbol in SYMBOLS:
        meta = manifest["symbols"][symbol]
        path = Path(meta["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != meta["sha256"]:
            raise ValueError(f"history hash mismatch: {symbol}")
        candles[symbol] = pd.read_pickle(path)
        sources[str(path)] = found
    return candles, sources


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window).sum().to_numpy(float)


def symbol_features(symbol: str, candle: dict) -> pd.DataFrame:
    times = np.asarray(candle["time"], dtype=np.int64)
    open_ = np.asarray(candle["open"], dtype=float)
    high = np.asarray(candle["high"], dtype=float)
    low = np.asarray(candle["low"], dtype=float)
    close = np.asarray(candle["close"], dtype=float)
    volume = np.asarray(candle["volume"], dtype=float)
    maximum = 168 * 12
    indices = np.arange(maximum, len(times) - 1, dtype=np.int64)
    mask = (times[indices + 1] % 3600 == 0)
    mask &= times[indices + 1] - times[indices] == 300
    mask &= times[indices] - times[indices - maximum] == maximum * 300
    indices = indices[mask]

    frame = pd.DataFrame({
        "symbol": symbol, "entry_index": indices + 1,
        "entry_time": times[indices + 1],
    })
    for hours in (1, 4, 12, 24, 72, 168):
        lag = hours * 12
        frame[f"ret{hours}"] = close[indices] / close[indices - lag] - 1
        path = _rolling_sum(np.abs(np.diff(close, prepend=close[0])), lag)
        frame[f"eff{hours}"] = np.divide(
            np.abs(close[indices] - close[indices - lag]), path[indices],
            out=np.full(len(indices), np.nan), where=path[indices] > 0)

    hourly_return = close[indices] / close[indices - 12] - 1
    sigma = pd.Series(hourly_return).rolling(168).std().shift(1).to_numpy(float)
    for hours in (12, 24, 72, 168):
        frame[f"trend_z{hours}"] = np.divide(
            frame[f"ret{hours}"].to_numpy(float), sigma * math.sqrt(hours),
            out=np.full(len(indices), np.nan), where=sigma > 0)

    prior_high24 = pd.Series(high).shift(1).rolling(24 * 12).max().to_numpy(float)
    prior_low24 = pd.Series(low).shift(1).rolling(24 * 12).min().to_numpy(float)
    prior_high72 = pd.Series(high).shift(1).rolling(72 * 12).max().to_numpy(float)
    prior_low72 = pd.Series(low).shift(1).rolling(72 * 12).min().to_numpy(float)
    previous = np.r_[close[0], close[:-1]]
    true_range = np.maximum(
        high - low, np.maximum(abs(high - previous), abs(low - previous)))
    atr24 = pd.Series(true_range).rolling(24 * 12).mean().to_numpy(float)
    frame["breakout_up24_atr"] = (
        close[indices] - prior_high24[indices]) / atr24[indices]
    frame["breakout_down24_atr"] = (
        prior_low24[indices] - close[indices]) / atr24[indices]
    frame["breakout_up72_atr"] = (
        close[indices] - prior_high72[indices]) / atr24[indices]
    frame["breakout_down72_atr"] = (
        prior_low72[indices] - close[indices]) / atr24[indices]
    hourly_volume = _rolling_sum(volume, 12)[indices]
    volume_baseline = pd.Series(hourly_volume).rolling(168).median().shift(1)
    frame["volume_ratio"] = np.divide(
        hourly_volume, volume_baseline.to_numpy(float),
        out=np.full(len(indices), np.nan),
        where=volume_baseline.to_numpy(float) > 0)
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    frame["year"] = stamp.dt.year.astype(np.int16)
    frame["month"] = stamp.dt.strftime("%Y-%m")
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def build_features(candles: dict[str, dict]) -> pd.DataFrame:
    return pd.concat(
        [symbol_features(symbol, candle) for symbol, candle in candles.items()],
        ignore_index=True)


def specs() -> list[dict]:
    output = []
    for hours, move, efficiency in (
            (hours, move, efficiency)
            for hours, moves in ((12, (.01, .02)), (24, (.015, .03)),
                                 (72, (.03, .06)))
            for move, efficiency in itertools.product(moves, (.15, .30))):
        output.append({
            "name": f"persistence_{hours}h_m{move}_e{efficiency}",
            "family": "trend_persistence", "hours": hours,
            "minimum_move": move, "minimum_efficiency": efficiency,
        })
    for hours, volume_minimum, efficiency in itertools.product(
            (24, 72), (.8, 1.2), (.10, .25)):
        output.append({
            "name": f"channel_{hours}h_v{volume_minimum}_e{efficiency}",
            "family": "channel_expansion", "hours": hours,
            "minimum_volume": volume_minimum,
            "minimum_efficiency": efficiency,
        })
    for hours, trend_z in itertools.product((72, 168), (1.0, 1.5)):
        output.append({
            "name": f"reclaim_{hours}h_z{trend_z}",
            "family": "pullback_reclaim", "hours": hours,
            "minimum_trend_z": trend_z,
        })
    return output


def proposals(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    family, hours = spec["family"], int(spec["hours"])
    if family == "trend_persistence":
        direction = np.sign(frame[f"ret{hours}"]).astype(np.int8)
        mask = (frame[f"ret{hours}"].abs().ge(spec["minimum_move"])
                & frame[f"eff{hours}"].ge(spec["minimum_efficiency"])
                & (frame.ret4 * direction).gt(0)
                & frame[f"trend_z{hours}"].abs().ge(.75))
        quality = (frame[f"trend_z{hours}"].abs()
                   * (1 + frame[f"eff{hours}"]))
    elif family == "channel_expansion":
        up = frame[f"breakout_up{hours}_atr"]
        down = frame[f"breakout_down{hours}_atr"]
        direction = np.where(up >= down, 1, -1).astype(np.int8)
        breakout = np.maximum(up, down)
        mask = (pd.Series(breakout, index=frame.index).ge(0)
                & frame.volume_ratio.ge(spec["minimum_volume"])
                & frame[f"eff{hours}"].ge(spec["minimum_efficiency"])
                & (frame.ret1 * direction).gt(0))
        quality = (pd.Series(breakout, index=frame.index).clip(lower=0)
                   + frame.volume_ratio + frame[f"eff{hours}"])
    else:
        direction = np.sign(frame[f"ret{hours}"]).astype(np.int8)
        directed_pullback = frame.ret4 * direction
        directed_reclaim = frame.ret1 * direction
        mask = (frame[f"trend_z{hours}"].abs().ge(
                    spec["minimum_trend_z"])
                & directed_pullback.between(-.03, .003)
                & directed_reclaim.ge(.001)
                & frame[f"eff{hours}"].ge(.12))
        quality = (frame[f"trend_z{hours}"].abs()
                   + 20 * directed_reclaim)
    found = frame.loc[mask].copy()
    found["direction"] = np.asarray(direction)[mask.to_numpy()]
    found["spec_name"] = spec["name"]
    found["family"] = family
    found["model_probability"] = quality.loc[mask].to_numpy(float)
    return found


def standard_rows(frame: pd.DataFrame) -> list[dict]:
    output = []
    for row in frame.to_dict("records"):
        signed = float(row["signed_return"])
        output.append({
            **row,
            "direction_n": int(row["direction"]),
            "direction": "LONG" if int(row["direction"]) > 0 else "SHORT",
            "signed_price_return": signed,
            "outcome": "SL" if signed <= 0 else "TP",
            "module": f"multiday:{row['spec_name']}",
            "source": "multiday_material_trend",
            "source_priority": 0, "module_priority": 0,
        })
    return output


def accepted(frame: pd.DataFrame, years: tuple[int, ...]) -> list[dict]:
    part = frame[frame.year.isin(years)].copy()
    part.sort_values(
        ["symbol", "entry_time", "model_probability", "spec_name"],
        ascending=[True, True, False, True], inplace=True, kind="mergesort")
    part.drop_duplicates(["symbol", "entry_time"], inplace=True)
    return rails.ranked_runtime_gate(
        standard_rows(part), loss_pause=3, hourly_cap=2)


def required_move(trades_per_day: float, margin: float = MARGIN) -> float | None:
    periods = MONTH_DAYS * trades_per_day
    return (None if periods <= 0 else
            ((1 + MONTH_TARGET) ** (1 / periods) - 1)
            / (margin * LEVERAGE) * 100)


def enrich_metrics(value: dict, rows: list[dict]) -> dict:
    signed = np.asarray([float(row["signed_price_return"]) for row in rows])
    frequency = float(value.get("trades_per_calendar_day", 0))
    mean = float(signed.mean()) if len(signed) else 0.0
    periods = frequency * MONTH_DAYS
    factor = 1 + MARGIN * LEVERAGE * mean
    projected = (100 * (factor ** periods - 1)
                 if periods > 0 and factor > 0 else -100.0)
    required = required_move(frequency)
    value.update({
        "profitable_exit_rate_pct": round(
            100 * float(np.mean(signed > 0)), 4) if len(signed) else 0.0,
        "literal_target_rate_pct": round(
            100 * sum(int(row["outcome_code"]) == 2 for row in rows)
            / len(rows), 4) if rows else 0.0,
        "mean_signed_price_return_pct": round(100 * mean, 6),
        "projected_30_day_pct": round(projected, 6),
        "required_mean_signed_move_at_actual_frequency_pct": (
            round(required, 6) if required is not None else None),
        "mandatory_three_daily_mean_signed_move_pct": round(
            required_move(3) or 0, 6),
        "economic_gap_multiple": (
            round(required / (100 * mean), 4)
            if required is not None and mean > 0 else None),
    })
    return value


def exact_stage(frame: pd.DataFrame, years: tuple[int, ...]) -> tuple[dict, list[dict]]:
    rows = accepted(frame, years)
    combined = enrich_metrics(money(rows, MARGIN), rows)
    annual = {}
    for year in years:
        year_rows = [row for row in rows if pd.Timestamp(
            float(row["entry_time"]), unit="s", tz="UTC").year == year]
        annual[str(year)] = enrich_metrics(money(year_rows, MARGIN), year_rows)
    return {"margin_3pct": combined, "annual_3pct": annual}, rows


def raw_summary(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    part = frame[frame.year.isin(years)]
    if part.empty:
        return {"n": 0, "mean_signed_price_return_pct": 0.0,
                "profitable_exit_rate_pct": 0.0, "positive_years": 0}
    annual = part.groupby("year").signed_return.mean()
    return {
        "n": len(part),
        "mean_signed_price_return_pct": round(
            100 * float(part.signed_return.mean()), 6),
        "profitable_exit_rate_pct": round(
            100 * float(part.signed_return.gt(0).mean()), 4),
        "positive_years": int((annual > 0).sum()),
        "annual_mean_signed_price_return_pct": {
            str(int(year)): round(100 * float(value), 6)
            for year, value in annual.items()
        },
    }


def development_ok(stage: dict) -> bool:
    value = stage["margin_3pct"]
    years = list(stage["annual_3pct"].values())
    return bool(
        value["n"] >= 2000 and value["return_pct"] > 0
        and value["dd_pct"] <= 15
        and value["trades_per_calendar_day"] >= 2.5
        and all(item["return_pct"] > 0 for item in years)
    )


def validation_ok(stage: dict) -> bool:
    value = stage["margin_3pct"]
    years = list(stage["annual_3pct"].values())
    return bool(
        value["n"] >= 1200 and value["return_pct"] > 0
        and value["dd_pct"] <= 12
        and value["trades_per_calendar_day"] >= 2.5
        and all(item["return_pct"] > 0 for item in years)
    )


def objective_ok(stage: dict) -> bool:
    value = stage["margin_3pct"]
    months = value.get("monthly_return_pct", {})
    return bool(
        months and min(months.values()) >= 30
        and value["negative_months"] == 0 and value["dd_pct"] <= 5
        and value["trades_per_calendar_day"] >= 3
        and value["projected_30_day_pct"] >= 30)


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if key not in ("_frame", "_accepted")}


def main() -> None:
    candles, sources = load()
    features = build_features(candles)
    rules = specs()
    proposal_frame = pd.concat(
        [proposals(features, rule) for rule in rules], ignore_index=True)
    candidates = []
    searched = 0
    for target, stop, hold in itertools.product(TARGETS, STOPS, HOLDS):
        labeled = replay.add_outcomes(
            proposal_frame, candles, target, stop, hold)
        labeled["year"] = pd.to_datetime(
            labeled.entry_time, unit="s", utc=True).dt.year.astype(np.int16)
        for rule in rules:
            searched += 1
            frame = labeled[labeled.spec_name.eq(rule["name"])].copy()
            raw = raw_summary(frame, DEVELOPMENT)
            if (raw["n"] < 2500
                    or raw["mean_signed_price_return_pct"] <= 0
                    or raw["positive_years"] < 2):
                continue
            score = (raw["mean_signed_price_return_pct"]
                     * math.sqrt(raw["n"]))
            candidates.append({
                "rule": rule,
                "exit": {"target_pct": target, "stop_pct": stop,
                         "hold_bars_5m": hold},
                "development_raw": raw, "raw_score": score,
                "_frame": frame,
            })
        print(json.dumps({"target": target, "stop": stop, "hold": hold,
                          "searched": searched,
                          "raw_passed": len(candidates)}), flush=True)

    candidates.sort(key=lambda item: item["raw_score"], reverse=True)
    exact = []
    replayed = []
    for item in candidates[:TOP_EXACT]:
        stage, rows = exact_stage(item["_frame"], DEVELOPMENT)
        value = stage["margin_3pct"]
        annual_floor = min(
            year["return_pct"] for year in stage["annual_3pct"].values())
        replayed_item = {
            **item, "development": stage, "_accepted": rows,
            "development_score": (
                annual_floor + .2 * value["return_pct"] - value["dd_pct"]),
            "development_pass": development_ok(stage),
        }
        replayed.append(replayed_item)
        if replayed_item["development_pass"]:
            exact.append(replayed_item)
    replayed.sort(key=lambda item: item["development_score"], reverse=True)
    exact.sort(key=lambda item: item["development_score"], reverse=True)

    validated = []
    for item in exact:
        stage, _ = exact_stage(item["_frame"], VALIDATION)
        item["validation"] = stage
        if validation_ok(stage):
            validated.append(item)
    frozen = validated[0] if validated else None
    hidden = None
    status = "REJECTED_DEVELOPMENT" if not exact else "REJECTED_VALIDATION"
    if frozen:
        hidden, rows = exact_stage(frozen["_frame"], HIDDEN)
        frozen["_accepted"] = rows
        status = ("MULTIDAY_MATERIAL_CANDIDATE" if validation_ok(hidden)
                  else "REJECTED_HIDDEN")

    selected = frozen if frozen else (exact[0] if exact else None)
    diagnostic = replayed[0] if replayed else None
    pd.DataFrame((selected or diagnostic)["_accepted"]
                 if (selected or diagnostic) else []).to_pickle(ROWS_OUT)
    objective = bool(
        frozen and hidden and objective_ok(frozen["development"])
        and objective_ok(frozen["validation"]) and objective_ok(hidden))
    report = {
        "status": status, "ready_for_live_orders": False,
        "orders_possible": False,
        "protocol": {
            "hypothesis": (
                "persistent 12-168h trend, channel expansion, or long-trend "
                "pullback/reclaim can support literal 2-5% moves"),
            "development": list(DEVELOPMENT),
            "validation": list(VALIDATION),
            "hidden_reveal": list(HIDDEN),
            "entry": "next five-minute market open after hourly closed signal",
            "orders": "market only", "collision": "stop first",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
            "portfolio": (
                "one ranked setup per timestamp, one position per symbol, "
                "two per rolling hour, three same-direction open, daily pause "
                "after three losses"),
        },
        "grid": {
            "feature_rows": len(features), "proposal_rows": len(proposal_frame),
            "entry_specs": len(rules), "exit_specs": 27,
            "searched_candidates": searched,
            "raw_development_passes": len(candidates),
            "exact_development_passes": len(exact),
            "validation_passes": len(validated),
            "exact_replayed": min(TOP_EXACT, len(candidates)),
        },
        "selected": public(selected) if selected else None,
        "best_development_diagnostic": (
            public(diagnostic) if diagnostic else None),
        "hidden": hidden,
        "all_stages_meet_monthly_objective": objective,
        "top_development": [public(item) for item in exact[:10]],
        "top_exact_replays": [public(item) for item in replayed[:10]],
        "sources": {
            **sources,
            str(Path(replay.__file__).resolve()):
                digest(Path(replay.__file__).resolve()),
            str(Path(rails.__file__).resolve()):
                digest(Path(rails.__file__).resolve()),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Binance spot history is a signal proxy and requires X-Perp transfer.",
            "The eight-symbol universe expands during 2020 as listings begin.",
            "The 2026 archive ends in September.",
            "Gross returns exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "grid": report["grid"],
        "selected": report["selected"], "hidden": hidden,
        "best_development_diagnostic":
            report["best_development_diagnostic"],
        "all_stages_meet_monthly_objective": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
