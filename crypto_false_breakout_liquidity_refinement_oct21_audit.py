"""Liquidity and level-quality refinement for the false-breakout edge."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

from crypto_5m_family_audit import load
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate
from crypto_actual_move_regime_router_oct20_audit import btc_context


ROOT = Path("reports/audit_2026_09_08")
SOURCE = ROOT / "crypto_5m_structural_breakout_oct20.pkl"
OUT = ROOT / "crypto_false_breakout_liquidity_refinement_oct21.json"
ROWS_OUT = ROOT / "crypto_false_breakout_liquidity_refinement_oct21.pkl"
CONTINUOUS = (
    "close_depth_atr", "extreme_depth_atr", "wick_fraction",
    "body_fraction", "prior_move_atr", "level_touches_96",
    "level_age_bars", "entry_gap_fade_pct", "room_to_ma96_pct",
    "same_direction_breakouts", "concurrent_breakouts",
    "btc_impulse_value",
)
QUANTILES = (.2, .4, .6, .8)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@nb.njit(cache=True)
def structure_features(open_, high, low, close, times, entry_indices,
                       fade_directions):
    n = len(entry_indices)
    values = np.full((n, 11), np.nan, dtype=np.float64)
    tr = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
    for pos in range(n):
        entry = entry_indices[pos]
        signal = entry - 1
        if signal < 96 or entry >= len(close) or times[entry] - times[signal] != 300:
            continue
        atr = 0.0
        for j in range(signal - 14, signal):
            atr += tr[j]
        atr /= 14.0
        if not np.isfinite(atr) or atr <= 0:
            continue
        breakout_direction = -fade_directions[pos]
        prior_high12 = np.max(high[signal - 12:signal])
        prior_low12 = np.min(low[signal - 12:signal])
        level = prior_high12 if breakout_direction > 0 else prior_low12
        extreme = high[signal] if breakout_direction > 0 else low[signal]
        span = high[signal] - low[signal]
        if span <= 0:
            continue
        close_depth = breakout_direction * (close[signal] - level) / atr
        extreme_depth = breakout_direction * (extreme - level) / atr
        if breakout_direction > 0:
            wick = (high[signal] - max(open_[signal], close[signal])) / span
        else:
            wick = (min(open_[signal], close[signal]) - low[signal]) / span
        body = abs(close[signal] - open_[signal]) / span
        prior_move = breakout_direction * (
            close[signal - 1] - close[signal - 13]) / atr
        tolerance = .20 * atr
        touches = 0
        last_touch = signal - 96
        for j in range(signal - 96, signal):
            probe = high[j] if breakout_direction > 0 else low[j]
            if abs(probe - level) <= tolerance:
                touches += 1
                last_touch = j
        age = signal - last_touch
        entry_gap = fade_directions[pos] * (
            open_[entry] - close[signal]) / close[signal]
        ma96 = 0.0
        for j in range(signal - 96, signal):
            ma96 += close[j]
        ma96 /= 96.0
        room = fade_directions[pos] * (ma96 - open_[entry]) / open_[entry]
        prior_high48 = np.max(high[signal - 48:signal])
        prior_low48 = np.min(low[signal - 48:signal])
        prior_high96 = np.max(high[signal - 96:signal])
        prior_low96 = np.min(low[signal - 96:signal])
        major48 = (close[signal] > prior_high48 if breakout_direction > 0
                   else close[signal] < prior_low48)
        major96 = (close[signal] > prior_high96 if breakout_direction > 0
                   else close[signal] < prior_low96)
        values[pos, 0] = close_depth
        values[pos, 1] = extreme_depth
        values[pos, 2] = wick
        values[pos, 3] = body
        values[pos, 4] = prior_move
        values[pos, 5] = touches
        values[pos, 6] = age
        values[pos, 7] = 100 * entry_gap
        values[pos, 8] = 100 * room
        values[pos, 9] = 1.0 if major48 else 0.0
        values[pos, 10] = 1.0 if major96 else 0.0
    return values


def enrich(source_rows: pd.DataFrame, candles: dict[str, dict]) -> pd.DataFrame:
    pieces = []
    names = [
        "close_depth_atr", "extreme_depth_atr", "wick_fraction",
        "body_fraction", "prior_move_atr", "level_touches_96",
        "level_age_bars", "entry_gap_fade_pct", "room_to_ma96_pct",
        "major_level_48", "major_level_96",
    ]
    for symbol, part in source_rows.groupby("symbol", sort=False):
        candle = candles[symbol]
        matrix = structure_features(
            np.asarray(candle["open"], dtype=float),
            np.asarray(candle["high"], dtype=float),
            np.asarray(candle["low"], dtype=float),
            np.asarray(candle["close"], dtype=float),
            np.asarray(candle["time"], dtype=np.int64),
            part.entry_index.to_numpy(np.int64),
            part.direction_n.to_numpy(np.int8))
        found = part.copy()
        for index, name in enumerate(names):
            found[name] = matrix[:, index]
        pieces.append(found)
    frame = pd.concat(pieces, ignore_index=True).dropna(subset=names)
    original_direction = -frame.direction_n.astype(int)
    frame["original_direction"] = np.where(original_direction > 0, "UP", "DOWN")
    frame["concurrent_breakouts"] = frame.groupby("entry_time").entry_time.transform("size")
    key = pd.MultiIndex.from_arrays([frame.entry_time, original_direction])
    counts = pd.Series(1, index=key).groupby(level=[0, 1]).sum()
    frame["same_direction_breakouts"] = [counts.loc[(time, direction)]
                                           for time, direction in zip(
                                               frame.entry_time, original_direction)]
    return frame.sort_values(["entry_time", "source_priority", "symbol"]).reset_index(drop=True)


def atom_mask(frame: pd.DataFrame, atom: dict) -> np.ndarray:
    values = frame[atom["field"]].to_numpy()
    if atom["op"] == "ge":
        return values >= atom["value"]
    if atom["op"] == "le":
        return values <= atom["value"]
    return values == atom["value"]


def atom_name(atom: dict) -> str:
    value = atom["value"]
    rendered = f"{float(value):.8g}" if isinstance(
        value, (int, float, np.integer, np.floating)) else str(value)
    return f'{atom["field"]}:{atom["op"]}:{rendered}'


def evaluate(records: list[dict], mask: np.ndarray,
             years: tuple[int, ...], margin: float) -> dict:
    wanted = set(years)
    selected = [records[index] for index in np.flatnonzero(mask)
                if int(records[index]["year"]) in wanted]
    accepted = runtime_gate(selected)
    result = money(accepted, margin)
    result["mean_signed_move_pct"] = round(
        100 * float(np.mean([row["signed_price_return"] for row in accepted])), 5
    ) if accepted else 0.0
    return result


def month_floor(periods: dict[str, dict]) -> float:
    months = [month for value in periods.values()
              for month in value.get("monthly_return_pct", {}).values()]
    return min(map(float, months), default=-999.0)


def main() -> None:
    source_frame = pd.read_pickle(SOURCE)
    candles, candle_sources = load()
    frame = enrich(source_frame, candles)
    frame = frame.merge(
        btc_context(candles["BTCUSDT"]), on="entry_time", how="left",
        validate="many_to_one").dropna(subset=["btc_impulse_value"])
    frame = frame.sort_values(
        ["entry_time", "source_priority", "symbol"]).reset_index(drop=True)
    records = frame.to_dict("records")
    selection = frame.year.isin((2020, 2021, 2022)).to_numpy()
    atoms = []
    for field in CONTINUOUS:
        for value in sorted(set(frame.loc[selection, field].quantile(
                list(QUANTILES)).astype(float))):
            atoms.extend(({"field": field, "op": "le", "value": value},
                          {"field": field, "op": "ge", "value": value}))
    for field in ("major_level_48", "major_level_96", "original_direction",
                  "btc_trend", "btc_impulse", "vol_regime", "session"):
        for value in sorted(frame[field].unique(), key=str):
            if isinstance(value, np.generic):
                value = value.item()
            atoms.append({"field": field, "op": "eq", "value": value})
    masks = {atom_name(atom): atom_mask(frame, atom) for atom in atoms}

    stable_atoms = []
    for atom in atoms:
        mask = masks[atom_name(atom)]
        stats = []
        stable = True
        for years in ((2020, 2021, 2022), (2023, 2024)):
            period = frame[mask & frame.year.isin(years)]
            if len(period) < 700 or period.signed_price_return.mean() <= 0:
                stable = False
                break
            stats.append(float(period.signed_price_return.mean()))
        if stable:
            stable_atoms.append((min(stats), atom))
    stable_atoms.sort(key=lambda item: item[0], reverse=True)
    shortlisted_atoms = [atom for _, atom in stable_atoms[:36]]
    policies = [()] + [(atom,) for atom in shortlisted_atoms]
    policies.extend(itertools.combinations(shortlisted_atoms, 2))
    candidates = []
    for policy in policies:
        mask = np.ones(len(frame), dtype=bool)
        for atom in policy:
            mask &= masks[atom_name(atom)]
        periods = {
            "selection_2020_2022": evaluate(
                records, mask, (2020, 2021, 2022), .03),
            "confirmation_2023_2024": evaluate(
                records, mask, (2023, 2024), .03),
        }
        admissible = all(
            value["n"] >= 600 and value["return_pct"] > 0
            and value["dd_pct"] <= 10
            and value["trades_per_calendar_day"] >= 3
            for value in periods.values())
        candidates.append({
            "atoms": list(policy), "development": periods,
            "development_floor_month_pct": round(month_floor(periods), 4),
            "development_negative_months": sum(
                value["negative_months"] for value in periods.values()),
            "development_worst_return_pct": min(
                value["return_pct"] for value in periods.values()),
            "development_max_dd_pct": max(
                value["dd_pct"] for value in periods.values()),
            "admissible": admissible,
        })
    candidates.sort(key=lambda item: (
        item["admissible"], item["development_floor_month_pct"],
        -item["development_negative_months"],
        item["development_worst_return_pct"],
        -item["development_max_dd_pct"]), reverse=True)
    selected = candidates[0]
    selected_mask = np.ones(len(frame), dtype=bool)
    for atom in selected["atoms"]:
        selected_mask &= masks[atom_name(atom)]
    holdouts = {
        f"margin_{int(100 * margin)}pct": {
            str(year): evaluate(records, selected_mask, (year,), margin)
            for year in (2025, 2026)
        } for margin in (.03, .05)
    }
    frame[selected_mask].to_pickle(ROWS_OUT)
    passed = bool(
        selected["admissible"]
        and all(value["return_pct"] > 0 and value["dd_pct"] <= 8
                and value["negative_months"] == 0
                and value["trades_per_calendar_day"] >= 3
                for value in holdouts["margin_3pct"].values()))
    report = {
        "status": ("LIQUIDITY_REFINEMENT_AWAITING_DIRECT_VENUE_FORWARD" if passed
                   else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "base": "frozen false-breakout fade, 1.5% target / 0.5% stop",
            "threshold_fit": "2020-2022 quantiles only",
            "policy_selection": [2020, 2021, 2022, 2023, 2024],
            "holdouts": [2025, 2026],
            "entry": "next 5m market open", "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "source_rows": len(source_frame), "enriched_rows": len(frame),
        "atom_count": len(atoms), "stable_atom_count": len(stable_atoms),
        "policy_count": len(candidates),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": selected,
        "top20_before_holdouts": candidates[:20],
        "holdouts": holdouts, "objective_pass": passed,
        "sources": {**candle_sources, str(SOURCE): digest(SOURCE),
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "The base strategy and bounded refinement grid create selection risk.",
            "Binance spot history still requires direct X-Perp transfer.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "stable_atom_count": len(stable_atoms),
        "policy_count": len(candidates),
        "admissible_count": report["admissible_count"],
        "selected": selected, "holdouts": holdouts,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
