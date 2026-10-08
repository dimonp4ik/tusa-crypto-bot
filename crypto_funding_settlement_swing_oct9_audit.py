"""Causal funding-settlement swing routes with literal 1.5-2% targets.

This is distinct from the old 0.5% funding feature model.  A candidate exists
only once per published funding event, uses a fully closed five-minute candle
after that event for confirmation, and enters at the following market open.
One asset represents each direction/event to suppress common-factor bursts.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

import crypto_okx_swap_1h_idiosyncratic_router_oct9_audit as gate_helpers


ROOT = Path("reports/audit_2026_09_08/crypto_futures_5m_funding_history")
MANIFEST = ROOT / "manifest.json"
OUT = Path("reports/audit_2026_09_08/crypto_funding_settlement_swing_oct9.json")
SELECTION_YEARS = (2020, 2021)
VALIDATION_YEAR = 2022
CONFIRMATION_YEAR = 2023
HIDDEN_YEARS = (2024, 2025, 2026)


@dataclass(frozen=True)
class Config:
    family: str
    extremity: float
    minimum_move: float
    target_pct: float

    @property
    def stop_pct(self) -> float:
        return .75 if self.target_pct == 1.5 else 1.0

    @property
    def hold_bars(self) -> int:
        return 288 if self.target_pct == 1.5 else 576

    @property
    def name(self) -> str:
        return (f"{self.family}|extreme={self.extremity}|"
                f"move={self.minimum_move}|target={self.target_pct}")


CONFIGS = tuple(
    Config(family, extremity, move, target)
    for family in ("crowded_reversal", "crowded_continuation")
    for extremity in (1.5, 2.0)
    for move in (.01, .02)
    for target in (1.5, 2.0)
) + tuple(
    Config("uncrowded_trend", extremity, move, target)
    for extremity in (0.0, .5)
    for move in (.02, .04)
    for target in (1.5, 2.0)
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def load() -> tuple[dict, dict, dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candles, funding = {}, {}
    sources = {str(MANIFEST): digest(MANIFEST)}
    coverage = {}
    for symbol, item in manifest["symbols"].items():
        candle_path = resolve_path(item["candles"]["path"])
        funding_path = resolve_path(item["funding"]["path"])
        candle_raw, funding_raw = candle_path.read_bytes(), funding_path.read_bytes()
        if hashlib.sha256(candle_raw).hexdigest() != item["candles"]["sha256"]:
            raise ValueError(f"candle hash mismatch: {symbol}")
        if hashlib.sha256(funding_raw).hexdigest() != item["funding"]["sha256"]:
            raise ValueError(f"funding hash mismatch: {symbol}")
        candle = pickle.loads(candle_raw)
        event = pickle.loads(funding_raw)
        times = [int(value) for value in candle["time"]]
        gaps = [(int(left), int(right), int(right - left))
                for left, right in zip(times, times[1:]) if right - left != 300]
        if any(right <= left for left, right in zip(event["time"], event["time"][1:])):
            raise ValueError(f"non-increasing funding timestamps: {symbol}")
        candles[symbol], funding[symbol] = candle, event
        coverage[symbol] = {
            "bars": len(times), "start": times[0], "end": times[-1],
            "gap_count": len(gaps), "gaps": gaps,
            "signals_crossing_any_gap_are_excluded": True,
        }
        sources[str(candle_path)] = hashlib.sha256(candle_raw).hexdigest()
        sources[str(funding_path)] = hashlib.sha256(funding_raw).hexdigest()
    return candles, funding, sources, coverage


def event_features(symbol: str, candle: dict, funding: dict) -> pd.DataFrame:
    times = np.asarray(candle["time"], dtype=np.int64)
    open_ = np.asarray(candle["open"], dtype=float)
    high = np.asarray(candle["high"], dtype=float)
    low = np.asarray(candle["low"], dtype=float)
    close = np.asarray(candle["close"], dtype=float)
    volume = np.asarray(candle["volume"], dtype=float)
    lookup = {int(timestamp): index for index, timestamp in enumerate(times)}
    events = pd.DataFrame({
        "funding_time": np.asarray(funding["time"], dtype=np.int64),
        "funding_rate": np.asarray(funding["rate"], dtype=float),
    }).sort_values("funding_time")
    mean90 = events.funding_rate.rolling(90, min_periods=60).mean()
    std90 = events.funding_rate.rolling(90, min_periods=60).std(ddof=0).replace(0, np.nan)
    events["funding_z90"] = (events.funding_rate - mean90) / std90
    events["funding_mean9"] = events.funding_rate.rolling(9, min_periods=6).mean()
    rows = []
    for event in events.dropna().to_dict("records"):
        # The bar opening at the funding timestamp is allowed to close first;
        # the strategy then enters at the next 5m open.
        signal_index = lookup.get(int(event["funding_time"]))
        if signal_index is None or signal_index < 864 or signal_index + 1 >= len(times):
            continue
        entry_index = signal_index + 1
        if times[entry_index] - times[signal_index] != 300:
            continue
        previous24 = lookup.get(int(times[signal_index] - 24 * 3600))
        previous72 = lookup.get(int(times[signal_index] - 72 * 3600))
        if (previous24 is None or previous72 is None
                or signal_index - previous24 != 288
                or signal_index - previous72 != 864):
            continue
        ret24 = close[signal_index] / close[previous24] - 1
        ret72 = close[signal_index] / close[previous72] - 1
        body = close[signal_index] / open_[signal_index] - 1
        candle_range = high[signal_index] - low[signal_index]
        location = ((close[signal_index] - low[signal_index]) / candle_range
                    if candle_range > 0 else .5)
        previous_volume = volume[signal_index - 288:signal_index]
        median_volume = float(np.median(previous_volume))
        volume_ratio = (volume[signal_index] / median_volume
                        if median_volume > 0 else np.nan)
        stamp = pd.Timestamp(int(times[entry_index]), unit="s", tz="UTC")
        rows.append({
            "symbol": symbol, "entry_index": entry_index,
            "entry_time": int(times[entry_index]),
            "year": int(stamp.year), "month": int(stamp.year * 100 + stamp.month),
            "funding_time": int(event["funding_time"]),
            "funding_rate": float(event["funding_rate"]),
            "funding_z90": float(event["funding_z90"]),
            "funding_mean9": float(event["funding_mean9"]),
            "ret24": float(ret24), "ret72": float(ret72),
            "confirmation_body": float(body),
            "close_location": float(location),
            "volume_ratio": float(volume_ratio),
        })
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()


def candidates(features: pd.DataFrame, config: Config) -> pd.DataFrame:
    funding_side = np.sign(features.funding_rate)
    price_side = np.sign(features.ret24)
    if config.family == "crowded_reversal":
        direction = -funding_side
        mask = (funding_side.ne(0)
                & features.funding_z90.abs().ge(config.extremity)
                & features.funding_rate.abs().ge(.00005)
                & (features.ret24 * funding_side).ge(config.minimum_move)
                & (features.confirmation_body * direction).gt(0)
                & features.volume_ratio.ge(.7))
        strength = (features.funding_z90.abs()
                    + 30 * features.ret24.abs()
                    + 10 * (features.confirmation_body * direction).clip(lower=0))
    elif config.family == "crowded_continuation":
        direction = funding_side
        mask = (funding_side.ne(0)
                & features.funding_z90.abs().ge(config.extremity)
                & features.funding_rate.abs().ge(.00005)
                & (features.ret24 * direction).ge(config.minimum_move)
                & (features.ret72 * direction).gt(0)
                & (features.confirmation_body * direction).gt(0)
                & features.volume_ratio.ge(.7))
        strength = (features.funding_z90.abs()
                    + 30 * (features.ret24 * direction)
                    + 10 * (features.confirmation_body * direction).clip(lower=0))
    else:
        direction = price_side
        directed_funding_z = features.funding_z90 * direction
        mask = (price_side.ne(0)
                & features.ret24.abs().ge(config.minimum_move)
                & (features.ret72 * direction).gt(0)
                & directed_funding_z.le(-config.extremity)
                & (features.funding_rate * direction).le(0)
                & (features.confirmation_body * direction).gt(0)
                & features.volume_ratio.ge(.7))
        strength = (40 * features.ret24.abs() - directed_funding_z
                    + 10 * (features.confirmation_body * direction).clip(lower=0))
    found = features.loc[mask].copy()
    found["direction"] = direction.loc[mask].astype(np.int8)
    found["route"] = config.family
    found["config"] = config.name
    found["setup_score"] = strength.loc[mask].astype(float)
    found["selected_target_pct"] = config.target_pct
    found["stop_pct"] = config.stop_pct
    found["hold_bars"] = config.hold_bars
    # Funding events are common timestamps across markets.  Keep one asset per
    # side of that event to avoid treating one derivatives shock as many bets.
    return (found.sort_values(
        ["funding_time", "direction", "setup_score", "symbol"],
        ascending=[True, True, False, True])
        .drop_duplicates(["funding_time", "direction"], keep="first")
        .reset_index(drop=True))


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, entries, directions,
            target_pct, stop_pct, hold_bars):
    count = len(entries)
    signed = np.empty(count, dtype=np.float64)
    exits = np.empty(count, dtype=np.int64)
    hits = np.zeros(count, dtype=np.int8)
    reasons = np.empty(count, dtype=np.int8)
    target_move, stop_move = target_pct / 100, stop_pct / 100
    for row in range(count):
        first, direction = entries[row], directions[row]
        entry = open_[first]
        target = entry * (1 + direction * target_move)
        stop = entry * (1 - direction * stop_move)
        last, value, reason, done = first, 0.0, 2, False
        end = min(len(times), first + hold_bars)
        for index in range(first, end):
            if index > first and times[index] - times[index - 1] != 300:
                break
            last = index
            stop_hit = low[index] <= stop if direction > 0 else high[index] >= stop
            target_hit = high[index] >= target if direction > 0 else low[index] <= target
            if stop_hit:
                fill = min(open_[index], stop) if direction > 0 else max(open_[index], stop)
                value, reason, done = direction * (fill / entry - 1), 0, True
                break
            if target_hit:
                value, hits[row], reason, done = target_move, 1, 1, True
                break
        if not done:
            value = direction * (close[last] / entry - 1)
            value = max(-stop_move, min(target_move, value))
        signed[row], exits[row], reasons[row] = value, times[last] + 300, reason
    return signed, exits, hits, reasons


def add_outcomes(frame: pd.DataFrame, candles: dict, config: Config) -> pd.DataFrame:
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        candle = candles[symbol]
        times = np.asarray(candle["time"], dtype=np.int64)
        valid = part.entry_index + config.hold_bars <= len(times)
        if bool(valid.any()):
            indices = part.entry_index.to_numpy(np.int64)
            last = np.minimum(indices + config.hold_bars - 1, len(times) - 1)
            continuous = (times[last] - times[indices]
                          == (config.hold_bars - 1) * 300)
            valid = valid.to_numpy() & continuous
        part = part.loc[valid].copy()
        if part.empty:
            continue
        values, exits, hits, reasons = resolve(
            *[np.asarray(candle[field], dtype=float)
              for field in ("open", "high", "low", "close")],
            times,
            part.entry_index.to_numpy(np.int64),
            part.direction.to_numpy(np.int8),
            config.target_pct, config.stop_pct, config.hold_bars)
        found = part.copy()
        found["signed_return"], found["exit_time"] = values, exits
        found["target_hit"], found["exit_reason"] = hits.astype(bool), reasons
        found["quality"] = 1.0
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.copy()


def main() -> None:
    candles, funding, sources, coverage = load()
    feature_frame = pd.concat([
        event_features(symbol, candles[symbol], funding[symbol])
        for symbol in sorted(candles)
    ], ignore_index=True)
    evaluated = []
    for config in CONFIGS:
        proposed = candidates(feature_frame, config)
        if proposed.empty:
            continue
        frame = add_outcomes(proposed, candles, config)
        selection = gate_helpers.evaluate(frame, SELECTION_YEARS)
        selection_pass = gate_helpers.screen(
            selection, config.target_pct, minimum_n=80)
        item = {
            "config": config.name, "parameters": asdict(config),
            "stop_pct": config.stop_pct, "hold_bars": config.hold_bars,
            "rows": len(frame), "selection_2020_2021": selection,
            "selection_screen_pass": selection_pass, "_frame": frame,
        }
        if selection_pass:
            validation = gate_helpers.evaluate(frame, (VALIDATION_YEAR,))
            item["validation_2022"] = validation
            item["validation_screen_pass"] = gate_helpers.screen(
                validation, config.target_pct, minimum_n=30)
            if item["validation_screen_pass"]:
                confirmation = gate_helpers.evaluate(frame, (CONFIRMATION_YEAR,))
                item["confirmation_2023"] = confirmation
                item["confirmation_screen_pass"] = gate_helpers.screen(
                    confirmation, config.target_pct, minimum_n=30)
        evaluated.append(item)

    developed = [item for item in evaluated if item["selection_screen_pass"]]
    validated = [item for item in developed
                 if item.get("validation_screen_pass", False)]
    confirmed = [item for item in validated
                 if item.get("confirmation_screen_pass", False)]
    confirmed.sort(key=lambda item: gate_helpers.rank([
        item["selection_2020_2021"], item["validation_2022"],
        item["confirmation_2023"]]), reverse=True)
    union_frame = None
    union = None
    if confirmed:
        union_frame = pd.concat([item["_frame"] for item in confirmed],
                                ignore_index=True, sort=False)
        union_frame = (union_frame.sort_values(
            ["entry_time", "direction", "selected_target_pct", "setup_score"],
            ascending=[True, True, False, False])
            .drop_duplicates(["entry_time", "direction"], keep="first"))
        union = {
            "selection_2020_2021": gate_helpers.evaluate(union_frame, SELECTION_YEARS),
            "validation_2022": gate_helpers.evaluate(union_frame, (VALIDATION_YEAR,)),
            "confirmation_2023": gate_helpers.evaluate(union_frame, (CONFIRMATION_YEAR,)),
        }
    early_objective = bool(union and all(
        gate_helpers.objective_pass(value) for value in union.values()))
    later = None
    if early_objective:
        later = {str(year): gate_helpers.evaluate(union_frame, (year,))
                 for year in HIDDEN_YEARS}
    later_objective = bool(later and all(
        gate_helpers.objective_pass(value) for value in later.values()))
    status = ("OBJECTIVE_PASS" if later_objective
              else "REJECTED_LATER" if later is not None
              else "REJECTED_EARLY_OBJECTIVE" if confirmed
              else "REJECTED_CONFIRMATION" if validated
              else "REJECTED_VALIDATION" if developed
              else "REJECTED_SELECTION")
    evaluated.sort(key=lambda item: gate_helpers.rank([
        item["selection_2020_2021"]]), reverse=True)
    public = lambda item: {key: value for key, value in item.items()
                           if key != "_frame"}
    family_best = {}
    for family in ("crowded_reversal", "crowded_continuation",
                   "uncrowded_trend"):
        members = [item for item in evaluated
                   if item["parameters"]["family"] == family]
        if members:
            family_best[family] = public(members[0])
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "venue": "Binance UM perpetual", "bar": "5m",
            "entry": "market open after one fully closed post-funding bar",
            "selection": list(SELECTION_YEARS), "validation": VALIDATION_YEAR,
            "confirmation": CONFIRMATION_YEAR,
            "hidden_revealed_only_after_complete_early_objective": list(HIDDEN_YEARS),
            "targets_pct": [1.5, 2.0], "stops_pct": [.75, 1.0],
            "hold_hours": {"1.5": 24, "2.0": 48},
            "episode_rule": "one strongest asset per direction and funding event",
            "money": "3% current equity margin, fixed 10x gross",
            "fees": "excluded by user instruction",
        },
        "funding_event_feature_rows": len(feature_frame),
        "integrity": {"candle_coverage": coverage},
        "tested_routes": len(evaluated),
        "selection_passed": len(developed),
        "validation_passed": len(validated),
        "confirmation_passed": len(confirmed),
        "best_by_family": family_best,
        "top_routes": [public(item) for item in evaluated[:12]],
        "confirmed_routes": [public(item) for item in confirmed],
        "confirmed_union": union,
        "early_objective_pass": early_objective,
        "later": later, "later_objective_pass": later_objective,
        "decision": "No runtime change unless every chronological and economic gate passes.",
        "sources": {**sources, str(Path(__file__)): digest(Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "funding_event_feature_rows": len(feature_frame),
        "tested_routes": len(evaluated),
        "selection_passed": len(developed),
        "validation_passed": len(validated),
        "confirmation_passed": len(confirmed),
        "top_routes": [public(item) for item in evaluated[:5]],
        "confirmed_union": union, "later": later,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
