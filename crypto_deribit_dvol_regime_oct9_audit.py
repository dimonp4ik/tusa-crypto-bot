"""Causal material-move crypto regimes conditioned on Deribit DVOL.

The bounded rule grid is declared in this file before evaluation.  It uses a
closed hourly BTC/ETH volatility-index candle, enters at the following Binance
perpetual 5-minute market open, and tests only literal 1.5% and 2.0% targets.
Later periods remain sealed unless every earlier economic gate passes.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as price
import crypto_futures_trade_flow_regime_oct7_audit as flow_source
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate
from src.research_profit_gate import economic_gate, minimum_binary_hit_rate_pct


ROOT = Path("reports/audit_2026_09_08")
DVOL_ROOT = ROOT / "crypto_deribit_dvol_history_oct9"
OUT = ROOT / "crypto_deribit_dvol_regime_oct9.json"
PROFILES = ("broad", "strict")
FAMILIES = (
    "compression_breakout", "expansion_continuation", "shock_reversal",
    "cooling_trend", "eth_btc_dispersion",
)
ROUTES = tuple((family, profile) for family in (*FAMILIES, "all_regimes")
               for profile in PROFILES)
CONTRACTS = (
    {"target_pct": 1.5, "stop_pct": .75, "hold_bars": 288},
    {"target_pct": 1.5, "stop_pct": 1.0, "hold_bars": 576},
    {"target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 576},
    {"target_pct": 2.0, "stop_pct": 1.0, "hold_bars": 864},
)
PERIODS = {
    "development_2021_2022": (pd.Timestamp("2021-05-01", tz="UTC"),
                              pd.Timestamp("2023-01-01", tz="UTC")),
    "validation_2023": (pd.Timestamp("2023-01-01", tz="UTC"),
                        pd.Timestamp("2024-01-01", tz="UTC")),
    "confirmation_2024": (pd.Timestamp("2024-01-01", tz="UTC"),
                          pd.Timestamp("2025-01-01", tz="UTC")),
    "test_2025": (pd.Timestamp("2025-01-01", tz="UTC"),
                  pd.Timestamp("2026-01-01", tz="UTC")),
    "untouched_2026": (pd.Timestamp("2026-01-01", tz="UTC"),
                       pd.Timestamp("2027-01-01", tz="UTC")),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dvol() -> tuple[dict[str, dict], dict[str, str]]:
    manifest_path = DVOL_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data, sources = {}, {str(manifest_path): digest(manifest_path)}
    for currency in ("BTC", "ETH"):
        item = manifest["currencies"][currency]
        path = Path(item["path"])
        raw = path.read_bytes()
        found = hashlib.sha256(raw).hexdigest()
        if found != item["sha256"]:
            raise ValueError(f"DVOL hash mismatch: {currency}")
        value = pickle.loads(raw)
        times = np.asarray(value["time"], dtype=np.int64)
        if len(times) < 720 or np.any(np.diff(times) != 3600):
            raise ValueError(f"DVOL is not gap-free hourly data: {currency}")
        data[currency] = value
        sources[str(path)] = found
    return data, sources


def dvol_frame(data: dict[str, dict]) -> pd.DataFrame:
    pieces = []
    for currency in ("BTC", "ETH"):
        raw = data[currency]
        frame = pd.DataFrame({
            "time": np.asarray(raw["time"], dtype=np.int64),
            "open": np.asarray(raw["open"], dtype=float),
            "high": np.asarray(raw["high"], dtype=float),
            "low": np.asarray(raw["low"], dtype=float),
            "close": np.asarray(raw["close"], dtype=float),
        })
        prefix = currency.lower()
        close = frame.close
        mean = close.rolling(720, min_periods=720).mean()
        std = close.rolling(720, min_periods=720).std().replace(0, np.nan)
        out = pd.DataFrame({
            "entry_time": frame.time + 3600,
            f"{prefix}_dvol": close,
            f"{prefix}_dvol_z720": (close - mean) / std,
            f"{prefix}_dvol_range": (frame.high - frame.low) / close,
        })
        for hours in (1, 6, 24, 72):
            out[f"{prefix}_dvol_ret{hours}"] = close / close.shift(hours) - 1
        # The row timestamp is the start of its hour.  Its close becomes usable
        # only at entry_time, exactly one hour later.
        pieces.append(out)
    merged = pieces[0].merge(pieces[1], on="entry_time", how="inner",
                             validate="one_to_one")
    spread = merged.eth_dvol - merged.btc_dvol
    mean = spread.rolling(720, min_periods=720).mean()
    std = spread.rolling(720, min_periods=720).std().replace(0, np.nan)
    merged["dvol_spread"] = spread
    merged["dvol_spread_z720"] = (spread - mean) / std
    merged["dvol_spread_delta6"] = spread - spread.shift(6)
    return merged.replace([np.inf, -np.inf], np.nan).dropna()


def strict_hourly_price(candles: dict[str, dict]) -> pd.DataFrame:
    frame = price.build_features(candles)
    frame = frame[frame.entry_time.mod(3600).eq(0)].copy()
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        times = np.asarray(candles[symbol]["time"], dtype=np.int64)
        index = part.entry_index.to_numpy(np.int64)
        signal = index - 1
        valid = signal >= 288
        safe = np.maximum(signal - 288, 0)
        valid &= times[signal] - times[safe] == 288 * 300
        pieces.append(part.loc[valid])
    return pd.concat(pieces, ignore_index=True)


def build_events(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    def add(family: str, profile: str, direction: pd.Series,
            mask: pd.Series, strength: pd.Series) -> None:
        selected = mask & direction.ne(0)
        part = frame.loc[selected].copy()
        part["direction"] = direction.loc[selected].astype(np.int8)
        part["family"], part["profile"] = family, profile
        part["semantic_score"] = strength.loc[selected].astype(float)
        pieces.append(part)

    for profile in PROFILES:
        strict = profile == "strict"

        direction = np.sign(frame.ret12)
        threshold = .004 if strict else .002
        mask = (
            frame.btc_dvol_z720.le(-.8 if strict else -.4)
            & frame.btc_dvol_ret6.ge(.015 if strict else .005)
            & frame.ret12.abs().ge(threshold)
            & (frame.ret72 * direction).gt(0)
            & frame.range_ratio.ge(1.10 if strict else .90)
        )
        strength = (-frame.btc_dvol_z720 + 25 * frame.btc_dvol_ret6
                    + 100 * frame.ret12.abs())
        add("compression_breakout", profile, direction, mask, strength)

        direction = np.sign(frame.btc_ret72)
        directional_breadth = frame.breadth72.where(
            direction > 0, 1 - frame.breadth72)
        mask = (
            frame.btc_dvol_ret6.ge(.04 if strict else .02)
            & frame.btc_ret72.abs().ge(.008 if strict else .004)
            & (frame.ret12 * direction).ge(.002 if strict else .0005)
            & directional_breadth.ge(.70 if strict else .60)
        )
        strength = (20 * frame.btc_dvol_ret6 + 100 * frame.btc_ret72.abs()
                    + directional_breadth)
        add("expansion_continuation", profile, direction, mask, strength)

        direction = -np.sign(frame.ret12)
        rejection = frame.lower_wick.where(direction > 0, frame.upper_wick)
        shock = (frame.btc_dvol_z720.ge(1.5 if strict else 1.0)
                 | frame.btc_dvol_ret6.ge(.15 if strict else .08))
        mask = (
            shock & frame.ret12.abs().ge(.010 if strict else .005)
            & rejection.ge(.35 if strict else .22)
            & (frame.body1 * direction).gt(0)
        )
        strength = (frame.btc_dvol_z720.clip(lower=0)
                    + 10 * frame.btc_dvol_ret6.clip(lower=0)
                    + 100 * frame.ret12.abs() + rejection)
        add("shock_reversal", profile, direction, mask, strength)

        direction = np.sign(frame.ret288)
        mask = (
            frame.btc_dvol_ret6.le(-.025 if strict else -.01)
            & frame.ret288.abs().ge(.02 if strict else .01)
            & (frame.ret72 * direction).ge(.006 if strict else .002)
            & (frame.ret12 * direction).between(-.001, .008)
            & (frame.body1 * direction).gt(0)
        )
        strength = (-20 * frame.btc_dvol_ret6 + 100 * frame.ret288.abs()
                    + 50 * (frame.ret72 * direction))
        add("cooling_trend", profile, direction, mask, strength)

        direction = np.sign(frame.cross_z72)
        directional_breadth = frame.breadth72.where(
            direction > 0, 1 - frame.breadth72)
        dispersion = frame.dvol_spread_delta6 * direction
        mask = (
            ~frame.symbol.isin(("BTCUSDT", "ETHUSDT"))
            & frame.cross_z72.abs().ge(1.2 if strict else .8)
            & dispersion.le(-1.0 if strict else -.35)
            & directional_breadth.ge(.625 if strict else .50)
            & (frame.ret12 * direction).gt(0)
        )
        strength = (frame.cross_z72.abs() - dispersion / 2
                    + directional_breadth)
        add("eth_btc_dispersion", profile, direction, mask, strength)

    events = pd.concat(pieces, ignore_index=True)
    return events.sort_values(
        ["entry_time", "semantic_score", "symbol"],
        ascending=[True, False, True]).reset_index(drop=True)


@nb.njit(cache=True)
def resolve(open_, high, low, close, times, indices, directions,
            target_pct, stop_pct, max_bars):
    n = len(indices)
    signed = np.empty(n, dtype=np.float64)
    exits = np.empty(n, dtype=np.int64)
    hits = np.zeros(n, dtype=np.int8)
    reasons = np.empty(n, dtype=np.int8)
    target_move, stop_move = target_pct / 100.0, stop_pct / 100.0
    for row in range(n):
        first, direction = indices[row], directions[row]
        entry = open_[first]
        target = entry * (1.0 + direction * target_move)
        stop = entry * (1.0 - direction * stop_move)
        result, reason, last = 0.0, 2, first
        resolved = False
        for j in range(first, first + max_bars):
            last = j
            stop_hit = low[j] <= stop if direction > 0 else high[j] >= stop
            target_hit = high[j] >= target if direction > 0 else low[j] <= target
            if stop_hit:  # pessimistic ordering when both touch in one candle
                fill = min(open_[j], stop) if direction > 0 else max(open_[j], stop)
                result = direction * (fill - entry) / entry
                reason, resolved = 0, True
                break
            if target_hit:
                result, hits[row], reason, resolved = target_move, 1, 1, True
                break
        if not resolved:
            result = direction * (close[last] - entry) / entry
            result = max(-stop_move, min(target_move, result))
        signed[row], exits[row], reasons[row] = result, times[last] + 300, reason
    return signed, exits, hits, reasons


def add_outcomes(events: pd.DataFrame, candles: dict, contract: dict) \
        -> tuple[pd.DataFrame, int]:
    pieces, excluded = [], 0
    max_bars = int(contract["hold_bars"])
    for symbol, part in events.groupby("symbol", sort=False):
        candle = candles[symbol]
        times = np.asarray(candle["time"], dtype=np.int64)
        index = part.entry_index.to_numpy(np.int64)
        eligible = index + max_bars <= len(times)
        end = np.minimum(index + max_bars - 1, len(times) - 1)
        eligible &= times[end] - times[index] == (max_bars - 1) * 300
        excluded += int((~eligible).sum())
        found = part.loc[eligible].copy()
        if found.empty:
            continue
        index = found.entry_index.to_numpy(np.int64)
        arrays = [np.asarray(candle[name], dtype=float)
                  for name in ("open", "high", "low", "close")]
        signed, exits, hits, reasons = resolve(
            *arrays, times, index, found.direction.to_numpy(np.int8),
            float(contract["target_pct"]), float(contract["stop_pct"]),
            max_bars)
        found["signed_return"], found["exit_time"] = signed, exits
        found["target_hit"], found["exit_reason"] = hits.astype(bool), reasons
        pieces.append(found)
    return pd.concat(pieces, ignore_index=True), excluded


def select_route(frame: pd.DataFrame, family: str, profile: str) -> pd.DataFrame:
    selected = frame[frame.profile.eq(profile)]
    if family != "all_regimes":
        selected = selected[selected.family.eq(family)]
    # A symbol/hour/direction may be proposed by several regimes.  Keep the
    # strongest causal explanation so the union cannot duplicate a trade.
    return selected.sort_values("semantic_score", ascending=False).drop_duplicates(
        ["symbol", "entry_time", "direction"])


def period_rows(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    start, end = PERIODS[period]
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return frame[(stamp >= start) & (stamp < end)]


def evaluate(frame: pd.DataFrame, period: str) -> dict:
    selected = period_rows(frame, period).sort_values(
        ["entry_time", "semantic_score", "symbol"],
        ascending=[True, False, True]).copy()
    selected["source_priority"] = selected.groupby("entry_time").cumcount()
    rows = []
    for row in selected.to_dict("records"):
        rows.append({
            **row,
            "direction": "LONG" if row["direction"] > 0 else "SHORT",
            "source": f"dvol_{row['family']}",
            "module": "dvol_material_router", "module_priority": 0,
            "outcome": "TP" if row["signed_return"] > 0 else "SL",
            "signed_price_return": row["signed_return"],
        })
    accepted = runtime_gate(rows)
    result = money(accepted, .03)
    n = len(accepted)
    result.update({
        "literal_target_hits": sum(bool(row["target_hit"]) for row in accepted),
        "literal_target_hit_rate": round(
            100 * sum(bool(row["target_hit"]) for row in accepted) / n, 4)
            if n else 0.0,
        "stops": sum(int(row["exit_reason"]) == 0 for row in accepted),
        "timeouts": sum(int(row["exit_reason"]) == 2 for row in accepted),
        "months": len(result.get("monthly_return_pct", {})),
        "months_ge_30pct": sum(
            value >= 30 for value in result.get("monthly_return_pct", {}).values()),
    })
    observed = float(result.get("trades_per_calendar_day", 0.0) or 0.0)
    result["economic_gate"] = economic_gate(
        [float(row["signed_price_return"]) for row in accepted], observed) \
        if accepted else {
            "n": 0, "observed_trades_per_day": 0.0,
            "projected_monthly_return_pct": 0.0,
            "frequency_pass": False, "return_pass": False, "passes": False,
        }
    return result


def gate(value: dict, minimum_n: int) -> bool:
    return bool(value["n"] >= minimum_n and value["return_pct"] > 0
                and value["dd_pct"] <= 8 and value["negative_months"] == 0
                and value["economic_gate"]["passes"])


def public(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "_frame"}


def main() -> None:
    candles, price_sources = flow_source.load()
    dvol, dvol_sources = load_dvol()
    hourly = strict_hourly_price(candles)
    context = dvol_frame(dvol)
    enriched = hourly.merge(context, on="entry_time", how="inner",
                            validate="many_to_one")
    events = build_events(enriched)

    frontier, survivors, exclusions = [], [], []
    for contract in CONTRACTS:
        outcomes, excluded = add_outcomes(events, candles, contract)
        exclusions.append({**contract, "excluded_crossing_gap_or_end": excluded})
        for family, profile in ROUTES:
            route = select_route(outcomes, family, profile)
            development = evaluate(route, "development_2021_2022")
            item = {**contract, "family": family, "profile": profile,
                    "development_2021_2022": development}
            frontier.append(item)
            if not gate(development, 600):
                continue
            validation = evaluate(route, "validation_2023")
            item["validation_2023"] = validation
            if not gate(validation, 300):
                continue
            item["score"] = round(
                development["return_pct"] + 2 * validation["return_pct"]
                - development["dd_pct"] - 2 * validation["dd_pct"], 8)
            item["_frame"] = route
            survivors.append(item)

    survivors.sort(key=lambda item: item["score"], reverse=True)
    frozen = survivors[0] if survivors else None
    confirmation = test = reveal = None
    if frozen:
        confirmation = evaluate(frozen["_frame"], "confirmation_2024")
        if gate(confirmation, 300):
            test = evaluate(frozen["_frame"], "test_2025")
        if test and gate(test, 300):
            reveal = evaluate(frozen["_frame"], "untouched_2026")

    frontier.sort(key=lambda item: (
        item["development_2021_2022"]["economic_gate"][
            "projected_monthly_return_pct"],
        item["development_2021_2022"]["return_pct"]), reverse=True)
    passed_all = bool(reveal and gate(reveal, 200))
    status = ("RESEARCH_CANDIDATE_AWAITING_DIRECT_FORWARD"
              if passed_all else "REJECTED")
    report = {
        "status": status,
        "protocol": {
            "development": "2021-05 through 2022",
            "validation": "2023, opened only after development gate",
            "confirmation": "2024, opened only after validation gate",
            "test": "2025, opened only after confirmation gate",
            "untouched_reveal": "2026, opened only after test gate",
            "entry": "market open after closed hourly DVOL and 5m price bars",
            "families": list(FAMILIES), "profiles": list(PROFILES),
            "contracts": list(CONTRACTS),
            "money": "3% current-equity margin, fixed 10x leverage",
            "execution_rails": "max 2 new trades/rolling hour; one position/symbol; 3h same-direction cooldown; pause after 3 daily losses",
            "fees": "excluded by user instruction",
        },
        "rows": {"strict_hourly_price": len(hourly),
                 "dvol_context": len(context), "enriched": len(enriched),
                 "semantic_events": len(events),
                 "outcome_exclusions": exclusions},
        "minimum_ideal_hit_rate_for_30pct_month": {
            f"tp{item['target_pct']}_sl{item['stop_pct']}": round(
                minimum_binary_hit_rate_pct(
                    item["target_pct"], item["stop_pct"]), 6)
            for item in CONTRACTS
        },
        "qualified_routes_before_2024": len(survivors),
        "frozen_before_2024": public(frozen) if frozen else None,
        "confirmation_2024": confirmation, "test_2025": test,
        "untouched_2026": reveal,
        "development_frontier": [public(item) for item in frontier[:30]],
        "sources": {**price_sources, **dvol_sources,
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "DVOL public history begins in April 2021, so 2020 is unavailable.",
            "Hourly decisions can miss intrahour regime changes.",
            "A surviving historical route would still require shadow forward evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "rows": report["rows"],
        "qualified_routes_before_2024": len(survivors),
        "frozen_before_2024": report["frozen_before_2024"],
        "confirmation_2024": confirmation, "test_2025": test,
        "untouched_2026": reveal,
        "development_frontier": report["development_frontier"][:8],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
