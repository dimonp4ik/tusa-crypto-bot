"""Chronological idiosyncratic-move router with literal 1.5-2% targets.

The routes are explicit market hypotheses rather than variants of the frozen
classifier.  At most one asset per direction and entry hour represents a
cross-market episode, which prevents a common BTC move from becoming a burst
of correlated positions.  Selection uses 2020-2021, validation uses 2022, and
confirmation uses 2023.  Later years stay unopened unless all three pass.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_5m_regime_router_literal_oct6_audit as portfolio
import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_target_cascade_oct7_audit as target_engine
from src.research_profit_gate import economic_gate


ROOT = Path("reports/audit_2026_09_08")
MANIFEST = ROOT / "crypto_okx_swap_1h_expanded_oct8" / "manifest.json"
OUT = ROOT / "crypto_okx_swap_1h_idiosyncratic_router_oct9.json"
SELECTION_YEARS = (2020, 2021)
VALIDATION_YEAR = 2022
CONFIRMATION_YEAR = 2023
HIDDEN_YEARS = (2024, 2025, 2026)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_route(routes: list[pd.DataFrame], frame: pd.DataFrame, *,
              name: str, config: str, direction: pd.Series,
              mask: pd.Series, score: pd.Series, target: float) -> None:
    selected = mask & direction.ne(0) & score.replace(
        [np.inf, -np.inf], np.nan).notna()
    found = frame.loc[selected].copy()
    if found.empty:
        return
    found["direction"] = direction.loc[selected].astype(np.int8)
    found["route"] = name
    found["config"] = config
    found["idio_score"] = score.loc[selected].astype(float)
    found["selected_target_pct"] = float(target)
    # One representative per direction and hour.  This is a signal semantic,
    # not position sizing: one common market episode is one hypothesis.
    found = (found.sort_values(
        ["entry_time", "direction", "idio_score", "symbol"],
        ascending=[True, True, False, True])
        .drop_duplicates(["entry_time", "direction"], keep="first"))
    routes.append(found)


def route_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    routes: list[pd.DataFrame] = []

    for z24 in (1.0, 1.5):
        for z6 in (0.0, .5):
            for minimum_return in (.005, .01):
                direction = np.sign(frame.cross_z24)
                directed_z24 = frame.cross_z24 * direction
                directed_z6 = frame.cross_z6 * direction
                score = (directed_z24 + .5 * directed_z6
                         + 10 * frame.ret24 * direction
                         + .15 * frame.volume_ratio)
                mask = (directed_z24.ge(z24) & directed_z6.ge(z6)
                        & (frame.ret24 * direction).ge(minimum_return)
                        & (frame.ret72 * direction).ge(0)
                        & (frame.body1 * direction).gt(0)
                        & frame.eff24.ge(.12) & frame.volume_ratio.ge(.8))
                config = f"z24={z24}|z6={z6}|ret24={minimum_return}"
                add_route(routes, frame, name="relative_persistence",
                          config=config, direction=direction, mask=mask,
                          score=score, target=2.0)

    for z6 in (1.5, 2.0):
        for z24 in (.5, 1.0):
            for volume in (.8, 1.2):
                direction = np.sign(frame.cross_z6)
                directed_z6 = frame.cross_z6 * direction
                directed_z24 = frame.cross_z24 * direction
                breakout = frame.breakout_up.where(
                    direction > 0, frame.breakout_down)
                score = (directed_z6 + .5 * directed_z24
                         + 30 * breakout + .2 * frame.range_ratio)
                mask = (directed_z6.ge(z6) & directed_z24.ge(z24)
                        & breakout.ge(-.0015)
                        & (frame.body1 * direction).gt(0)
                        & frame.range_ratio.ge(1.1)
                        & frame.volume_ratio.ge(volume))
                config = f"z6={z6}|z24={z24}|volume={volume}"
                add_route(routes, frame, name="relative_breakout",
                          config=config, direction=direction, mask=mask,
                          score=score, target=1.5)

    for z72 in (1.0, 1.5):
        for z24 in (.25, .75):
            direction = np.sign(frame.cross_z72)
            directed_z72 = frame.cross_z72 * direction
            directed_z24 = frame.cross_z24 * direction
            directed_ret6 = frame.ret6 * direction
            score = (directed_z72 + .5 * directed_z24
                     - 15 * directed_ret6 + .1 * frame.volume_ratio)
            mask = (directed_z72.ge(z72) & directed_z24.ge(z24)
                    & (frame.ret168 * direction).ge(.015)
                    & directed_ret6.between(-.018, -.001)
                    & (frame.body1 * direction).gt(0)
                    & frame.volume_ratio.ge(.7))
            config = f"z72={z72}|z24={z24}"
            add_route(routes, frame, name="relative_pullback",
                      config=config, direction=direction, mask=mask,
                      score=score, target=1.5)

    for z6 in (1.5, 2.0):
        for wick in (.25, .4):
            extreme = np.sign(frame.cross_z6)
            direction = -extreme
            rejection = frame.lower_wick.where(
                direction > 0, frame.upper_wick)
            score = (frame.cross_z6.abs() + rejection
                     + .2 * frame.range_ratio)
            mask = (frame.cross_z6.abs().ge(z6)
                    & (frame.ret6 * extreme).ge(.008)
                    & rejection.ge(wick)
                    & (frame.body1 * direction).gt(0)
                    & frame.range_ratio.ge(1.0))
            config = f"z6={z6}|wick={wick}"
            add_route(routes, frame, name="relative_exhaustion",
                      config=config, direction=direction, mask=mask,
                      score=score, target=1.5)

    output = {}
    for found in routes:
        output[str(found.config.iloc[0]) + "|" + str(found.route.iloc[0])] = found
    return output


def add_outcomes(frame: pd.DataFrame, candles: dict) -> pd.DataFrame:
    pieces = []
    for target, found in frame.groupby("selected_target_pct", sort=True):
        part = found.reset_index(drop=True).copy()
        target_engine.add_target(part, candles, float(target))
        suffix = str(float(target)).replace(".", "_")
        part["signed_return"] = part[f"signed_{suffix}"]
        part["exit_time"] = part[f"exit_{suffix}"]
        part["target_hit"] = part[f"hit_{suffix}"]
        part["exit_reason"] = part[f"reason_{suffix}"]
        part["quality"] = 1.0
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.copy()


def evaluate(frame: pd.DataFrame, years: tuple[int, ...]) -> dict:
    metrics = portfolio.evaluate(frame, years, .5)
    rows = portfolio.portfolio_rows(frame, years, .5)
    gate = economic_gate(
        [float(row["signed_price_return"]) for row in rows],
        metrics["trades_per_calendar_day"], market_days=30)
    return {"portfolio": metrics, "economic_gate": gate}


def screen(value: dict, target: float, minimum_n: int) -> bool:
    metrics, gate = value["portfolio"], value["economic_gate"]
    minimum_hit = 50.0 if target == 2.0 else 58.0
    return bool(metrics["n"] >= minimum_n
                and metrics["literal_target_hit_rate"] >= minimum_hit
                and metrics["return_pct"] > 0
                and metrics["dd_pct"] <= 8
                and gate["projected_monthly_return_pct"] >= 3)


def rank(values: list[dict]) -> tuple[float, float, float, float]:
    projected = [item["economic_gate"]["projected_monthly_return_pct"]
                 for item in values]
    hits = [item["portfolio"]["literal_target_hit_rate"] for item in values]
    drawdowns = [item["portfolio"]["dd_pct"] for item in values]
    return min(projected), min(hits), -max(drawdowns), sum(projected)


def public_route(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "_frame"}


def objective_pass(value: dict) -> bool:
    metrics, gate = value["portfolio"], value["economic_gate"]
    months = metrics.get("monthly_return_pct", {})
    return bool(gate["passes"] and metrics["negative_months"] == 0
                and metrics["dd_pct"] <= 5
                and metrics["literal_target_hit_rate"] >= 75
                and months and all(month >= 30 for month in months.values()))


def main() -> None:
    base.MANIFEST = MANIFEST
    candles, sources = base.load()
    raw = base.build_features(candles)
    definitions = route_frames(raw)
    evaluated = []
    for key, candidate in definitions.items():
        with_outcomes = add_outcomes(candidate, candles)
        target = float(with_outcomes.selected_target_pct.iloc[0])
        selection = evaluate(with_outcomes, SELECTION_YEARS)
        selected = screen(selection, target, 150)
        item = {
            "key": key, "route": str(with_outcomes.route.iloc[0]),
            "config": str(with_outcomes.config.iloc[0]),
            "target_pct": target, "rows": len(with_outcomes),
            "selection_2020_2021": selection,
            "selection_screen_pass": selected,
            "_frame": with_outcomes,
        }
        if selected:
            validation = evaluate(with_outcomes, (VALIDATION_YEAR,))
            item["validation_2022"] = validation
            item["validation_screen_pass"] = screen(
                validation, target, 60)
            if item["validation_screen_pass"]:
                confirmation = evaluate(with_outcomes, (CONFIRMATION_YEAR,))
                item["confirmation_2023"] = confirmation
                item["confirmation_screen_pass"] = screen(
                    confirmation, target, 60)
        evaluated.append(item)

    developed = [item for item in evaluated if item["selection_screen_pass"]]
    validated = [item for item in developed
                 if item.get("validation_screen_pass", False)]
    confirmed = [item for item in validated
                 if item.get("confirmation_screen_pass", False)]
    confirmed.sort(key=lambda item: rank([
        item["selection_2020_2021"], item["validation_2022"],
        item["confirmation_2023"]]), reverse=True)

    union_frame = None
    union = None
    if confirmed:
        union_frame = pd.concat([item["_frame"] for item in confirmed],
                                ignore_index=True, sort=False)
        union_frame = (union_frame.sort_values(
            ["entry_time", "direction", "selected_target_pct", "idio_score"],
            ascending=[True, True, False, False])
            .drop_duplicates(["entry_time", "direction"], keep="first"))
        union = {
            "selection_2020_2021": evaluate(union_frame, SELECTION_YEARS),
            "validation_2022": evaluate(union_frame, (VALIDATION_YEAR,)),
            "confirmation_2023": evaluate(union_frame, (CONFIRMATION_YEAR,)),
        }

    early_objective = bool(union and all(
        objective_pass(value) for value in union.values()))
    later = None
    if early_objective:
        later = {str(year): evaluate(union_frame, (year,))
                 for year in HIDDEN_YEARS}
    later_objective = bool(later and all(
        objective_pass(value) for value in later.values()))
    status = ("OBJECTIVE_PASS" if later_objective
              else "REJECTED_LATER" if later is not None
              else "REJECTED_EARLY_OBJECTIVE" if confirmed
              else "REJECTED_CONFIRMATION" if validated
              else "REJECTED_VALIDATION" if developed
              else "REJECTED_SELECTION")

    evaluated.sort(key=lambda item: rank([
        item["selection_2020_2021"]]), reverse=True)
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "venue": "OKX global USDT swap", "bar": "1H",
            "entry": "next-hour market open",
            "selection": [2020, 2021], "validation": 2022,
            "confirmation": 2023,
            "hidden_revealed_only_after_complete_early_objective": list(HIDDEN_YEARS),
            "targets_pct": [1.5, 2.0], "stop_pct": 1.0,
            "hold_hours": {"1.5": 36, "2.0": 48},
            "episode_rule": "one strongest asset per direction and entry hour",
            "money": "3% current equity margin, fixed 10x gross",
            "fees": "excluded by user instruction",
        },
        "feature_rows": len(raw), "tested_routes": len(evaluated),
        "selection_passed": len(developed),
        "validation_passed": len(validated),
        "confirmation_passed": len(confirmed),
        "top_routes": [public_route(item) for item in evaluated[:12]],
        "confirmed_routes": [public_route(item) for item in confirmed],
        "confirmed_union": union,
        "early_objective_pass": early_objective,
        "later": later, "later_objective_pass": later_objective,
        "decision": (
            "No runtime change. A route must pass every chronological stage "
            "and the complete economic objective before any live consideration."),
        "sources": {**sources, str(MANIFEST): digest(MANIFEST),
                    str(Path(__file__)): digest(Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status, "feature_rows": len(raw),
        "tested_routes": len(evaluated),
        "selection_passed": len(developed),
        "validation_passed": len(validated),
        "confirmation_passed": len(confirmed),
        "top_routes": [public_route(item) for item in evaluated[:5]],
        "confirmed_union": union, "later": later,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
