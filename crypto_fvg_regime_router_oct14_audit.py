"""Regime-specific causal FVG router frozen before the 2025 reveal."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

import crypto_fvg_causal_stop_oct14_audit as fvg


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_fvg_regime_router_oct14.json"
ROUTER_ATOMS = (
    "base",
    "volume_ratio_ge_1.5", "volume_ratio_ge_2.0", "volume_ratio_ge_2.5",
    "extension_atr_ge_0.75", "extension_atr_ge_1.0",
    "extension_atr_ge_1.25", "entry_range_atr_ge_1.0",
    "entry_range_atr_ge_1.25", "quality_score_ge_80.0",
    "quality_score_ge_82.5", "mtf_score_ge_14.0", "mtf_score_ge_15.0",
    "bos_age_le_2.0", "bos_age_le_3.0", "confirmation_count_ge_1.0",
    "confirmation_count_ge_2.0", "has_liquidity_sweep", "has_engulfing",
    "has_wick", "has_nested_ob", "has_body_strength",
    "btc_strong_alignment", "reversal_candle", "reversal_or_sweep",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regimes(frame: pd.DataFrame) -> dict[str, pd.Series]:
    one = frame.aligned_1h.fillna(False)
    four = frame.aligned_4h.fillna(False)
    return {
        "aligned_1h_4h": one & four,
        "aligned_4h_only": ~one & four,
        "aligned_1h_only": one & ~four,
        "countertrend_both": ~one & ~four,
    }


def rules(atoms: dict[str, pd.Series]) -> dict[str, tuple[list[str], pd.Series]]:
    available = [name for name in ROUTER_ATOMS if name in atoms]
    result = {name: ([name], atoms[name]) for name in available}
    pair_atoms = [name for name in available if name != "base"]
    for first, second in itertools.combinations(pair_atoms, 2):
        name = f"{first}__and__{second}"
        result[name] = ([first, second], atoms[first] & atoms[second])
    return result


def route_score(development: dict, validation: dict) -> tuple:
    return (
        min(development["return_pct"], validation["return_pct"]),
        min(development["wr"], validation["wr"]),
        -max(development["dd_pct"], validation["dd_pct"]),
        development["n"] + validation["n"],
    )


def main() -> None:
    frame, coverage = fvg.load()
    atoms = fvg.atom_masks(frame)
    candidates = rules(atoms)
    regime_masks = regimes(frame)
    frozen_routes = {}
    diagnostics = {}
    selected_mask = pd.Series(False, index=frame.index)
    for regime_name, regime_mask in regime_masks.items():
        development = []
        seen = set()
        for rule_name, (rule_atoms, rule_mask) in candidates.items():
            mask = regime_mask & rule_mask
            key = tuple(frame.index[mask & frame.year.eq(2023)])
            if key in seen:
                continue
            seen.add(key)
            value = fvg.exact_year(frame[mask], 2023, coverage)
            if value["n"] >= 60:
                development.append({
                    "variant": rule_name, "atoms": rule_atoms,
                    "development": value,
                })
        development.sort(key=lambda item: fvg.score(item["development"]),
                         reverse=True)
        shortlist = development[:20]
        for item in shortlist:
            mask = regime_mask & candidates[item["variant"]][1]
            item["validation"] = fvg.exact_year(frame[mask], 2024, coverage)
            item["qualified_before_2025"] = bool(
                item["development"]["n"] >= 60
                and item["validation"]["n"] >= 50
                and item["development"]["return_pct"] > 0
                and item["validation"]["return_pct"] > 0
                and item["development"]["mean_signed_price_return_pct"] > 0
                and item["validation"]["mean_signed_price_return_pct"] > 0)
        qualified = [item for item in shortlist
                     if item["qualified_before_2025"]]
        qualified.sort(key=lambda item: route_score(
            item["development"], item["validation"]), reverse=True)
        frozen = qualified[0] if qualified else None
        diagnostics[regime_name] = {
            "development_candidates": len(development),
            "qualified_before_2025": len(qualified),
            "top_qualified": qualified[:10],
            "frozen": frozen,
        }
        if frozen:
            frozen_routes[regime_name] = frozen
            selected_mask |= regime_mask & candidates[frozen["variant"]][1]

    exact = {str(year): fvg.exact_year(frame[selected_mask], year, coverage)
             for year in range(2023, 2027)}
    objective_checks = {
        year: bool(value["n"] >= 100 and 75 <= value["wr"] <= 95
                   and value["trades_per_calendar_day"] >= 3
                   and value["negative_months"] == 0
                   and value["dd_pct"] <= 5
                   and value["economic_gate"]["passes"])
        for year, value in exact.items()
    }
    component_checks = {
        year: bool(value["n"] >= 100 and value["return_pct"] > 0
                   and value["mean_signed_price_return_pct"] > 0
                   and value["dd_pct"] <= 8)
        for year, value in exact.items()
    }
    status = ("OBJECTIVE_PASS" if all(objective_checks.values())
              else "STABLE_COMPONENT" if all(component_checks.values())
              else "REJECTED_FORWARD")
    report = {
        "status": status, "ready_for_live_orders": False,
        "protocol": {
            "router": "four mutually exclusive 1h/4h trend-alignment regimes",
            "route_selection": (
                "rule ranking on 2023, validation and freeze on 2024, global "
                "portfolio reveal on 2025 and observed 2026"),
            "rule_atoms": list(ROUTER_ATOMS),
            "orders": "historical market-entry records only",
            "position": "3% current-equity margin, fixed 10x gross",
            "target_move_filter_pct": [0.5, 2.0],
        },
        "frozen_routes_before_2025": frozen_routes,
        "route_diagnostics": diagnostics,
        "selected_rows": int(selected_mask.sum()),
        "exact_frozen_portfolio": exact,
        "objective_checks": objective_checks,
        "component_checks": component_checks,
        "decision": (
            "Keep the regime router out of runtime unless the combined frozen "
            "portfolio remains profitable and low-drawdown in every revealed era."),
        "sources": {
            str(fvg.SOURCE): digest(fvg.SOURCE),
            str(Path(fvg.__file__)): digest(Path(fvg.__file__)),
            str(Path(__file__)): digest(Path(__file__)),
        },
        "limitations": [
            "Historical slices cover roughly six months per labeled year.",
            "The regime split is causal but the underlying FVG generator is legacy research.",
            "2026 cause profiles were inspected before this router study, so it is observed stress evidence.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": status,
        "frozen_routes": {name: item["variant"]
                           for name, item in frozen_routes.items()},
        "selected_rows": int(selected_mask.sum()),
        "exact": exact,
        "objective_checks": objective_checks,
        "component_checks": component_checks,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
