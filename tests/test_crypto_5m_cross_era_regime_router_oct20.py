import pandas as pd

import crypto_5m_cross_era_regime_router_oct20_audit as audit


def test_stable_experts_require_every_development_year():
    rows = []
    for year in audit.DEVELOPMENT_YEARS:
        rows.append({
            "year": year, "family": "stable", "signed_price_return": .01,
            "geometry": "fast",
        })
    rows.extend([
        {"year": 2020, "family": "missing", "signed_price_return": .01,
         "geometry": "fast"},
        {"year": 2021, "family": "missing", "signed_price_return": .01,
         "geometry": "fast"},
    ])
    frame = pd.DataFrame(rows)

    result = audit.stable_experts(
        {"fast": frame}, ("family",), min_count=1,
        min_year_mean=0, min_year_win_rate=.5)

    assert result.family.tolist() == ["stable"]


def test_stable_experts_choose_best_worst_year_geometry():
    frames = {}
    for geometry, value in (("fast", .01), ("slow", .005)):
        frames[geometry] = pd.DataFrame([
            {"year": year, "family": "pullback",
             "signed_price_return": value, "geometry": geometry}
            for year in audit.DEVELOPMENT_YEARS
        ])

    result = audit.stable_experts(
        frames, ("family",), min_count=1,
        min_year_mean=0, min_year_win_rate=.5)

    assert result.geometry.tolist() == ["fast"]


def test_market_context_uses_only_same_and_prior_closes():
    times = [300 * index for index in range(120)]
    candles = {}
    for number, symbol in enumerate(("A", "B", "C", "D", "E"), 1):
        close = [100 + number + .1 * index for index in range(120)]
        candles[symbol] = {
            "time": times, "close": close,
            "high": [value + .2 for value in close],
            "low": [value - .2 for value in close],
        }

    context = audit.market_context(candles).set_index("signal_time")

    assert context.loc[times[-1], "market_universe"] == 5
    assert context.loc[times[-1], "market_breadth_positive"] == 1.0
