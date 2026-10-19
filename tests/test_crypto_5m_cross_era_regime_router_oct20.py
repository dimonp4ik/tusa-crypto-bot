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
