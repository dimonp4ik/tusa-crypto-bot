import pandas as pd

import crypto_xperp_style_health_router_oct20_audit as audit


def test_health_router_excludes_unfinished_outcomes():
    rows = []
    for style, signed, exit_time in (
            ("reversal", .01, 90), ("momentum", -.01, 90),
            ("reversal", -.5, 150), ("momentum", .5, 150)):
        rows.append({
            "entry_time": 80 if exit_time == 90 else 100,
            "exit_time": exit_time, "style": style,
            "signed_return": signed, "model_quality": 1,
            "symbol": style,
        })
    frame = pd.DataFrame(rows)
    # At t=100 only the trades exited at 90 may affect the choice.  The large
    # opposite outcomes that exit at 150 are unavailable and must be ignored.
    found = audit.trailing_health_route(
        frame, lookback_days=3, min_count=1, min_mean=0)
    at_100 = found[found.entry_time.eq(100)]
    assert at_100["style"].tolist() == ["reversal"]


def test_health_router_pauses_when_both_styles_negative():
    frame = pd.DataFrame([
        {"entry_time": 80, "exit_time": 90, "style": style,
         "signed_return": -.01, "model_quality": 1, "symbol": style}
        for style in ("reversal", "momentum")
    ] + [
        {"entry_time": 100, "exit_time": 110, "style": style,
         "signed_return": .01, "model_quality": 1, "symbol": style}
        for style in ("reversal", "momentum")
    ])
    found = audit.trailing_health_route(
        frame, lookback_days=3, min_count=1, min_mean=0)
    assert found[found.entry_time.eq(100)].empty
