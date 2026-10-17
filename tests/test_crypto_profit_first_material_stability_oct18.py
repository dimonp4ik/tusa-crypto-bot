import pandas as pd

import crypto_profit_first_material_stability_oct18_audit as audit


def test_regime_margin_uses_weaker_crossover_distance():
    frame = pd.DataFrame({"btc_fast_slow_atr": [-2.0, .2],
                          "btc_slow_long_atr": [.5, -3.0]})
    result = audit.enrich(frame)
    assert result.btc_regime_margin_atr.tolist() == [.5, .2]


def test_gate_requires_both_stability_and_efficiency():
    frame = pd.DataFrame({"btc_regime_margin_atr": [.4, .8, .8],
                          "btc_eff20": [.4, .1, .4]})
    result = audit.apply_gate(frame, .5, .2)
    assert result.index.tolist() == [2]


def test_development_gate_ignores_win_rate():
    selection = {"all": {"return_pct": 1.0, "dd_pct": 2.0,
                          "trades_per_calendar_day": 3.0, "wr": 1.0}}
    confirmation = {"all": {"return_pct": 1.0, "dd_pct": 2.0,
                              "trades_per_calendar_day": 2.0, "wr": 1.0},
                    "periods": {"x": {"return_pct": 1.0}}}
    assert audit.development_ok(selection, confirmation)
