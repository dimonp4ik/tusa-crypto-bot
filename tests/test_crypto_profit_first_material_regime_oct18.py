import crypto_profit_first_material_regime_oct18_audit as audit


def test_route_development_does_not_gate_win_rate():
    selection = {"all": {"n": 30, "return_pct": 1.0, "dd_pct": 5.0,
                          "wr": 10.0}}
    confirmation = {"all": {"n": 50, "return_pct": 2.0, "dd_pct": 5.0,
                              "wr": 10.0},
                    "periods": {"a": {"return_pct": 1.0, "dd_pct": 4.0},
                                "b": {"return_pct": 1.0, "dd_pct": 4.0}}}
    assert audit.route_development_ok(selection, confirmation)


def test_route_development_rejects_losing_regime_period():
    selection = {"all": {"n": 30, "return_pct": 1.0, "dd_pct": 5.0}}
    confirmation = {"all": {"n": 50, "return_pct": 2.0, "dd_pct": 5.0},
                    "periods": {"a": {"return_pct": -1.0, "dd_pct": 4.0}}}
    assert not audit.route_development_ok(selection, confirmation)


def test_router_has_profit_first_fraction_grid():
    assert audit.FRACTIONS[:2] == (0.0, .05)


def test_subset_search_covers_all_nonempty_regime_combinations(monkeypatch):
    calls = []
    monkeypatch.setattr(audit, "detailed", lambda frame, periods, routes: (
        calls.append(tuple(sorted(routes))) or {
            "all": {"n": 100, "return_pct": 1.0, "dd_pct": 1.0,
                    "trades_per_calendar_day": 3.0,
                    "monthly_return_pct": {"x": 1.0}, "negative_months": 0},
            "periods": {"x": {"return_pct": 1.0}},
        }))
    routes = {name: {} for name in ("a", "b", "c", "d")}
    audit.select_route_subset(None, routes)
    assert len(set(calls)) == 15


def test_runtime_rail_grid_never_allows_more_than_two_hourly_entries():
    source = __import__("inspect").getsource(audit.select_runtime_rails)
    assert "(1, 2, 3), (1, 2)" in source
