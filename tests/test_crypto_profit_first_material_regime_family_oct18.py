import crypto_profit_first_material_regime_family_oct18_audit as audit


def test_route_key_is_unambiguous():
    assert audit.key("bear", "breakout") == "bear|breakout"


def test_route_gate_does_not_require_win_rate():
    selection = {"all": {"n": 20, "return_pct": 1.0, "dd_pct": 2.0,
                          "wr": 1.0}}
    confirmation = {"all": {"n": 30, "return_pct": 1.0, "dd_pct": 2.0},
                    "periods": {"a": {"return_pct": 1.0, "dd_pct": 2.0}}}
    assert audit.route_ok(selection, confirmation)


def test_runtime_rails_keep_cluster_cap():
    assert audit.RAILS == {"loss_pause": 3, "hourly_cap": 2}
