from crypto_actual_move_group_health_oct20_audit import group_health_features


def test_health_uses_only_exits_before_review_boundary():
    rows = [
        {"entry_time": 100, "exit_time": 200, "module": "m",
         "symbol": "BTC", "direction": "LONG",
         "signed_price_return": .01, "literal_target_hit": True},
        {"entry_time": 500, "exit_time": 90000, "module": "m",
         "symbol": "BTC", "direction": "LONG",
         "signed_price_return": -.01, "literal_target_hit": False},
        {"entry_time": 86500, "exit_time": 86600, "module": "m",
         "symbol": "BTC", "direction": "LONG",
         "signed_price_return": .02, "literal_target_hit": True},
    ]

    result = group_health_features(rows, ("module", "symbol"), 1, 14)

    assert result.loc[0, "health_n"] == 0
    # At day two start, only the first trade has completed.  The second trade
    # entered earlier but cannot leak into health because it exits later.
    assert result.loc[2, "health_n"] == 1
    assert result.loc[2, "health_mean"] == .01
    assert result.loc[2, "health_hit"] == 1.0
