import crypto_profit_first_cluster_rails_oct20_audit as audit


def test_cluster_grid_is_small_and_fixed():
    assert audit.LOSS_PAUSES == (1, 2, 3)
    assert audit.HOURLY_CAPS == (1, 2)


def test_development_score_penalizes_drawdown():
    def value(dd):
        return {"all": {"return_pct": 30, "dd_pct": dd,
                        "negative_months": 0,
                        "monthly_return_pct": {"2022-01": 30}}}
    assert audit.development_score(value(2), value(2)) > \
        audit.development_score(value(4), value(4))
