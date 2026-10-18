import crypto_xperp_native_geometry_oct20_audit as audit


def metrics(return_pct=1.0, dd_pct=2.0, trades=3.0,
            negative_months=0, n=80):
    return {
        "return_pct": return_pct, "dd_pct": dd_pct,
        "trades_per_calendar_day": trades,
        "negative_months": negative_months, "n": n,
    }


def test_period_ok_does_not_require_a_win_rate():
    value = metrics()
    value["wr"] = 20.0

    assert audit.period_ok(value)


def test_period_ok_rejects_negative_month():
    assert not audit.period_ok(metrics(return_pct=-.01, negative_months=1))


def test_rank_key_prefers_higher_worst_development_month():
    weak = {
        "admissible": True, "selection": metrics(1),
        "confirmation": {"a": metrics(2), "b": metrics(3)},
    }
    robust = {
        "admissible": True, "selection": metrics(2),
        "confirmation": {"a": metrics(2), "b": metrics(2)},
    }

    assert audit.rank_key(robust) > audit.rank_key(weak)
