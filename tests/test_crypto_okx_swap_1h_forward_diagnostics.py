from crypto_okx_swap_1h_forward_append_oct16_audit import stop_cause_diagnostics


def row(outcome, *, pullback, direction="SHORT", ret24=.01,
        volume_ratio=1.0):
    return {
        "outcome": outcome,
        "family_pullback": 1.0 if pullback else None,
        "direction": direction,
        "ret24": ret24,
        "volume_ratio": volume_ratio,
        "signed_price_return": .005 if outcome == "TP" else -.01,
        "breakout_up": -.01,
        "breakout_down": -.02,
    }


def test_interpretation_counts_mismatch_stops_instead_of_all_mismatches():
    prior = [row("TP", pullback=True)]
    fresh = [
        row("SL", pullback=True, volume_ratio=3.0),
        row("TP", pullback=True, volume_ratio=1.0),
        row("SL", pullback=False, volume_ratio=2.0),
    ]

    found = stop_cause_diagnostics(prior, fresh)

    assert found["fresh_pullback_recent_24h_regime_mismatch"]["n"] == 2
    assert found["fresh_pullback_recent_24h_regime_mismatch"]["stops"] == 1
    assert "1 of the 2 fresh stops" in found["interpretation"]
    assert "losses had higher volume" in found["interpretation"]
