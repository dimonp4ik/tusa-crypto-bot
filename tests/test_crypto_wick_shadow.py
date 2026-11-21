from src import db
from src.crypto_wick_shadow import countertrend_wick_evidence


def test_countertrend_wick_uses_aligned_closed_bars_and_direction():
    btc = dict(time=list(range(900, 18 * 900, 900)), close=[100.] * 16 + [101.])
    candles = dict(time=[btc["time"][-1]], low=[100.], high=[110.], close=[109.])
    short = countertrend_wick_evidence(candles, btc, "SHORT")
    assert short["wick_would_skip"] == 1
    assert countertrend_wick_evidence(candles, btc, "LONG")["wick_would_skip"] == 0
    assert countertrend_wick_evidence(dict(candles, confirmed=[False]), btc, "SHORT") == {}
    assert countertrend_wick_evidence(dict(candles, time=[900]), btc, "SHORT") == {}
    assert countertrend_wick_evidence(dict(candles, close=[106.]), btc, "SHORT")["wick_would_skip"] == 0


def test_forward_counts_use_first_linked_snapshot_and_observed_paper_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "audit.sqlite"))
    db.init_db()
    base = dict(symbol="TEST", direction="SHORT", current_price=100., source="venue_shadow_profit_first_2r")
    for veto, realized in ((1, -1.1), (0, None)):
        signal = db.log_signal(base, 99., 98., 101.)
        for flag in (veto, 1-veto):
            setup = db.log_setup_candidate(dict(base, _decision_context={
                "wick_shadow_profile": "countertrend_wick_v1", "wick_would_skip": flag,
            }))
            db.link_setup_to_signal(setup, signal)
        with db._conn() as conn:
            conn.execute("UPDATE signals SET realized_r=? WHERE id=?", (realized, signal))
    stats = db.countertrend_wick_shadow_outcomes(0)
    assert stats == {
        "kept": {"observed": 1, "resolved": 0, "total_r": 0.0},
        "would_skip": {"observed": 1, "resolved": 1, "total_r": -1.1},
    }
