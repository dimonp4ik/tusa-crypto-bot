import pandas as pd

import crypto_actual_move_variant_router_oct20_audit as audit


def test_scores_use_only_exits_strictly_before_cutoff():
    cutoff = int(pd.Timestamp("2025-02-01", tz="UTC").timestamp())
    ledgers = {"a": pd.DataFrame({
        "exit_time": [cutoff - 1, cutoff],
        "signed_price_return": [.01, -1.0],
    })}
    found = audit.scores_at(
        ledgers, cutoff, window_days=30, minimum=1, minimum_mean=0)
    assert len(found) == 1
    assert found[0]["n"] == 1
    assert found[0]["mean"] == .01


def test_no_qualified_variant_falls_back_to_sweep():
    january = int(pd.Timestamp("2025-01-15", tz="UTC").timestamp())
    variants = {
        "sweep_only": [{"entry_time": january, "exit_time": january + 1,
                        "source_priority": 0, "module_priority": 0,
                        "symbol": "BTCUSDT", "direction": "LONG",
                        "outcome": "TP", "year": 2025,
                        "signed_price_return": .01}],
        "broad_first_plus_sweep": [], "stable_first_plus_sweep": [],
    }
    ledgers = {name: pd.DataFrame(columns=["exit_time", "signed_price_return"])
               for name in variants}
    rows, decisions = audit.monthly_route(
        variants, ledgers, "2025-01", "2025-01",
        window_days=30, minimum=20, minimum_mean=0)
    assert decisions[0]["selected"] == "sweep_only"
    assert len(rows) == 1
