import json

import pandas as pd

import crypto_profit_first_exact_health_router_oct30_audit as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_health_history_only_matures_after_a_strictly_earlier_exit():
    rows = []
    for sequence in range(10):
        rows.append({
            "entry_time": sequence,
            "exit_time": 100,
            "symbol": f"S{sequence}",
            "family": "breakout",
            "direction": "LONG",
            "net_r": -1.0,
        })
    rows.extend([
        {
            "entry_time": 100,
            "exit_time": 101,
            "symbol": "S10",
            "family": "breakout",
            "direction": "LONG",
            "net_r": 0.0,
        },
        {
            "entry_time": 101,
            "exit_time": 102,
            "symbol": "S11",
            "family": "breakout",
            "direction": "LONG",
            "net_r": 0.0,
        },
    ])
    filtered = audit.apply_health(
        pd.DataFrame(rows), scope="family", lookback=20, threshold=0.0)

    # Exits stamped at the current entry time are not yet observable.  They
    # become eligible for the following entry only.
    assert 100 in filtered.entry_time.tolist()
    assert 101 not in filtered.entry_time.tolist()


def test_rejected_paper_outcome_still_updates_later_health():
    rows = []
    for sequence in range(10):
        rows.append({
            "entry_time": sequence,
            "exit_time": sequence + 1,
            "symbol": f"S{sequence}",
            "family": "breakout",
            "direction": "LONG",
            "net_r": -1.0,
        })
    rows.extend([
        {
            "entry_time": 20,
            "exit_time": 21,
            "symbol": "REJECTED",
            "family": "breakout",
            "direction": "LONG",
            "net_r": 20.0,
        },
        {
            "entry_time": 22,
            "exit_time": 23,
            "symbol": "RECOVERED",
            "family": "breakout",
            "direction": "LONG",
            "net_r": 0.0,
        },
    ])
    filtered = audit.apply_health(
        pd.DataFrame(rows), scope="family", lookback=20, threshold=0.0)
    accepted_symbols = set(filtered.symbol)

    assert "REJECTED" not in accepted_symbols
    assert "RECOVERED" in accepted_symbols


def test_all_bounded_health_policies_are_rejected_before_holdouts():
    value = report()
    assert value["candidate_count"] == 54
    assert value["development_pass_count"] == 0
    assert value["selected_before_holdouts"] is None
    assert value["status"] == "REJECTED_EXACT_HEALTH_ROUTER"
    assert value["robust_improvement"] is False
    assert value["runtime_change_allowed"] is False
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
    assert value["baseline"]["direct"]["october"]["n"] == 186
    assert value["baseline"]["direct"]["october"]["return_pct"] == 15.7126
