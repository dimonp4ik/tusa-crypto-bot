from __future__ import annotations

import crypto_profit_first_capacity_oct31_audit as audit


def _row(symbol: str, direction: str, entry: int, exit_: int,
         outcome: str = "TP") -> dict:
    return {
        "symbol": symbol, "direction": direction,
        "entry_time": entry, "exit_time": exit_, "outcome": outcome,
        "source_priority": 3, "module_priority": 999,
        "signed_price_return": .01 if outcome == "TP" else -.01,
        "period": "2024",
    }


def test_capacity_gate_enforces_rolling_hour_limit():
    rows = [_row(f"S{i}", "LONG", i*60, 7200) for i in range(4)]

    accepted = audit.capacity_gate(
        rows, direction_cap=6, rolling_hour_cap=3, loss_pause=3)

    assert [row["symbol"] for row in accepted] == ["S0", "S1", "S2"]


def test_capacity_gate_enforces_same_direction_limit():
    rows = [_row(f"S{i}", "LONG", i*60, 7200) for i in range(4)]

    accepted = audit.capacity_gate(
        rows, direction_cap=2, rolling_hour_cap=6, loss_pause=3)

    assert [row["symbol"] for row in accepted] == ["S0", "S1"]


def test_peak_open_counts_total_and_direction():
    rows = [
        _row("A", "LONG", 0, 100),
        _row("B", "LONG", 10, 90),
        _row("C", "SHORT", 20, 80),
    ]

    assert audit.peak_open(rows) == (3, 2)
