import crypto_profit_first_exposure_cap_oct20_audit as audit


def row(symbol, direction, entry, exit_):
    return {"symbol": symbol, "direction": direction,
            "entry_time": entry, "exit_time": exit_, "source_priority": 0,
            "module_priority": 0, "outcome": "TP"}


def test_total_exposure_cap_blocks_third_open_position():
    rows = [row("A", "LONG", 0, 100), row("B", "SHORT", 10, 100),
            row("C", "LONG", 20, 100)]
    assert len(audit.exposure_gate(rows, direction_cap=2, total_cap=2)) == 2


def test_direction_cap_blocks_correlated_side():
    rows = [row("A", "LONG", 0, 100), row("B", "LONG", 10, 100),
            row("C", "SHORT", 20, 100)]
    selected = audit.exposure_gate(rows, direction_cap=1, total_cap=3)
    assert [(item["symbol"], item["direction"]) for item in selected] == [
        ("A", "LONG"), ("C", "SHORT")]
