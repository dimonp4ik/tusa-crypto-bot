import crypto_profit_first_loss_cooldown_nov1_audit as audit


def row(entry, exit_, net_r, symbol="BTCUSDT"):
    return {"entry_time": entry, "exit_time": exit_, "net_r": net_r,
            "symbol": symbol}


def test_cooldown_uses_only_accepted_closed_outcomes():
    rows = [
        row(0, 10, -1),
        row(20, 30, -1, "ETHUSDT"),  # rejected by first loss cooldown
        row(3610, 3620, 1, "XRPUSDT"),
    ]
    accepted = audit.apply_cooldown(rows, 1, 1)
    assert [item["entry_time"] for item in accepted] == [0, 3610]


def test_same_timestamp_exit_is_not_known_before_entry():
    rows = [row(0, 10, -1), row(10, 20, 1, "ETHUSDT")]
    accepted = audit.apply_cooldown(rows, 1, 1)
    assert len(accepted) == 2
