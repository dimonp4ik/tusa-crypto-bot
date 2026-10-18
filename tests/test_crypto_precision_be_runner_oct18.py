import crypto_precision_be_runner_oct18_audit as audit


def candles(rows):
    return {
        "time": [900 * i for i in range(len(rows))],
        "open": [row[0] for row in rows],
        "high": [row[1] for row in rows],
        "low": [row[2] for row in rows],
        "close": [row[3] for row in rows],
    }


def test_runner_breakeven_precedes_target_after_activation_bar():
    data = candles([
        (100.0, 100.6, 99.9, 100.5),  # activates +0.25R
        (100.5, 104.2, 99.8, 103.0),  # both BE and 2R: resting BE wins
    ])
    outcome, net_r, exit_time = audit._runtime_runner(
        data, 0, direction="LONG", entry=100.0, sl=98.0)
    assert outcome == "BE"
    assert net_r == 0.0
    assert exit_time == 1800.0


def test_runner_allows_target_on_activation_bar_before_be_is_armed():
    data = candles([(100.0, 104.2, 99.9, 103.5)])
    outcome, net_r, exit_time = audit._runtime_runner(
        data, 0, direction="LONG", entry=100.0, sl=98.0)
    assert outcome == "TP2"
    assert net_r == 2.0
    assert exit_time == 900.0


def test_runner_uses_gap_aware_original_stop():
    data = candles([(97.0, 97.5, 96.5, 97.0)])
    outcome, net_r, _ = audit._runtime_runner(
        data, 0, direction="LONG", entry=100.0, sl=98.0)
    assert outcome == "SL"
    assert net_r == -1.5
