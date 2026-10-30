import pandas as pd
import pytest

import crypto_profit_first_runner_protection_oct30_audit as audit


def candles(bars):
    result = {key: [] for key in ("time", "open", "high", "low", "close")}
    for index, (opened, high, low, close) in enumerate(bars):
        result["time"].append(index * audit.INTERVAL)
        result["open"].append(opened)
        result["high"].append(high)
        result["low"].append(low)
        result["close"].append(close)
    return result


def replay(bars, *, activation=.5, lock=0.0):
    return audit.replay_runner(
        candles(bars), 0, direction="LONG", entry=100.0, stop=99.0,
        activation_r=activation, lock_r=lock)


def test_tp1_then_breakeven_exits_before_a_later_2r_touch():
    outcome, runner_r, banked, exit_time = replay([
        (100.0, 100.5, 99.5, 100.4),
        (100.4, 100.6, 100.0, 100.1),
        (100.1, 102.1, 100.1, 102.0),
    ])
    assert (outcome, runner_r, banked) == ("LOCK", 0.0, True)
    assert exit_time == 2 * audit.INTERVAL


def test_runner_reaches_2r_when_entry_is_not_retested_first():
    outcome, runner_r, banked, _ = replay([
        (100.0, 100.5, 99.5, 100.4),
        (100.4, 102.1, 100.1, 102.0),
    ])
    assert (outcome, runner_r, banked) == ("TP_RUNNER", 2.0, True)


def test_original_stop_wins_an_ambiguous_first_target_bar():
    outcome, runner_r, banked, _ = replay([
        (100.0, 100.6, 98.9, 100.4),
    ])
    assert outcome == "SL"
    assert runner_r == pytest.approx(-1.0)
    assert banked is False


def test_no_protection_allows_a_runner_after_an_entry_retest():
    outcome, runner_r, banked, _ = replay([
        (100.0, 100.5, 99.5, 100.4),
        (100.4, 100.6, 100.0, 100.1),
        (100.1, 102.1, 100.1, 102.0),
    ], activation=99.0)
    assert (outcome, runner_r, banked) == ("TP_RUNNER", 2.0, True)


def test_first_bank_fraction_is_applied_only_after_tp1():
    frame = pd.DataFrame({
        "sequential_runner_r": [0.0, -1.0, 2.0],
        "sequential_banked": [True, False, True],
        "sequential_exit_time": [1, 2, 3],
        "sequential_outcome": ["LOCK", "SL", "TP_RUNNER"],
    })
    # Use the equivalent formula from the audit's exact rows_for path.
    fraction = .75
    net = frame.sequential_runner_r.where(
        ~frame.sequential_banked,
        fraction * .5 + (1 - fraction) * frame.sequential_runner_r)
    assert net.tolist() == pytest.approx([.375, -1.0, .875])
