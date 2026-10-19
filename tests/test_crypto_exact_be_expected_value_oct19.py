import pandas as pd

import crypto_exact_be_expected_value_oct19_audit as audit
import crypto_precision_be_runner_oct18_audit as runner


def test_eligible_requires_literal_runner_move_between_half_and_two_percent():
    frame = pd.DataFrame({"risk_pct": [.0024, .0025, .01, .0101]})

    selected = audit.eligible(frame)

    assert selected.index.tolist() == [1, 2]


def test_month_floor_requires_every_named_month():
    value = {"monthly_return_pct": {"2026-07": 31.0, "2026-08": 28.0}}

    assert audit.month_floor(value, ("2026-07", "2026-08")) == 28.0
    assert audit.month_floor(value, ("2026-07", "2026-09")) == -999.0


def test_admissible_uses_profit_rails_without_a_win_rate_gate():
    value = {
        "n": 100,
        "dd_pct": 10.0,
        "trades_per_calendar_day": 3.0,
        "negative_months": 0,
        "wr": 20.0,
    }

    assert audit.admissible(value)
    value["negative_months"] = 1
    assert not audit.admissible(value)


def test_exact_runner_accepts_configurable_activation_and_target():
    candles = {
        "time": [0, 900, 1800],
        "open": [100.0, 100.6, 100.0],
        "high": [100.6, 101.1, 100.2],
        "low": [99.8, 100.4, 99.9],
        "close": [100.5, 100.8, 100.1],
    }

    outcome = runner._runtime_runner(
        candles, 0, direction="LONG", entry=100.0, sl=99.0,
        activation_r=.5, runner_r=1.0)

    assert outcome[:2] == ("TP2", 1.0)


def test_negative_expiry_advances_loss_pause_streak():
    frame = pd.DataFrame([{
        "runner_outcome": "EXPIRED", "net_r": -.2, "exit_time": 10.0,
        "entry_time": 1.0, "symbol": "BTCUSDT", "direction": "LONG",
        "family": "breakout", "model_probability": .8,
        "risk_pct": .01,
    }])

    rows = runner.rows_for(frame, "test")

    assert rows[0]["outcome"] == "SL"
    assert runner.maximum_stop_streak(rows) == 1
