import pandas as pd

import crypto_raw_hourly_direction_oct19_audit as audit


def candles(high, low, close=None):
    size = len(high)
    return pd.DataFrame({
        "time": [1000 + 900 * index for index in range(size)],
        "open": [100.0] * size,
        "high": high,
        "low": low,
        "close": close or [100.0] * size,
        "volume": [1.0] * size,
    })


def test_outcome_gives_stop_priority_on_ambiguous_bar():
    frame = candles([101.1], [99.4])

    value, outcome, _ = audit._outcome(frame, 0, 1)

    assert outcome == "SL"
    assert round(value, 6) == -audit.STOP_RETURN


def test_outcome_closes_at_timeout_market_price():
    frame = candles([100.2] * 16, [99.8] * 16,
                    [100.0] * 15 + [100.4])

    value, outcome, _ = audit._outcome(frame, 0, 1)

    assert outcome == "TIME"
    assert round(value, 6) == 0.004


def test_candidate_rows_count_negative_timeout_as_a_loss():
    frame = pd.DataFrame([{
        "symbol": "BTCUSDT", "entry_time": 1000,
        "exit_time": 2000, "direction_n": 1, "direction": "LONG",
        "signed_return": -0.002, "outcome": "TIME",
        "model_quality": .8,
    }])

    row = audit.candidate_rows(frame, .5)[0]

    assert row["exit_reason"] == "TIME"
    assert row["outcome"] == "SL"
    assert row["signed_price_return"] == -0.002
