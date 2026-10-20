import numpy as np

from crypto_15m_native_cross_section_oct20_audit import (
    resample_15m,
    resolve_fixed,
)


def test_resample_keeps_only_complete_contiguous_buckets():
    candle = {
        "time": np.array([0, 300, 600, 900, 1200, 1800, 2100, 2400]),
        "open": np.arange(8, dtype=float) + 10,
        "high": np.arange(8, dtype=float) + 11,
        "low": np.arange(8, dtype=float) + 9,
        "close": np.arange(8, dtype=float) + 10.5,
    }

    result = resample_15m(candle)

    assert result.time.tolist() == [0, 1800]
    assert result.iloc[0].to_dict() == {
        "time": 0.0, "open": 10.0, "high": 13.0,
        "low": 9.0, "close": 12.5,
    }


def test_resolver_is_stop_first_when_target_and_stop_share_bar():
    open_ = np.array([100.0, 100.0])
    high = np.array([102.5, 100.0])
    low = np.array([98.5, 100.0])
    close = np.array([101.0, 100.0])
    times = np.array([0, 900], dtype=np.int64)

    signed, exits, codes, valid = resolve_fixed(
        open_, high, low, close, times, np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int8), 0.02, 0.01, 2)

    assert valid.tolist() == [1]
    assert np.isclose(signed[0], -0.01)
    assert codes.tolist() == [-1]
    assert exits.tolist() == [900]


def test_resolver_rejects_missing_entry_candle():
    values = np.array([np.nan])
    signed, _, _, valid = resolve_fixed(
        values, values, values, values, np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64), np.array([1], dtype=np.int8),
        0.02, 0.01, 1)

    assert valid.tolist() == [0]
    assert signed.tolist() == [0.0]
