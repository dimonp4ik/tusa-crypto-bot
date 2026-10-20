import numpy as np

from crypto_5m_structural_breakout_oct20_audit import resolve


def test_structural_breakout_resolver_is_stop_first():
    open_ = np.array([100.0, 100.0])
    high = np.array([102.0, 100.0])
    low = np.array([99.0, 100.0])
    close = np.array([101.0, 100.0])
    times = np.array([0, 300], dtype=np.int64)

    signed, exits, codes, valid = resolve(
        open_, high, low, close, times, np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int8), .015, .005, 2)

    assert valid.tolist() == [1]
    assert np.isclose(signed[0], -.005)
    assert codes.tolist() == [-1]
    assert exits.tolist() == [300]


def test_structural_breakout_resolver_stops_at_data_gap():
    open_ = np.array([100.0, 100.0])
    high = np.array([100.2, 102.0])
    low = np.array([99.8, 99.0])
    close = np.array([100.1, 101.0])
    times = np.array([0, 600], dtype=np.int64)

    signed, exits, codes, valid = resolve(
        open_, high, low, close, times, np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int8), .015, .005, 2)

    assert valid.tolist() == [1]
    assert np.isclose(signed[0], .001)
    assert codes.tolist() == [0]
    assert exits.tolist() == [300]
