import numpy as np

from crypto_false_breakout_liquidity_refinement_oct21_audit import (
    structure_features,
)


def test_structure_features_do_not_read_bars_after_entry():
    size = 130
    close = np.linspace(100.0, 110.0, size)
    open_ = close - .05
    high = close + .2
    low = close - .2
    times = np.arange(size, dtype=np.int64) * 300
    indices = np.array([110], dtype=np.int64)
    directions = np.array([-1], dtype=np.int8)

    original = structure_features(
        open_, high, low, close, times, indices, directions)
    changed_open = open_.copy()
    changed_high = high.copy()
    changed_low = low.copy()
    changed_close = close.copy()
    changed_open[111:] *= 2
    changed_high[111:] *= 2
    changed_low[111:] *= 2
    changed_close[111:] *= 2
    revised = structure_features(
        changed_open, changed_high, changed_low, changed_close,
        times, indices, directions)

    np.testing.assert_allclose(original, revised, equal_nan=True)
