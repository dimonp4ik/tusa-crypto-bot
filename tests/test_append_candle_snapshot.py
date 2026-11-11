import pytest

from append_candle_snapshot import FIELDS, merge


def candles(times):
    return {k: list(times) if k == "time" else [float(t + 1) for t in times]
            for k in FIELDS}


def test_merge_preserves_full_warmup_and_appends_tail():
    old, new = candles([0, 900, 1800]), candles([900, 1800, 2700])
    result, stats = merge(old, new)
    assert result == candles([0, 900, 1800, 2700])
    assert stats == {"overlap_bars": 2, "new_bars": 1}
    assert old == candles([0, 900, 1800])


def test_merge_rejects_revised_history():
    new = candles([900, 1800])
    new["close"][0] += 1
    with pytest.raises(ValueError, match="overlap revised"):
        merge(candles([0, 900]), new)


def test_merge_rejects_tail_gap_and_no_new_evidence():
    with pytest.raises(ValueError, match="contiguous new tail"):
        merge(candles([0, 900]), candles([900, 2700]))
    with pytest.raises(ValueError, match="contiguous new tail"):
        merge(candles([0, 900]), candles([0, 900]))
