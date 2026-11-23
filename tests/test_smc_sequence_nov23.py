import pytest

from smc_sequence_nov23 import Sequence, sequences, state


def closes_with_retest():
    bars = [(101., 102., 100., 101.25, 20.)] * 36
    bars[21] = (100., 101., 98., 100.5, 20.)
    bars[26] = (100., 101., 98.1, 100.5, 20.)
    bars[30] = (101.8, 101.9, 97.4, 100.5, 35.)
    bars[31] = (100.6, 103.7, 100.5, 103.5, 30.)
    bars[32] = (102.5, 103., 102., 102.7, 20.)
    bars[33] = (102.3, 103.4, 101.7, 103., 25.)
    bars[34] = (103., 104., 102., 103.5, 20.)
    return bars


def test_sweep_displacement_and_retest_first_become_known_on_their_own_closes():
    bars = closes_with_retest()
    assert state(bars[:30], 'LONG', .05) == set()
    assert state(bars[:31], 'LONG', .05) == {'sweep', 'equal_sweep'}
    assert state(bars[:32], 'LONG', .05) == {
        'sweep', 'equal_sweep', 'displacement', 'equal_displacement'}
    assert state(bars[:33], 'LONG', .05) == {'displacement', 'equal_displacement'}
    assert state(bars[:34], 'LONG', .05) == {'block_retest'}
    assert state(bars, 'LONG', .05) == set()


def test_mirror_and_order_block_failure():
    bars = closes_with_retest()[:34]
    mirrored = [(230-o, 230-low, 230-high, 230-close, vol) for o, high, low, close, vol in bars]
    assert Sequence(30, True, 31, 33, 30, 230-98.1) in sequences(mirrored, 'SHORT', .05)
    bars[32] = (102., 102.1, 100.2, 100.3, 20.)
    assert Sequence(30, True, 31, None, 30, 98.1) in sequences(bars, 'LONG', .05)
    assert 'block_retest' not in state(bars, 'LONG', .05)


def test_only_first_zone_touch_can_confirm_retest():
    bars = closes_with_retest()[:34]
    bars[32] = (102., 102.1, 101.5, 101.7, 20.)
    assert Sequence(30, True, 31, None, 30, 98.1) in sequences(bars, 'LONG', .05)


def test_preceding_retest_stops_being_usable_on_a_close_through_the_block():
    bars = closes_with_retest()[:35]
    bars[34] = (101., 101.2, 100.2, 100.3, 25.)
    assert state(bars[:34], 'LONG', .05) == {'block_retest'}
    assert state(bars, 'LONG', .05) == set()


def test_old_level_already_pierced_is_not_reswept_as_fresh():
    bars = closes_with_retest()[:31]
    bars[28] = (100., 102., 97.5, 100.1, 20.)
    assert state(bars, 'LONG', .05) == set()


def test_sequence_rejects_bad_ohlc_and_missing_context_stays_unknown():
    bars = closes_with_retest()
    bars[12] = (101., 102., 100., float('nan'), 10.)
    assert state(None, 'LONG', .05) is None
    with pytest.raises(ValueError, match='Invalid closed bars'):
        state(bars, 'LONG', .05)


def test_standalone_orders_only_use_information_through_the_trigger_close():
    from okx_smc_standalone_nov23 import discover, outcome

    bars = [(o+2000, h+2000, low+2000, c+2000, v) for o, h, low, c, v in closes_with_retest()]
    values = dict(zip(('open', 'high', 'low', 'close', 'volume'), map(list, zip(*bars)), strict=True),
                  time=list(range(0, len(bars)*900, 900)))

    def scan(frame):
        return discover(frame, 'TEST', {'tickSz': '.05'}, stocks=False, begin=0, end=len(bars)*900)[0]

    full = scan(values)
    for name, length in (('sweep', 31), ('displacement', 32), ('displacement_local_stop', 32), ('block_retest', 34)):
        expected = next(row for row in full if row['family'] == name and row['entry_time'] == length*900)
        prefix = {key: series[:length] for key, series in values.items()}
        assert expected in scan(prefix)
        assert outcome(expected, prefix, traded=True)['unknown_reason'] == 'missing_entry_candle'
