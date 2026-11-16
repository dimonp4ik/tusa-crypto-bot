import numpy as np
import pytest

from crypto_runner_exit_path_nov16 import replay
from crypto_profit_first_runner_protection_oct30_audit import replay_runner


def data(bars):
    return dict(time=[900 * i for i in range(len(bars))],
                **{key: [row[j] for row in bars] for j, key in enumerate(('open', 'high', 'low', 'close'))})


@pytest.mark.parametrize('direction', ['LONG', 'SHORT'])
def test_frozen_baseline_matches_existing_replayer_on_random_complete_paths(direction):
    rng = np.random.default_rng(1616)
    sign = 1 if direction == 'LONG' else -1
    for _ in range(100):
        close = 100 + np.cumsum(rng.normal(0, .25, 48))
        opened = np.r_[100., close[:-1]] + rng.normal(0, .1, 48)
        high = np.maximum(opened, close) + rng.uniform(0, .4, 48)
        low = np.minimum(opened, close) - rng.uniform(0, .4, 48)
        candles = data(list(zip(opened, high, low, close)))
        for stress in (False, True):
            expected = replay_runner(candles, 0, direction=direction, entry=100., stop=100. - sign,
                activation_r=.5, lock_r=0., activation_bar_stress=stress)
            actual = replay(candles, 0, direction=direction, entry=100., stop=100. - sign, stress=stress)
            assert (actual['outcome'], actual['runner_r'], actual['banked'], actual['exit_time']) == expected


def test_new_trailing_stop_cannot_use_an_earlier_low_as_its_fill():
    candles = data([(100., 102.5, 99.5, 102.), (102., 102.7, 101.7, 102.2), (102.2, 103.1, 102., 103.)])
    out = replay(candles, 0, direction='LONG', entry=100., stop=99., target_r=3.,
                 trail_distance_r=1., bars=3)
    assert out['outcome'] == 'TP_RUNNER' and out['runner_r'] == 3.
    stress = replay(candles, 0, direction='LONG', entry=100., stop=99., target_r=3.,
                    trail_distance_r=1., stress=True, bars=3)
    assert stress['outcome'] == 'LOCK_ACTIVATION_AMBIGUOUS'


def test_trailing_after_2r_preserves_a_gap_through_its_active_level():
    candles = data([(100., 100.6, 99.7, 100.5), (100.5, 102.5, 100.1, 102.3),
                    (101.2, 101.4, 101., 101.1)])
    out = replay(candles, 0, direction='LONG', entry=100., stop=99., target_r=4.,
                 trail_distance_r=1., bars=3)
    assert out['outcome'] == 'TRAIL' and out['runner_r'] == pytest.approx(1.2)


def test_incomplete_fast_stop_is_not_treated_as_a_mature_outcome():
    with pytest.raises(ValueError, match='incomplete'):
        replay(data([(100., 101., 98., 99.)]), 0, direction='LONG', entry=100., stop=99.)


def test_new_stop_already_crossed_at_close_is_marked_at_close_not_its_earlier_level():
    candles = data([(100., 102.5, 99.5, 100.4)])
    out = replay(candles, 0, direction='LONG', entry=100., stop=99., target_r=3.,
                 trail_distance_r=1., bars=1)
    assert out['outcome'] == 'TRAIL_MARK_ALREADY_CROSSED'
    assert out['runner_r'] == pytest.approx(.4)
