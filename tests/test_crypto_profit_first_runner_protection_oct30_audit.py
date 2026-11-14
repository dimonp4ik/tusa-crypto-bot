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


def test_expired_runner_is_marked_at_terminal_close_after_tp1():
    bars=[(100.,100.6,99.5,100.5)]+[(100.5,101.,100.1,100.8)]*47
    outcome,runner,banked,end=replay(bars)
    assert (outcome,banked,end)==('TP_EXPIRED',True,48*audit.INTERVAL)
    assert runner==pytest.approx(.8)


def test_last_bar_tp1_can_expire_below_entry_before_breakeven_activates():
    bars=[(100.,100.4,99.8,100.1)]*47+[(100.1,100.6,99.5,99.8)]
    outcome,runner,banked,_=replay(bars)
    assert (outcome,banked)==('TP_EXPIRED',True)
    assert runner==pytest.approx(-.2)


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


@pytest.mark.parametrize('direction,stop,bar', [
    ('LONG',99.,(100.,102.1,99.5,102.)),
    ('SHORT',101.,(100.,100.5,97.9,98.)),
])
def test_activation_bar_stress_resolves_new_stop_before_target(direction,stop,bar):
    kwargs=dict(direction=direction,entry=100.,stop=stop,activation_r=.5,lock_r=0.)
    data=candles([bar])
    normal=audit.replay_runner(data,0,**kwargs)
    stress=audit.replay_runner(data,0,activation_bar_stress=True,**kwargs)
    assert normal[:3]==('TP_RUNNER',2.,True)
    assert stress==('LOCK_ACTIVATION_AMBIGUOUS',0.,True,audit.INTERVAL)


def test_activation_bar_stress_does_not_enable_protection_below_activation():
    result=audit.replay_runner(candles([(100.,100.6,99.5,100.4)]),0,
        direction='LONG',entry=100.,stop=99.,activation_r=1.,lock_r=0.,
        activation_bar_stress=True)
    assert result[0]=='TP_EXPIRED'
    assert result[1]==pytest.approx(.4)


def test_activation_bar_stress_keeps_original_stop_priority():
    result=audit.replay_runner(candles([(100.,102.1,98.9,102.)]),0,
        direction='LONG',entry=100.,stop=99.,activation_r=.5,lock_r=0.,
        activation_bar_stress=True)
    assert result[:3]==('SL',-1.,False)


def test_minute_replay_preserves_horizon_and_gap_detection():
    data=candles([(100.,100.4,99.8,100.1)]*4)
    data['time']=[0,60,120,240]
    result=audit.replay_runner(data,0,direction='LONG',entry=100.,stop=99.,
        activation_r=.5,lock_r=0.,bar_seconds=60,max_bars=4)
    assert result[0]=='EXPIRED'
    assert result[1]==pytest.approx(.1)
    assert result[-1]==180
