import numpy as np
import pytest
from banked_runner_path_nov19 import resolve,POLICIES
from price_bracket_path_nov19 import resolve as baseline


def data(rows):
    return dict(time=[900*i for i in range(len(rows))],
        **{k:[r[i] for r in rows] for i,k in enumerate(('open','high','low','close','volume'))})


@pytest.mark.parametrize('direction',['LONG','SHORT'])
def test_baseline_parity_with_price_denominated_closed_bar_resolver(direction):
    rng=np.random.default_rng(191919);sign=1 if direction=='LONG' else -1
    for _ in range(200):
        close=100+np.cumsum(rng.normal(0,.25,48));opened=np.r_[100.,close[:-1]]
        high=np.maximum(opened,close)+rng.uniform(0,.5,48)
        low=np.minimum(opened,close)-rng.uniform(0,.5,48)
        c=data(list(zip(opened,high,low,close,rng.integers(0,10,48))))
        old=baseline(c,0,direction=direction,entry=100.,stop=100-sign,tp1=100+sign*.5,
            tp2=100+sign*2,bars=48)
        new=resolve(c,0,direction=direction,entry=100.,stop=100-sign,policy='be_tp1')
        # The old resolver defers BE until next bar even when the observed
        # confirming close has already crossed it. New resolver closes there.
        if new.get('outcome')=='AMEND_CROSSED_AT_CLOSE':continue
        assert new['known']==old['known']
        if new['known']:
            for k in ('exit_time','runner_price_return','bank_price_return','banked','near_exit_time'):
                assert new[k]==pytest.approx(old[k]) if new[k] is not None else old[k] is None


@pytest.mark.parametrize('policy',POLICIES)
def test_zero_volume_target_is_not_a_bank_or_profit(policy):
    c=data([(100,103,99.5,102,0),(100,100.2,98,99,1)])
    r=resolve(c,0,direction='LONG',entry=100,stop=99,policy=policy,bars=2,exit_stress=0)
    assert r['known'] and not r['banked'] and r['outcome']=='SL'


def test_half_stop_keeps_only_half_initial_risk_after_bank_and_gap_is_preserved():
    c=data([(100,100.7,99.8,100.5,1),(99.2,99.4,99,99.1,1)])
    r=resolve(c,0,direction='LONG',entry=100,stop=99,policy='half_stop_until_1r',bars=2,exit_stress=0)
    assert r['banked'] and r['outcome']=='TP1_STOP'
    assert r['runner_price_return']==pytest.approx(-.008)


def test_one_r_activation_does_not_retroactively_fill_an_earlier_low():
    c=data([(100,100.6,99.7,100.5,1),(100.5,101.2,99.8,101,1),(101,102.2,100.5,102,1)])
    r=resolve(c,0,direction='LONG',entry=100,stop=99,policy='be_1r',bars=3,exit_stress=0)
    assert r['outcome']=='TARGET' and r['runner_price_return']==pytest.approx(.02)


def test_already_crossed_new_stop_uses_observed_close_instead_of_past_level():
    c=data([(100,100.6,99.7,99.8,1)])
    r=resolve(c,0,direction='LONG',entry=100,stop=99,policy='be_tp1',bars=1,exit_stress=0)
    assert r['outcome']=='AMEND_CROSSED_AT_CLOSE'
    assert r['runner_price_return']==pytest.approx(-.002)


def test_stale_terminal_price_is_unknown_and_future_mutation_is_irrelevant():
    c=data([(100,100.1,99.8,100,1),(100,100.2,99.8,100,0),(100,103,98,102,1)])
    r=resolve(c,0,direction='LONG',entry=100,stop=99,policy='be_1r',bars=2)
    assert not r['known'] and r['reason']=='stale_runner_time_exit'
    c['high'][2]=10000
    assert resolve(c,0,direction='LONG',entry=100,stop=99,policy='be_1r',bars=2)==r


def test_structural_stop_ignores_empty_candle_wick_and_preserves_gap_fill():
    c=data([(100,100.1,99.8,100,1),(100,100.1,1,100,0),
        (100,100.7,99.9,100.5,1),(99.5,99.6,99.4,99.5,1)])
    r=resolve(c,2,direction='LONG',entry=100,stop=99,policy='structure_until_1r',bars=2,exit_stress=0)
    assert r['banked'] and r['outcome']=='TP1_STOP'
    assert r['runner_price_return']==pytest.approx(-.005)


@pytest.mark.parametrize('policy',POLICIES)
def test_known_target_is_causal_under_future_changes_and_long_short_reflection(policy):
    rows=[(100,100.7,99.8,100.5,1),(100.5,101.2,100.3,101,1),
        (101,102.2,100.5,102,1),(102,105,95,99,1)]
    c=data(rows)
    r=resolve(c,0,direction='LONG',entry=100,stop=99,policy=policy,bars=3,exit_stress=0)
    assert r['outcome']=='TARGET' and r['banked']
    c['high'][3]=10000;c['low'][3]=1
    assert resolve(c,0,direction='LONG',entry=100,stop=99,policy=policy,bars=3,exit_stress=0)==r
    reflected=data([(200-o,200-l,200-h,200-close,v) for o,h,l,close,v in rows])
    short=resolve(reflected,0,direction='SHORT',entry=100,stop=101,policy=policy,bars=3,exit_stress=0)
    for k in ('runner_price_return','bank_price_return','exit_time'):
        assert short[k]==pytest.approx(r[k])
