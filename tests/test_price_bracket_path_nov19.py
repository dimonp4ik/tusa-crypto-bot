import pytest
from price_bracket_path_nov19 import resolve


def data():
    return dict(time=[i*900 for i in range(40)],open=[100.]*40,high=[100.1]*40,
        low=[99.9]*40,close=[100.]*40,volume=[1.]*40)


def test_new_tp1_stop_applies_only_from_following_bar():
    c=data();c['high'][0]=100.8;c['low'][0]=99.5
    out=resolve(c,0,direction='LONG',entry=100,stop=99,tp1=100.6,tp2=101.8,exit_stress=0)
    assert out['outcome']=='LOCK'
    assert out['exit_time']==1800
    assert out['near_exit_time']==900


def test_zero_volume_touch_is_not_a_trade_and_all_zero_future_is_unknown():
    c=data();c['volume']=[0.]*40;c['high'][0]=200
    assert resolve(c,0,direction='LONG',entry=100,stop=99,tp1=100.6,tp2=101.8)['known'] is False


def test_exact_fixed_price_bank_and_runner_returns_for_both_sides():
    for direction,sign in [('LONG',1),('SHORT',-1)]:
        c=data();c['high' if sign>0 else 'low'][0]=100*(1+sign*.02)
        out=resolve(c,0,direction=direction,entry=100,stop=100-sign,tp1=100*(1+sign*.006),tp2=100*(1+sign*.018),exit_stress=0)
        assert out['known'];assert out['outcome']=='TARGET'
        assert out['bank_price_return']==pytest.approx(.006)
        assert out['runner_price_return']==pytest.approx(.018)


def test_time_exit_with_stale_last_trade_is_not_a_free_flat_outcome():
    c=data();c['volume'][31]=0
    out=resolve(c,0,direction='LONG',entry=100,stop=99,tp1=100.6,tp2=101.8)
    assert out==dict(known=False,reason='stale_reference_at_time_exit')
