import pytest
from stock_dense_minute_path_nov19 import resolve,PROFILES


def source(rows):
    return dict(time=[60*i for i in range(len(rows))],
        **{k:[row[j] for row in rows] for j,k in enumerate(('open','high','low','close','volume'))})


def row(end):return dict(entry_time=0,expire_at=end,direction='LONG',entry=100,sl=99,tp1=100.5,tp2=101.5)


def test_ordered_minutes_recover_target_before_a_later_stop_without_guessing_ohlc_order():
    c=source([(100,100.8,99.9,100.7,1),(100.7,101.6,100.6,101.5,1),(101.5,101.6,98,99,1)])
    r=resolve(c,row(180),PROFILES[0],exit_stress=0)
    assert r['known'] and r['outcome']=='TARGET' and r['banked']
    assert r['exit_time']==120 and r['near_exit_time']==60


def test_new_stop_participates_at_declared_next_minute_or_parent_boundary_only():
    c=source([(100,100.8,99.9,100.6,1),(100.6,100.7,99.8,100.2,1),(100.2,101.6,100.1,101.5,1)])
    slow=resolve(c,row(180),PROFILES[0],exit_stress=0)
    fast=resolve(c,row(180),PROFILES[1],exit_stress=0)
    assert slow['outcome']=='TARGET' and fast['outcome']=='LOCK'
    assert fast['exit_time']==120 and fast['runner_price_return']==0


def test_same_minute_initial_stop_target_ambiguity_remains_stop_first():
    c=source([(100,101,98.5,100.7,1)])
    r=resolve(c,row(60),PROFILES[1],exit_stress=0)
    assert r['outcome']=='SL' and not r['banked']
    assert r['intraminute_stop_target_ambiguities']==1


def test_empty_minute_does_not_trigger_or_bank_and_stale_expiry_is_unknown():
    c=source([(100,102,98,101,0),(100,100.2,99.8,100.1,1),(100.1,100.1,100.1,100.1,0)])
    r=resolve(c,row(180),PROFILES[0])
    assert not r['known'] and r['reason']=='stale_last_minute_at_time_exit'


def test_gap_through_previously_active_be_keeps_the_worse_open_and_future_is_irrelevant():
    c=source([(100,100.8,99.9,100.6,1),(99.5,99.6,99.4,99.5,1),(99.5,101,98,100,1)])
    r=resolve(c,row(120),PROFILES[1],exit_stress=0)
    assert r['outcome']=='LOCK' and r['runner_price_return']==pytest.approx(-.005)
    c['high'][2]=10000
    assert resolve(c,row(120),PROFILES[1],exit_stress=0)==r


def test_parent_stop_activates_at_original_quarter_hour_boundary():
    rows=[(100,100.2,99.8,100,1)]*14+[(100,100.8,99.9,100.6,1),(100.6,100.7,99.8,100.1,1)]
    r=resolve(source(rows),row(960),PROFILES[0],exit_stress=0)
    assert r['outcome']=='LOCK' and r['near_exit_time']==900 and r['exit_time']==960


@pytest.mark.parametrize('profile',PROFILES)
def test_long_short_reflection_preserves_path_returns_and_events(profile):
    rows=[(100,100.8,99.9,100.6,1),(100.6,101.6,100.1,101.5,1)]
    long=resolve(source(rows),row(120),profile,exit_stress=0)
    short_row=dict(row(120),direction='SHORT',sl=101,tp1=99.5,tp2=98.5)
    mirrored=source([(200-o,200-l,200-h,200-c,v) for o,h,l,c,v in rows])
    short=resolve(mirrored,short_row,profile,exit_stress=0)
    assert short['outcome']==long['outcome']
    for key in ('runner_price_return','bank_price_return','exit_time','near_exit_time'):
        assert short[key]==pytest.approx(long[key])
