import copy
import numpy as np
import liquidity_trap_entries_nov19 as module


def data():
    n=1200;start=1779062400
    return dict(time=[start+i*900 for i in range(n)],open=[100.]*n,close=[100.]*n,
        high=[100.2]*n,low=[99.8]*n,volume=[10.]*n)


def mock_context(monkeypatch,one_trend=0,four_trend=0):
    class Context:
        def __init__(self,c,factor):
            self.factor=factor;self.ends=[t+900 for t in c['time']];self.fast=np.full(len(c['time']),100.)
        def at(self,t):return dict(known_at=t,trend=one_trend if self.factor==4 else four_trend,
            unbroken_highs=[],unbroken_lows=[])
    monkeypatch.setattr(module,'HigherContext',Context)


def trap_source():
    c=data();c['low'][1080]=97
    for j,values in [(1100,dict(open=100,high=100.5,low=100,close=100.4,volume=20)),
        (1101,dict(open=100.35,high=100.4,low=99.9,close=100.,volume=15)),
        (1102,dict(open=100.05,high=100.08,low=99.94,close=99.98,volume=12))]:
        for k,v in values.items():c[k][j]=v
    return c


def test_failed_break_needs_return_and_a_separate_acceptance_close(monkeypatch):
    mock_context(monkeypatch);c=trap_source()
    before={k:v[:1102] for k,v in c.items()}
    assert module.candidates(before,before,'TEST',{'tickSz':.01})==[]
    rows=module.candidates(c,c,'TEST',{'tickSz':.01})
    chosen=[r for r in rows if r['setup_started']==c['time'][1101]]
    assert len(chosen)==1
    assert chosen[0]['family']=='failed_break_accept'
    assert chosen[0]['direction']=='SHORT'
    assert chosen[0]['entry_time']==c['time'][1103]
    assert chosen[0]['sl']>100.5


def test_future_changes_cannot_create_an_earlier_trap_signal(monkeypatch):
    mock_context(monkeypatch);c=trap_source();before=module.candidates(c,c,'TEST',{'tickSz':.01})
    assert any(r['family']=='failed_break_accept' for r in before)
    changed=copy.deepcopy(c)
    for k in ('open','high','low','close'):changed[k][1103:]=[1000.]*97
    after=module.candidates(changed,changed,'TEST',{'tickSz':.01})
    cutoff=c['time'][1103]
    assert [r for r in before if r['entry_time']<=cutoff]==[r for r in after if r['entry_time']<=cutoff]


def test_strong_opposing_four_hour_trend_blocks_trap_fade(monkeypatch):
    mock_context(monkeypatch,four_trend=1);c=trap_source()
    assert not [r for r in module.candidates(c,c,'TEST',{'tickSz':.01}) if r['direction']=='SHORT']


def test_trend_pullback_uses_closed_countermove_then_reclaim_of_hourly_zone(monkeypatch):
    mock_context(monkeypatch,one_trend=1,four_trend=1);c=data()
    for j,values in [(1100,dict(open=100.15,high=100.2,low=99.85,close=99.95)),
        (1101,dict(open=99.95,high=100.2,low=99.9,close=100.15,volume=12))]:
        for k,v in values.items():c[k][j]=v
    rows=module.candidates(c,c,'TEST',{'tickSz':.01})
    assert any(r['family']=='trend_pullback_hold' and r['entry_time']==c['time'][1102] for r in rows)
