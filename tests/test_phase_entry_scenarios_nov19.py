import copy
import math
import pytest
import phase_entry_scenarios_nov19 as module


def source(n=1400):
    start=1779062400 # UTC-aligned15m timestamps.
    close=[100+.001*j+.03*math.sin(j) for j in range(n)]
    return dict(time=[start+900*j for j in range(n)],open=close.copy(),close=close,
        high=[v+.1 for v in close],low=[v-.1 for v in close],volume=[10.]*n)


def test_higher_timeframe_context_never_reads_a_forming_or_future_bar():
    c=source();entry=c['time'][1200]
    before=module.HigherContext(c,16).at(entry)
    assert before is not None and before['known_at']<=entry
    changed=copy.deepcopy(c)
    for k in ('open','high','low','close'):changed[k][1200:]=[1000.]*200
    assert module.HigherContext(changed,16).at(entry)==before


def test_level_closed_through_after_formation_is_not_unbroken_resistance():
    c=source(320);c['time']=[i*900 for i in range(320)]
    for hour in range(80):
        price=100+hour*.01
        if hour==65:price=106
        if hour==70:price=104
        for j in range(hour*4,hour*4+4):
            c['open'][j]=c['close'][j]=price;c['high'][j]=price+.2;c['low'][j]=price-.2
    for j in range(60*4,60*4+4):c['high'][j]=105
    context=module.HigherContext(c,4)
    before=context.at(65*3600);after=context.at(71*3600)
    assert 105 in before['unbroken_highs']
    assert 105 in after['pivot_highs']
    assert 105 not in after['unbroken_highs']


def controlled_source():
    c=source(1200)
    c['open']=[100.]*1200;c['close']=[100.]*1200;c['high']=[100.1]*1200;c['low']=[99.9]*1200
    return c


def mock_context(monkeypatch,trend):
    class Context:
        def __init__(self,c,factor):self.factor=factor
        def at(self,t):return dict(known_at=t,trend=trend,range=trend==0,efficiency=.2 if trend==0 else .5,
            gap_atr=0,pivot_highs=[],pivot_lows=[],unbroken_highs=[],unbroken_lows=[])
    monkeypatch.setattr(module,'HigherContext',Context)


def test_range_sweep_waits_for_later_closed_confirmation_and_is_causal(monkeypatch):
    mock_context(monkeypatch,0);c=controlled_source();c['high'][1090]=103
    for k,v in dict(open=100,high=100.2,low=99.75,close=100.1,volume=20).items():c[k][1100]=v
    for k,v in dict(open=100.1,high=100.3,low=99.9,close=100.2,volume=15).items():c[k][1101]=v
    rows=module.candidates(c,c,'TEST')
    selected=[r for r in rows if r['entry_time']==c['time'][1102]]
    assert len(selected)==1 and selected[0]['family']=='range_sweep_confirm'
    assert selected[0]['setup_started']==c['time'][1101]
    changed=copy.deepcopy(c)
    for k in ('open','high','low','close'):changed[k][1102:]=[1000.]*98
    later=module.candidates(changed,changed,'TEST')
    assert [r for r in later if r['entry_time']==c['time'][1102]]==selected


def test_trend_breakout_needs_a_retest_then_another_confirmation(monkeypatch):
    mock_context(monkeypatch,1);c=controlled_source()
    for k,v in dict(open=100,high=100.6,low=99.95,close=100.5,volume=20).items():c[k][1100]=v
    for k,v in dict(open=100.1,high=100.3,low=100.05,close=100.25,volume=15).items():c[k][1101]=v
    for k,v in dict(open=100.25,high=100.4,low=100.15,close=100.35,volume=15).items():c[k][1102]=v
    rows=module.candidates(c,c,'TEST')
    selected=[r for r in rows if r['setup_started']==c['time'][1101]]
    assert len(selected)==1
    assert selected[0]['entry_time']==c['time'][1103]
    assert selected[0]['family']=='trend_retest_confirm'


def test_intraday_trend_enters_after_the_closed_retest_without_chasing_a_third_bar(monkeypatch):
    mock_context(monkeypatch,1);c=controlled_source()
    for k,v in dict(open=100,high=100.6,low=99.95,close=100.5,volume=20).items():c[k][1100]=v
    for k,v in dict(open=100.1,high=100.3,low=100.05,close=100.25,volume=15).items():c[k][1101]=v
    diag={};rows=module.candidates(c,c,'TEST',profile='intraday_context',diagnostics=diag)
    selected=[r for r in rows if r['setup_started']==c['time'][1101]]
    assert len(selected)==1
    assert selected[0]['entry_time']==c['time'][1102]
    assert selected[0]['target1_fraction']>=.005
    assert diag['signals']>=1
