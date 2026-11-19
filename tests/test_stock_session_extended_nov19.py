from copy import deepcopy
from datetime import datetime,timezone
from stock_session_entries_nov19 import SessionContext
from stock_session_extended_nov19 import candidates,MarkSessionMacro
from test_stock_session_entries_nov19 import source,Macro


def test_mark_reference_is_available_without_a_reference_trade_but_own_entry_is_not():
    c,j,put=source();mark={k:list(v) for k,v in c.items() if k!='volume'}
    cutoff=c['time'][j+3]+900
    c['volume'][j:j+4]=[0.]*4
    own=SessionContext(mark,c)
    assert own.at(cutoff) is None
    point=MarkSessionMacro(own).at(cutoff)
    assert point is not None and point['known_at']==cutoff
    assert 'not VWAP' in point['reference_kind']
    changed=deepcopy(mark)
    for k in ('open','high','low','close'):changed[k][j+4:]=[500.]*len(changed[k][j+4:])
    assert MarkSessionMacro(SessionContext(changed,c)).at(cutoff)==point


def test_first_impulse_uses_a_new_crossing_and_does_not_repeat_outside_the_range():
    c,j,put=source()
    put(2,(100.7,101.2,100.65,101.1,2));put(3,(101.1,101.6,101,101.5,2))
    # A rising reference with actual mark prices; no lookahead required.
    ref,_,ref_put=source()
    ref_put(0,(100,100.2,99.9,100.1,1));ref_put(1,(100.1,100.7,100,100.6,1))
    ref_put(2,(100.6,101.3,100.5,101.2,1));ref_put(3,(101.2,101.5,101,101.4,1))
    def run(data):
        return candidates(SessionContext({k:v for k,v in data.items() if k!='volume'},data),
            'TESTUSDT',dict(tickSz='.001'),references=[SessionContext({k:v for k,v in ref.items() if k!='volume'},ref)])
    cutoff=c['time'][j+3]+900
    events=[r for r in run(c) if c['time'][j]<r['entry_time']<=cutoff]
    impulses=[r for r in events if r['family']=='open_range_impulse']
    assert len(impulses)==1 and impulses[0]['entry_time']==c['time'][j+2]+900
    changed=deepcopy(c)
    for k in ('open','high','low','close'):changed[k][j+4:]=[500.]*len(changed[k][j+4:])
    assert [r for r in run(changed) if c['time'][j]<r['entry_time']<=cutoff]==events
