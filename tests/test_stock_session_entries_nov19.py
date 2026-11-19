from copy import deepcopy
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import pytest
from stock_session_entries_nov19 import SessionContext,candidates,macro_allowed
from us_cash_session_nov19 import session_bounds,ET


def stamp(text):return datetime.fromisoformat(text).timestamp()


@pytest.mark.parametrize('text',['2026-07-03T15:00:00+00:00','2026-09-07T15:00:00+00:00',
    '2026-11-26T15:00:00+00:00','2026-07-05T15:00:00+00:00'])
def test_official_full_holidays_and_weekend_have_no_cash_session(text):
    assert session_bounds(stamp(text)) is None


def test_regular_hours_follow_dst_and_early_closes_without_false_july2_short_session():
    july=session_bounds(stamp('2026-07-02T12:00:00+00:00'))
    november=session_bounds(stamp('2026-11-09T12:00:00+00:00'))
    early=session_bounds(stamp('2026-11-27T12:00:00+00:00'))
    assert datetime.fromtimestamp(july[0],timezone.utc).hour==13
    assert datetime.fromtimestamp(november[0],timezone.utc).hour==14
    assert july[1]-july[0]==6.5*3600
    assert early[1]-early[0]==3.5*3600
    with pytest.raises(ValueError,match='year_not_supported'):session_bounds(stamp('2027-01-04T15:00:00+00:00'))


class Macro:
    def __init__(self,direction=1):self.direction=direction
    def at(self,known_at):
        return dict(session_return_atr=self.direction,close=100+self.direction,vwap=100,atr=1)


def source():
    start=stamp('2026-07-06T13:30:00+00:00');times=[int(start)+900*i for i in range(96*5)]
    c=dict(time=times,open=[100.]*len(times),high=[100.3]*len(times),low=[99.7]*len(times),
        close=[100.]*len(times),volume=[1.]*len(times))
    for t in times:
        local=datetime.fromtimestamp(t,ET)
        if local.hour==9 and local.minute==30:
            i=times.index(t);c['high'][i]=103;c['low'][i]=98
    target=times.index(int(stamp('2026-07-09T13:30:00+00:00')))
    def put(offset,bar):
        for key,value in zip(('open','high','low','close','volume'),bar):c[key][target+offset]=value
    put(0,(100,100.6,99.2,100,1));put(1,(100,100.8,99.8,100.3,1))
    return c,target,put


def event_rows(c,macro=1):
    mark={k:v for k,v in c.items() if k!='volume'}
    return candidates(SessionContext(mark,c),'TESTUSDT',dict(tickSz='.001'),references=[Macro(macro)])


def test_opening_retest_is_separate_closed_candle_and_future_mutation_is_nonvacuous():
    c,j,put=source()
    put(2,(100.7,101.2,100.65,101.1,2));put(3,(100.8,101.2,100.7,101.1,1.5))
    cutoff=c['time'][j+3]+900
    rows=[r for r in event_rows(c) if r['entry_time']<=cutoff and r['entry_time']>c['time'][j]]
    assert any(r['family']=='open_range_retest' and r['entry_time']==cutoff for r in rows)
    changed=deepcopy(c)
    for k in ('open','high','low','close'):
        changed[k][j+4:]=[500.]*len(changed[k][j+4:])
    assert [r for r in event_rows(changed) if c['time'][j]<r['entry_time']<=cutoff]==rows


def test_failed_opening_break_requires_failure_then_a_further_acceptance():
    c,j,put=source()
    put(2,(100.7,101.05,100.65,101,2));put(3,(101,101.05,100.55,100.65,2))
    put(4,(100.7,100.75,100.4,100.53,1.5))
    rows=event_rows(c,macro=0)
    assert any(r['family']=='open_range_failure' and r['entry_time']==c['time'][j+4]+900 for r in rows)
    assert not any(r['family']=='open_range_failure' and r['entry_time']==c['time'][j+3]+900 for r in rows)


def test_vwap_pullback_uses_current_session_trade_volume_and_closed_reclaim():
    c,j,put=source()
    put(2,(100.3,100.8,100.25,100.7,1));put(3,(100.7,100.9,100.5,100.8,1))
    put(4,(100.7,100.75,100.2,100.3,1));put(5,(100.4,101,100.3,100.9,1))
    rows=event_rows(c)
    assert any(r['family']=='session_vwap_pullback' and r['entry_time']==c['time'][j+5]+900 for r in rows)


def test_inverse_macro_is_reversed_and_missing_context_never_passes():
    assert macro_allowed([Macro(-1)],1,1,inverse=True)
    assert not macro_allowed([Macro(-1)],1,1,inverse=False)
    assert not macro_allowed([],1,1)


def test_empty_opening_bar_and_no_current_trade_cannot_generate_an_entry():
    c,j,put=source()
    put(2,(100.7,101.2,100.65,101.1,2));put(3,(100.8,101.2,100.7,101.1,1.5))
    cutoff=c['time'][j+3]+900
    c['volume'][j]=0
    assert not [r for r in event_rows(c) if c['time'][j]<r['entry_time']<=cutoff]
