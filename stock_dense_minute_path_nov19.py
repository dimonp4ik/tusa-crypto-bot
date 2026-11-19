"""Ordered minute brackets with declared amendment latency; no guessed ticks."""
from bisect import bisect_left
import math

PROFILES=('amend_next15m_parent_bar','amend_next1m_bar')


def resolve(c,row,profile,*,exit_stress=.0005):
    if profile not in PROFILES:raise ValueError('unknown_dense_minute_amendment_profile')
    if row['direction'] not in ('LONG','SHORT'):raise ValueError('unknown_dense_minute_direction')
    sign=1 if row['direction']=='LONG' else -1
    entry,stop,tp1,tp2=[float(row[k]) for k in ('entry','sl','tp1','tp2')]
    if not all(math.isfinite(x) and x>0 for x in (entry,stop,tp1,tp2)) or not sign*(entry-stop)>0:
        raise ValueError('invalid_dense_minute_geometry')
    if not 0<sign*(tp1-entry)<sign*(tp2-entry) or not 0<=exit_stress<=.001:
        raise ValueError('invalid_dense_minute_target')
    start=int(row['entry_time']);end=int(row['expire_at']);i=bisect_left(c['time'],start);last_end=bisect_left(c['time'],end)
    if start%60 or end%60 or end<=start or c['time'][i:last_end]!=list(range(start,end,60)):
        return dict(known=False,reason='missing_minute_horizon')
    banked=False;first=None;activation=math.inf;last=None;last_index=None;ambiguous=0
    for j in range(i,last_end):
        stamp=int(c['time'][j]);t=stamp+60
        o,h,l,close=[float(c[k][j]) for k in ('open','high','low','close')]
        if not all(math.isfinite(x) and x>0 for x in (o,h,l,close)) or not l<=min(o,close)<=max(o,close)<=h:
            return dict(known=False,reason='invalid_minute_ohlc')
        volume=float(c['volume'][j])
        if not math.isfinite(volume) or volume<0:return dict(known=False,reason='invalid_minute_volume')
        if not volume:continue
        last=close;last_index=j;protected=stamp>=activation;level=entry if protected else stop
        if j>i and sign*(o-tp2)>=0:
            price=tp2;code='TARGET';banked=True;first=first or t;break
        if j>i and not banked and sign*(o-tp1)>=0:
            banked=True;first=t
            activation=(stamp//900*900+900 if profile==PROFILES[0] else stamp+60)
        hit_stop=l<=level if sign>0 else h>=level
        target=tp2 if banked else tp1;hit_target=h>=target if sign>0 else l<=target
        if hit_stop and hit_target:ambiguous+=1
        if hit_stop:
            price=min(o,level) if sign>0 else max(o,level)
            code='LOCK' if protected else 'TP1_STOP' if banked else 'SL';break
        first_hit=h>=tp1 if sign>0 else l<=tp1
        if first_hit and not banked:
            banked=True;first=t
            activation=(stamp//900*900+900 if profile==PROFILES[0] else stamp+60)
        final_hit=h>=tp2 if sign>0 else l<=tp2
        if final_hit:price=tp2;code='TARGET';banked=True;break
    else:
        if last is None:return dict(known=False,reason='no_positive_minute_after_entry')
        if last_index!=last_end-1:return dict(known=False,reason='stale_last_minute_at_time_exit')
        price=last;code='TIME';t=end
    fill=price*(1-sign*exit_stress);bank_fill=tp1*(1-sign*exit_stress)
    return dict(known=True,outcome=code,exit_time=t,banked=banked,near_exit_time=first,
        runner_price_return=sign*(fill/entry-1),bank_price_return=sign*(bank_fill/entry-1),
        intraminute_stop_target_ambiguities=ambiguous,trigger_minute=int(c['time'][last_index]))
