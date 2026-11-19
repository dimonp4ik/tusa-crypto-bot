"""Causal stop geometry with fixed original TP prices and explicit price paths."""
import math

POLICIES=('control','swing4_widen','swing8_widen','confirmed_pivot_widen')


def stop_geometry(c,index,row,policy):
    if policy not in POLICIES:raise ValueError('unknown_structure_policy')
    entry=float(row['entry']);original=float(row['sl']);sign=1 if row['direction']=='LONG' else -1
    if row['direction'] not in ('LONG','SHORT'):raise ValueError('invalid_structure_direction')
    risk=sign*(entry-original)
    if min(entry,original,risk)<=0 or index<100:raise ValueError('invalid_structure_history')
    stamp=int(row['entry_time'])
    if list(c['time'][index-100:index])!=list(range(stamp-100*900,stamp,900)):
        raise ValueError('missing_closed_structure_history')
    if policy=='control':return dict(stop=original,actual_risk_fraction=risk/entry,changed=False)
    atr=risk/2 # Original risk is exactly two closed ATRs in this frozen pool.
    values=c['low'] if sign>0 else c['high']
    if policy.startswith('swing'):
        n=4 if policy=='swing4_widen' else 8
        level=min(values[index-n:index]) if sign>0 else max(values[index-n:index])
    else:
        pivots=[]
        # Every right confirmation is closed by index-1. Flat repeated values
        # are not treated as an unlimited set of new local extrema.
        for k in range(index-96,index-2):
            if sign>0:
                confirmed=values[k]<min(values[k-2:k]) and values[k]<=min(values[k+1:k+3])
            else:
                confirmed=values[k]>max(values[k-2:k]) and values[k]>=max(values[k+1:k+3])
            if confirmed:pivots.append(values[k])
        if not pivots:return None
        level=pivots[-1]
        if sign*(entry-level)<=0:return None
    proposed=float(level)-sign*.25*atr
    stop=min(original,proposed) if sign>0 else max(original,proposed)
    fraction=sign*(entry-stop)/entry
    if not math.isfinite(stop) or stop<=0 or fraction>.02+1e-12:return None
    return dict(stop=stop,actual_risk_fraction=fraction,changed=abs(stop-original)>1e-12)


def replay_fixed_targets(c,index,*,direction,entry,stop,original_risk,bars=48):
    if direction not in ('LONG','SHORT'):raise ValueError('unknown_direction')
    sign=1 if direction=='LONG' else -1
    if not all(math.isfinite(v) for v in (entry,stop,original_risk)) or min(entry,stop,original_risk)<=0:
        raise ValueError('invalid_fixed_target_geometry')
    if sign*(entry-stop)<=0:raise ValueError('stop_wrong_side')
    if list(c['time'][index:index+bars])!=list(range(c['time'][index],c['time'][index]+bars*900,900)):
        raise ValueError('incomplete_structure_path')
    tp1=entry+sign*original_risk*.5;tp2=entry+sign*original_risk*2
    banked=protected=False;first=None
    for j in range(index,index+bars):
        t=int(c['time'][j])+900;o,h,l,close=[float(c[k][j]) for k in ('open','high','low','close')]
        if not all(math.isfinite(v) and v>0 for v in (o,h,l,close)) or not l<=min(o,close)<=max(o,close)<=h:
            raise ValueError('invalid_structure_ohlc')
        level=entry if protected else stop
        # A later bar's known opening price is ordered before its range.
        # The initial bar opening may precede publication and is not used to
        # claim that either resting TP already filled before market entry.
        if j>index and sign*(o-tp2)>=0:
            return dict(outcome='TP_RUNNER',runner_r=2.,banked=True,exit_time=t,near_exit_time=first or t)
        if j>index and not banked and sign*(o-tp1)>=0:
            banked=True;first=t
        stop_hit=l<=level if sign>0 else h>=level
        if stop_hit:
            fill=min(o,level) if sign>0 else max(o,level)
            kind='LOCK' if protected else 'TP1_STOP' if banked else 'SL'
            return dict(outcome=kind,runner_r=sign*(fill-entry)/original_risk,
                banked=banked,exit_time=t,near_exit_time=first)
        first_hit=h>=tp1 if sign>0 else l<=tp1
        final_hit=h>=tp2 if sign>0 else l<=tp2
        if first_hit and first is None:first=t
        if final_hit:return dict(outcome='TP_RUNNER',runner_r=2.,banked=True,exit_time=t,near_exit_time=first)
        banked=banked or first_hit;protected=protected or banked
    return dict(outcome='TP_EXPIRED' if banked else 'EXPIRED',runner_r=sign*(close-entry)/original_risk,
        banked=banked,exit_time=t,near_exit_time=first)
