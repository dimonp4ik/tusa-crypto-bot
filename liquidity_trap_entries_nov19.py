"""Failed-break acceptance and closed trend pullback hypotheses; no orders."""
from bisect import bisect_right
import math
import numpy as np
from phase_entry_scenarios_nov19 import HigherContext,stock_clock,HORIZON
from price_grid_bracket_nov19 import rounded_bracket

FAMILIES=('failed_break_accept','trend_pullback_hold')


def macro_allows(sign,four,market):
    return four['trend']!=-sign and all(r['trend']!=-sign for r in market)


def candidates(signal,trade,symbol,spec,*,references=(),stocks=False,diagnostics=None):
    diag=diagnostics if diagnostics is not None else {};diag.clear()
    def note(key):diag[key]=diag.get(key,0)+1
    h4=HigherContext(signal,16);h1=HigherContext(signal,4);refs=[HigherContext(r,16) for r in references]
    t=np.asarray(signal['time'],dtype=np.int64)
    o,h,l,c=[np.asarray(signal[k],dtype=float) for k in ('open','high','low','close')]
    if len(t)<1040:return []
    if (np.diff(t)<=0).any() or not all(np.isfinite(v).all() for v in (o,h,l,c)):
        raise ValueError('invalid_trap_history')
    if (l<=0).any() or (l>np.minimum(o,c)).any() or (h<np.maximum(o,c)).any():raise ValueError('invalid_trap_ohlc')
    previous=np.r_[c[0],c[:-1]];tr=np.maximum.reduce((h-l,abs(h-previous),abs(l-previous)))
    atr=np.r_[np.full(13,np.nan),np.convolve(tr,np.ones(14)/14,mode='valid')]
    lookup={int(stamp):i for i,stamp in enumerate(trade['time'])}
    gaps=np.r_[0,np.cumsum(np.diff(t)!=900)];pending=None;records=[]
    for j in range(1040,len(t)):
        stamp=int(t[j]);entry_time=stamp+900;expire=stock_clock(entry_time) if stocks else None
        if stocks and expire is None:pending=None;continue
        if gaps[j]!=gaps[j-1039]:pending=None;continue
        four=h4.at(entry_time);one=h1.at(entry_time);market=[r.at(entry_time) for r in refs]
        if four is None or one is None or any(r is None for r in market):pending=None;continue
        i=lookup.get(stamp)
        if i is None or i<20 or list(trade['time'][i-20:i+1])!=list(range(stamp-20*900,stamp+900,900)):
            pending=None;continue
        volume=np.asarray(trade['volume'][i-20:i+1],dtype=float)
        if not np.isfinite(volume).all() or (volume<0).any():raise ValueError('invalid_trap_volume')
        mean=float(np.mean(volume[:-1]));reference=float(trade['close'][i]);a=float(atr[j])
        if mean<=0 or volume[-1]<=0 or np.count_nonzero(volume[-16:]>0)<12:
            note('trade_activity_rejected');pending=None;continue
        if not math.isfinite(reference) or min(reference,a,c[j])<=0 or abs(reference/c[j]-1)>.001:
            note('basis_rejected');pending=None;continue
        note('data_eligible_bars');ratio=float(volume[-1]/mean);body=float(c[j]-o[j]);span=float(h[j]-l[j])
        emitted=None;had_pending=pending is not None
        if pending is not None:
            p=pending;p['high']=max(p['high'],float(h[j]));p['low']=min(p['low'],float(l[j]))
            fade=-p['break_sign'];age=j-p['opened']
            position=(c[j]-l[j] if fade>0 else h[j]-c[j])/span if span>0 else .5
            if p['stage']=='wait_failure':
                if age>4:note('breakout_expired');pending=None
                elif (fade*(c[j]-p['level'])>.25*a and fade*body>=.4*a and ratio>=1.
                        and position>=.6 and macro_allows(fade,four,market)):
                    p.update(stage='wait_acceptance',failed_at=j);note('failed_break_awaiting_acceptance')
            else:
                if j-p['failed_at']>2 or fade*(c[j]-p['level'])<-.1*a or not macro_allows(fade,four,market):
                    note('failure_not_accepted');pending=None
                elif fade*(c[j]-p['level'])>=.1*a and fade*body>0 and ratio>=.75 and position>=.6:
                    emitted=dict(family='failed_break_accept',sign=fade,opened=p['opened'],level=p['level'],
                        wick=p['low'] if fade>0 else p['high'],opposite=p['opposite'])
                    note('trap_confirmed');pending=None
        if emitted is None:
            sign=one['trend']
            if sign and macro_allows(sign,four,market):
                k=bisect_right(h1.ends,entry_time)-1;zone=float(h1.fast[k])
                touch=min(l[j-2:j])<=zone+.25*a if sign>0 else max(h[j-2:j])>=zone-.25*a
                position=(c[j]-l[j] if sign>0 else h[j]-c[j])/span if span>0 else .5
                if (touch and sign*(c[j-1]-o[j-1])<0 and .1*a<=sign*(c[j]-zone)<=1.5*a
                        and sign*(c[j]-c[j-1])>0 and sign*body>0 and ratio>=.75 and position>=.6):
                    emitted=dict(family='trend_pullback_hold',sign=sign,opened=j-1,level=zone,
                        wick=float(min(l[j-2:j+1]) if sign>0 else max(h[j-2:j+1])),opposite=None)
                    note('pullback_confirmed')
        if pending is None and not had_pending and emitted is None:
            high=float(max(h[j-20:j]));low=float(min(l[j-20:j]))
            sign=1 if c[j]>high+.1*a and body>=.5*a else -1 if c[j]<low-.1*a and body<=-.5*a else 0
            if sign and ratio>=1.2:
                pending=dict(break_sign=sign,level=high if sign>0 else low,opened=j,stage='wait_failure',
                    high=float(h[j]),low=float(l[j]),opposite=float(min(l[j-64:j]) if sign>0 else max(h[j-64:j])))
                note('breakout_observed')
        if emitted is None:continue
        sign=emitted['sign'];entry=reference*(1+sign*.0005);risk_cap=min(.015,max(.008,1.5*a/entry))
        risk=max(.003,sign*(entry-(emitted['wick']-sign*.15*a))/entry)
        if risk>risk_cap:note('stop_distance_rejected');continue
        first=max(.005,min(.01,a/entry));final=max(.01,min(.02,2.5*a/entry))
        ahead=[v for v in (four['unbroken_highs'] if sign>0 else four['unbroken_lows']) if sign*(v-entry)>0]
        if emitted['opposite'] is not None and sign*(emitted['opposite']-entry)>0:ahead.append(emitted['opposite'])
        if ahead:
            level=min(ahead) if sign>0 else max(ahead);final=min(final,sign*(level-entry)/entry-.1*a/entry)
        if final<.008 or final<=first or final/risk<1.5:note('target_space_rejected');continue
        raw=dict(symbol=symbol,entry_time=entry_time,entry_index=i+1,direction='LONG' if sign>0 else 'SHORT',
            entry=entry,sl=entry*(1-sign*risk),tp1=entry*(1+sign*first),tp2=entry*(1+sign*final),
            risk_pct=risk,risk_cap=risk_cap,family=emitted['family'],level=emitted['level'],
            signal_strength=ratio*abs(body)/a,setup_started=int(t[emitted['opened']])+900,
            signal_known_at=entry_time,h1_known_at=one['known_at'],h4_known_at=four['known_at'],
            reference_known_at=[r['known_at'] for r in market],expire_at=expire,hold_bars=HORIZON)
        row=rounded_bracket(raw,spec)
        if row is None:note('price_grid_rejected');continue
        records.append(row);note('signals')
    return records
