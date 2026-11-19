"""Paired runner protection after a real-sized first bank; public-price proxies."""
import math

POLICIES=('be_tp1','half_stop_until_1r','be_1r','structure_until_1r')


def resolve(c,index,*,direction,entry,stop,policy,bars=48,exit_stress=.0005):
    if policy not in POLICIES:raise ValueError('unknown_banked_runner_policy')
    if direction not in ('LONG','SHORT'):raise ValueError('unknown_banked_runner_direction')
    sign=1 if direction=='LONG' else -1
    if not all(math.isfinite(x) and x>0 for x in (entry,stop)):
        raise ValueError('invalid_banked_runner_prices')
    risk=sign*(entry-stop)
    if risk<=0 or index<0 or bars<1 or not 0<=exit_stress<=.001:
        raise ValueError('invalid_banked_runner_geometry')
    if index+bars>len(c['time']):return dict(known=False,reason='incomplete_runner_horizon')
    start=int(c['time'][index])
    if list(c['time'][index:index+bars])!=list(range(start,start+bars*900,900)):
        return dict(known=False,reason='noncontiguous_runner_horizon')
    tp1=entry+sign*risk*.5;tp2=entry+sign*risk*2
    level=stop;banked=False;first=None;last=None;last_index=None
    for j in range(index,index+bars):
        stamp=int(c['time'][j]);t=stamp+900
        o,h,l,close=[float(c[k][j]) for k in ('open','high','low','close')]
        if not all(math.isfinite(x) and x>0 for x in (o,h,l,close)) or not l<=min(o,close)<=max(o,close)<=h:
            return dict(known=False,reason='invalid_runner_ohlc')
        volume=float(c['volume'][j])
        if not math.isfinite(volume) or volume<0:return dict(known=False,reason='invalid_runner_volume')
        if not volume:continue
        last=close;last_index=j
        # Later known opening trades precede that bar's range. The initial
        # opening print may predate publication, so it receives no TP priority.
        if j>index and sign*(o-tp2)>=0:
            price=tp2;code='TARGET';banked=True;first=first or t;break
        if j>index and not banked and sign*(o-tp1)>=0:banked=True;first=t
        if l<=level if sign>0 else h>=level:
            price=min(o,level) if sign>0 else max(o,level)
            code='SL' if not banked else 'LOCK' if sign*(level-entry)>=-1e-12 else 'TP1_STOP'
            break
        first_hit=h>=tp1 if sign>0 else l<=tp1
        if first_hit and first is None:first=t
        if h>=tp2 if sign>0 else l<=tp2:
            price=tp2;code='TARGET';banked=True;break
        banked=banked or first_hit
        if not banked:continue
        new=level
        if policy=='be_tp1':new=entry
        else:
            favorable=sign*((h if sign>0 else l)-entry)/risk
            if favorable>=1:new=entry
            elif policy=='half_stop_until_1r':new=entry-sign*.5*risk
            elif policy=='structure_until_1r':
                # Two completed candles only. Clamp between initial SL and BE;
                # use 0.15 of the original ATR (initial stop = two ATR).
                past=[k for k in range(max(0,j-7),j+1) if c['volume'][k]>0][-2:]
                if len(past)==2:
                    extreme=min(c['low'][k] for k in past) if sign>0 else max(c['high'][k] for k in past)
                    candidate=extreme-sign*.075*risk
                    new=min(entry,candidate) if sign>0 else max(entry,candidate)
        new=max(level,new) if sign>0 else min(level,new)
        # A newly calculated stop cannot fill at an earlier wick. If already
        # crossed at the confirming close, mark a market exit at that close.
        if sign*(new-level)>1e-12 and sign*(close-new)<=0:
            price=close;code='AMEND_CROSSED_AT_CLOSE';break
        level=new
    else:
        if last is None:return dict(known=False,reason='no_positive_volume_after_entry')
        if last_index!=index+bars-1:return dict(known=False,reason='stale_runner_time_exit')
        price=last;code='TIME';t=int(c['time'][index+bars-1])+900
    fill=price*(1-sign*exit_stress);bank_fill=tp1*(1-sign*exit_stress)
    return dict(known=True,outcome=code,exit_time=t,banked=banked,near_exit_time=first,
        runner_price_return=sign*(fill/entry-1),bank_price_return=sign*(bank_fill/entry-1),
        exit_reference=fill,trigger_bar=int(c['time'][last_index]))
