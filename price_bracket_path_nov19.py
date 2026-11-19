"""Price-denominated bracket paths; no position sizing or inferred private fills."""
import math


def resolve(c,index,*,direction,entry,stop,tp1,tp2,bars=32,expire_at=None,
        require_trade_volume=True,exit_stress=.0005):
    if direction not in ('LONG','SHORT'):raise ValueError('unknown_price_direction')
    sign=1 if direction=='LONG' else -1
    if (index<0 or bars<1 or index+bars>len(c['time']) or not 0<=exit_stress<=.001
            or not all(math.isfinite(v) and v>0 for v in (entry,stop,tp1,tp2))
            or not sign*(entry-stop)>0 or not 0<sign*(tp1-entry)<sign*(tp2-entry)):
        raise ValueError('invalid_price_bracket')
    start=c['time'][index]
    if list(c['time'][index:index+bars])!=list(range(start,start+bars*900,900)):
        return dict(known=False,reason='missing_price_path')
    banked=protected=False;first=None;last=None;last_index=None;last_expected=None
    for j in range(index,index+bars):
        stamp=int(c['time'][j])
        if expire_at is not None and stamp>=expire_at:break
        last_expected=j
        o,h,l,close=[float(c[k][j]) for k in ('open','high','low','close')]
        if not all(math.isfinite(v) and v>0 for v in (o,h,l,close)) or not l<=min(o,close)<=max(o,close)<=h:
            return dict(known=False,reason='invalid_path_prices')
        if require_trade_volume:
            volume=float(c['volume'][j])
            if not math.isfinite(volume) or volume<0:return dict(known=False,reason='invalid_path_volume')
            if volume==0:continue
        last=close;last_index=j;t=stamp+900;level=entry if protected else stop
        if j>index and sign*(o-tp2)>=0:
            price=tp2;code='TARGET';banked=True;first=first or t;break
        if j>index and not banked and sign*(o-tp1)>=0:banked=True;first=t
        hit=l<=level if sign>0 else h>=level
        if hit:
            price=min(o,level) if sign>0 else max(o,level)
            code='LOCK' if protected else 'TP1_STOP' if banked else 'SL';break
        first_hit=h>=tp1 if sign>0 else l<=tp1
        final_hit=h>=tp2 if sign>0 else l<=tp2
        if first_hit and first is None:first=t
        if final_hit:price=tp2;code='TARGET';banked=True;break
        banked=banked or first_hit
        # The newly observed TP1 does not retroactively amend this bar's stop.
        protected=protected or banked
    else:
        code='TIME';price=last;t=int(c['time'][index+bars-1])+900
    if 'code' not in locals():
        code='TIME';price=last;t=int(expire_at)
    if price is None:return dict(known=False,reason='no_positive_volume_after_entry')
    if code=='TIME' and last_index!=last_expected:
        return dict(known=False,reason='stale_reference_at_time_exit')
    fill=price*(1-sign*exit_stress)
    bank_fill=tp1*(1-sign*exit_stress)
    return dict(known=True,outcome=code,exit_time=t,exit_reference=fill,
        runner_price_return=sign*(fill/entry-1),bank_price_return=sign*(bank_fill/entry-1),
        banked=banked,near_exit_time=first,trigger_bar=int(c['time'][j]))
