"""Closed US-session opening range and native-volume-weighted price hypotheses."""
from bisect import bisect_right
from collections import defaultdict,Counter
import math
import numpy as np
from us_cash_session_nov19 import session_bounds
from price_grid_bracket_nov19 import rounded_bracket

FAMILIES=('open_range_failure','open_range_retest','session_vwap_pullback')
HORIZON=32


class SessionContext:
    def __init__(self,mark,trade):
        self.mark=mark;self.trade=trade
        self.lookup={int(t):i for i,t in enumerate(trade['time'])}
        self.mark_lookup={int(t):i for i,t in enumerate(mark['time'])}
        self.days=defaultdict(list);self.regular=[];self.previous={};self.memo={}
        for candle in (mark,trade):
            keys=('time','open','high','low','close')+(('volume',) if candle is trade else ())
            if len({len(candle[k]) for k in keys})!=1:raise ValueError('session_source_length_mismatch')
            if any(b<=a for a,b in zip(candle['time'],candle['time'][1:])):
                raise ValueError('unordered_session_source')
            for i in range(len(candle['time'])):
                o,h,l,c=[float(candle[k][i]) for k in ('open','high','low','close')]
                if not all(math.isfinite(x) and x>0 for x in (o,h,l,c)) or not l<=min(o,c)<=max(o,c)<=h:
                    raise ValueError('invalid_session_source_ohlc')
                if candle is trade and (not math.isfinite(float(candle['volume'][i])) or candle['volume'][i]<0):
                    raise ValueError('invalid_session_trade_volume')
        for i,t in enumerate(trade['time']):
            bounds=session_bounds(t)
            if bounds and bounds[0]<=t<bounds[1]:
                self.days[bounds[0]].append(i);self.regular.append(i)
        prior=None
        for start,indices in sorted(self.days.items()):
            self.previous[start]=prior
            bounds=session_bounds(start);expected=list(range(int(start),int(bounds[1]),900))
            if [int(trade['time'][i]) for i in indices]==expected:
                mids=[self.mark_lookup.get(t) for t in expected]
                if all(j is not None for j in mids):
                    prior=dict(high=max(float(mark['high'][j]) for j in mids),
                        low=min(float(mark['low'][j]) for j in mids),known_at=bounds[1])

    def at(self,known_at):
        i=bisect_right(self.trade['time'],known_at-900)-1
        if i<0 or self.trade['time'][i]+900!=known_at:return None
        if i in self.memo:return self.memo[i]
        t=int(self.trade['time'][i]);bounds=session_bounds(t)
        if not bounds or not bounds[0]<=t<bounds[1]:return None
        full=self.days[bounds[0]];upto=bisect_right(full,i);indices=full[:upto]
        expected=list(range(int(bounds[0]),t+900,900))
        if [int(self.trade['time'][k]) for k in indices]!=expected:return None
        if any(stamp not in self.mark_lookup for stamp in expected):return None
        active=[k for k in indices if self.trade['volume'][k]>0]
        if not active or active[-1]!=i or len(active)<math.ceil(.75*len(indices)):return None
        history=self.regular[:bisect_right(self.regular,i)]
        active_history=[k for k in history if self.trade['volume'][k]>0]
        if len(active_history)<21:return None
        recent=active_history[-15:];tr=[]
        for before,k in zip(recent,recent[1:]):
            previous=self.trade['close'][before]
            tr.append(max(self.trade['high'][k]-self.trade['low'][k],
                abs(self.trade['high'][k]-previous),abs(self.trade['low'][k]-previous)))
        atr=float(np.mean(tr))
        if not atr>0:return None
        volumes=[float(self.trade['volume'][k]) for k in active]
        typical=[sum(float(self.trade[key][k]) for key in ('high','low','close'))/3 for k in active]
        vwap=float(np.dot(volumes,typical)/sum(volumes))
        m=self.mark_lookup[t];close=float(self.trade['close'][i]);mark_close=float(self.mark['close'][m])
        if abs(close/mark_close-1)>.001:return None
        prior_volume=float(np.mean([self.trade['volume'][k] for k in active_history[-21:-1]]))
        if prior_volume<=0:return None
        first=indices[0];opening=float(self.trade['open'][first])
        result=dict(index=i,mark_index=m,start=bounds[0],end=bounds[1],known_at=known_at,
            indices=indices,atr=atr,vwap=vwap,close=close,opening=opening,
            session_return_atr=(close-opening)/atr,volume_ratio=self.trade['volume'][i]/prior_volume,
            previous=self.previous[bounds[0]])
        self.memo[i]=result;return result


def macro_allowed(references,known_at,sign,*,inverse=False,reversal=False):
    exposure=sign*(-1 if inverse else 1)
    points=[context.at(known_at) for context in references]
    if not points or any(p is None for p in points):return False
    for p in points:
        directional=exposure*p['session_return_atr']
        relative=exposure*(p['close']-p['vwap'])/p['atr']
        if reversal:
            if directional<-.5 or relative<-.5:return False
        elif directional<.25 or relative<0:return False
    return True


def bracket(point,mark,trade,symbol,spec,sign,extreme,family,or_high,or_low):
    i=point['index'];m=point['mark_index'];atr=point['atr']
    entry=point['close']*(1+sign*.0005)
    stop=extreme-sign*.15*atr;risk=max(.003,sign*(entry-stop)/entry)
    cap=min(.015,max(.008,1.5*atr/entry))
    if not .003<=risk<=cap:return None
    first=min(.01,max(.005,atr/entry));final=min(.02,max(.01,2.5*atr/entry))
    ahead=[]
    if family=='open_range_failure':ahead.append(or_low if sign<0 else or_high)
    else:
        level=or_high if sign>0 else or_low
        if sign*(level-entry)>0:ahead.append(level)
    previous=point['previous']
    if previous is not None and previous['known_at']<=point['known_at']:
        level=previous['high'] if sign>0 else previous['low']
        if sign*(level-entry)>0:ahead.append(level)
    if ahead:final=min(final,min(sign*(x-entry)/entry for x in ahead))
    if final<.008 or final<=first or final/risk<1.5:return None
    width=or_high-or_low
    row=dict(symbol=symbol,direction='LONG' if sign>0 else 'SHORT',family=family,
        entry_time=point['known_at'],entry_index=i+1,entry=entry,sl=entry*(1-sign*risk),
        tp1=entry*(1+sign*first),tp2=entry*(1+sign*final),risk_pct=risk,risk_cap=cap,
        expire_at=point['end'],signal_strength=abs(mark['close'][m]-mark['open'][m])/atr,
        signal_known_at=point['known_at'],session_open=point['start'],session_close=point['end'],
        session_vwap=point['vwap'],vwap_kind='native trade typical-bar-price weighted by native contract volume',
        opening_range_high=or_high,opening_range_low=or_low,opening_range_fraction=width/entry,
        prior_session_known_at=previous['known_at'] if previous is not None else None)
    return rounded_bracket(row,spec)


def candidates(context,symbol,spec,*,references,diagnostics=None):
    mark=context.mark;trade=context.trade;events=[];pending=None;failure=None;day=None
    diag=diagnostics if diagnostics is not None else {};diag.clear()
    def note(key):diag[key]=diag.get(key,0)+1
    for i,stamp in enumerate(trade['time']):
        known_at=int(stamp)+900;point=context.at(known_at)
        if point is None:continue
        if point['start']!=day:day=point['start'];pending=failure=None
        # Wait until the first30 minutes are fully known. Never open during the
        # final30 minutes, including official early-close sessions.
        if not day+1800<=known_at<min(day+5*3600,point['end']-1800):continue
        indices=point['indices']
        if len(indices)<3:continue
        initial=indices[:2]
        if any(trade['volume'][k]<=0 for k in initial):note('opening_no_trade');continue
        mids=[context.mark_lookup[trade['time'][k]] for k in initial]
        or_high=max(float(mark['high'][m]) for m in mids);or_low=min(float(mark['low'][m]) for m in mids)
        if not .002<=(or_high-or_low)/point['close']<=.04:note('opening_range_width');continue
        m=point['mark_index'];atr=point['atr']
        o,h,l,close=[float(mark[k][m]) for k in ('open','high','low','close')]
        width=h-l;choices=[];note('session_eligible')
        if failure is not None:
            f=failure;sign=-f['sign'];level=f['level']
            if i-f['failed_i']>2:failure=None;note('failure_acceptance_expired')
            elif i>f['failed_i'] and sign*(close-level)>=.1*atr and sign*(close-o)>0:
                location=(close-l if sign>0 else h-close)/width if width>0 else .5
                if location>=.6 and point['volume_ratio']>=.75:
                    if macro_allowed(references,known_at,sign,inverse=symbol=='SOXSUSDT',reversal=True):
                        extreme=min(f['extreme'],l) if sign>0 else max(f['extreme'],h)
                        choices.append((FAMILIES[0],sign,extreme))
                    else:note('failure_macro_rejected')
                    failure=None
        if pending is not None:
            p=pending;sign=p['sign'];level=p['level']
            if i-p['i']>4:pending=None;note('range_pending_expired')
            elif i>p['i']:
                p['extreme']=max(p['extreme'],h) if sign>0 else min(p['extreme'],l)
                if -sign*(close-level)>=.1*atr and -sign*(close-o)>=.4*atr and point['volume_ratio']>=1.2:
                    extreme=max(p['extreme'],h) if sign>0 else min(p['extreme'],l)
                    failure=dict(p,failed_i=i,extreme=extreme);pending=None;note('range_failed')
                else:
                    touch=l<=level+.25*atr if sign>0 else h>=level-.25*atr
                    location=(close-l if sign>0 else h-close)/width if width>0 else .5
                    if (touch and sign*(close-level)>=.1*atr and sign*(close-o)>=.3*atr
                            and location>=.6 and point['volume_ratio']>=.75):
                        if macro_allowed(references,known_at,sign,inverse=symbol=='SOXSUSDT'):
                            choices.append((FAMILIES[1],sign,l if sign>0 else h))
                        else:note('retest_macro_rejected')
                        pending=None
        if len(indices)>=6 and trade['volume'][i-1]>0:
            prior=context.at(known_at-900);older=context.at(known_at-2700)
            if prior is not None and older is not None:
                pm=prior['mark_index'];po,ph,pl,pc=[float(mark[k][pm]) for k in ('open','high','low','close')]
                for sign in (1,-1):
                    touch=pl<=prior['vwap']+.25*atr if sign>0 else ph>=prior['vwap']-.25*atr
                    extension=sign*(close-point['vwap'])/atr
                    location=(close-l if sign>0 else h-close)/width if width>0 else .5
                    if (touch and sign*(pc-po)<0 and sign*(close-o)>=.3*atr
                            and sign*(close-pc)>=.1*atr and .1<=extension<=1.5
                            and sign*point['session_return_atr']>=.5
                            and sign*(point['vwap']-older['vwap'])>0 and location>=.6
                            and point['volume_ratio']>=.75):
                        if macro_allowed(references,known_at,sign,inverse=symbol=='SOXSUSDT'):
                            choices.append((FAMILIES[2],sign,min(l,pl) if sign>0 else max(h,ph)))
                        else:note('vwap_macro_rejected')
        for family,sign,extreme in sorted(choices,key=lambda x:FAMILIES.index(x[0])):
            row=bracket(point,mark,trade,symbol,spec,sign,extreme,family,or_high,or_low)
            if row is None:note('bracket_rejected');continue
            events.append(row);note('published_'+family);break
        if pending is None and failure is None:
            sign=1 if close>or_high+.1*atr else -1 if close<or_low-.1*atr else 0
            location=(close-l if sign>0 else h-close)/width if width>0 else .5
            if sign and sign*(close-o)>=.3*atr and location>=.6 and point['volume_ratio']>=1.2:
                pending=dict(sign=sign,level=or_high if sign>0 else or_low,i=i,extreme=h if sign>0 else l)
                note('range_breakout_wait')
    return events
