"""Closed multi-timeframe sweep/retest hypotheses, with market entries only."""
from bisect import bisect_right
from datetime import datetime,time,timezone
from zoneinfo import ZoneInfo
import math
import numpy as np

FAMILIES=('range_sweep_confirm','trend_retest_confirm')
TARGET1=.006
TARGET2=.018
MINIMUM_FINAL_MOVE=.01
MINIMUM_STOP=.003
MAXIMUM_STOP=.008
HORIZON=32
ET=ZoneInfo('America/New_York')


def finite_ema(values,span,window=64):
    values=np.asarray(values,dtype=float);out=np.full(len(values),np.nan)
    if len(values)<window:return out
    alpha=2/(span+1)
    weights=np.r_[alpha*(1-alpha)**np.arange(window-1),(1-alpha)**(window-1)]
    out[window-1:]=np.convolve(values,weights,mode='valid');return out


class HigherContext:
    def __init__(self,candles,factor):
        seconds=factor*900;groups={}
        for i,stamp in enumerate(candles['time']):groups.setdefault(int(stamp)//seconds*seconds,[]).append(i)
        rows=[]
        for bucket,indices in sorted(groups.items()):
            if [int(candles['time'][i]) for i in indices]!=list(range(bucket,bucket+seconds,900)):continue
            rows.append((bucket+seconds,float(candles['open'][indices[0]]),
                max(float(candles['high'][i]) for i in indices),min(float(candles['low'][i]) for i in indices),
                float(candles['close'][indices[-1]])))
        self.ends=[r[0] for r in rows];self.seconds=seconds;self.memo={}
        self.o,self.h,self.l,self.c=[np.asarray([r[i] for r in rows],dtype=float) for i in range(1,5)]
        self.fast=finite_ema(self.c,20);self.slow=finite_ema(self.c,50)

    def at(self,entry_time):
        j=bisect_right(self.ends,entry_time)-1
        if j<64 or self.ends[j]-self.ends[j-64]!=64*self.seconds:return None
        if j in self.memo:return self.memo[j]
        previous=self.c[j-14:j];tr=np.maximum.reduce((self.h[j-13:j+1]-self.l[j-13:j+1],
            abs(self.h[j-13:j+1]-previous),abs(self.l[j-13:j+1]-previous)))
        atr=float(np.mean(tr));recent=self.c[j-19:j+1];path=float(np.sum(abs(np.diff(recent))))
        if min(atr,path)<=0:return None
        efficiency=abs(float(recent[-1]-recent[0]))/path
        trend=0
        if efficiency>.25:
            if self.fast[j]>self.slow[j] and self.c[j]>self.fast[j]>self.fast[j-1]:trend=1
            elif self.fast[j]<self.slow[j] and self.c[j]<self.fast[j]<self.fast[j-1]:trend=-1
        gap=float((self.fast[j]-self.slow[j])/atr)
        highs=[];lows=[]
        for k in range(j-31,j-1):
            if self.h[k]>max(self.h[k-2:k]) and self.h[k]>=max(self.h[k+1:k+3]):highs.append(float(self.h[k]))
            if self.l[k]<min(self.l[k-2:k]) and self.l[k]<=min(self.l[k+1:k+3]):lows.append(float(self.l[k]))
        result=dict(known_at=self.ends[j],trend=trend,range=efficiency<=.25 and abs(gap)<=1,
            efficiency=efficiency,gap_atr=gap,pivot_highs=highs,pivot_lows=lows)
        self.memo[j]=result;return result


def stock_clock(entry_time):
    now=datetime.fromtimestamp(entry_time,timezone.utc).astimezone(ET)
    if now.weekday()>=5 or not 600<=now.hour*60+now.minute<930:return None
    # A weekday-clock research filter, not an official holiday/early-close calendar.
    return now.replace(hour=16,minute=0,second=0,microsecond=0).timestamp()


def candidates(signal,trade,symbol,*,references=(),stocks=False,entry_stress=.0005):
    if entry_stress not in (0.,.0005):raise ValueError('unknown_entry_price_scenario')
    h4=HigherContext(signal,16);h1=HigherContext(signal,4)
    refs=[HigherContext(c,16) for c in references]
    t=np.asarray(signal['time'],dtype=np.int64)
    o,h,l,c=[np.asarray(signal[k],dtype=float) for k in ('open','high','low','close')]
    if len(t)<1040:return []
    if (np.diff(t)<=0).any():raise ValueError('unordered_signal_history')
    if (not all(np.isfinite(x).all() for x in (o,h,l,c)) or (l<=0).any()
            or (l>np.minimum(o,c)).any() or (h<np.maximum(o,c)).any()):
        raise ValueError('invalid_signal_prices')
    previous=np.r_[c[0],c[:-1]];tr=np.maximum.reduce((h-l,abs(h-previous),abs(l-previous)))
    atr=np.r_[np.full(13,np.nan),np.convolve(tr,np.ones(14)/14,mode='valid')]
    lookup={int(x):i for i,x in enumerate(trade['time'])};records=[];pending=None
    gaps=np.r_[0,np.cumsum(np.diff(t)!=900)]
    for j in range(1040,len(t)):
        stamp=int(t[j]);entry_time=stamp+900;expire=stock_clock(entry_time) if stocks else None
        if stocks and expire is None:pending=None;continue
        if gaps[j]!=gaps[j-1039]:pending=None;continue
        four=h4.at(entry_time);one=h1.at(entry_time);market=[r.at(entry_time) for r in refs]
        if four is None or one is None or any(r is None for r in market):pending=None;continue
        i=lookup.get(stamp)
        if i is None or i<20 or list(trade['time'][i-20:i+1])!=list(range(stamp-20*900,stamp+900,900)):
            pending=None;continue
        volume=np.asarray(trade['volume'][i-20:i+1],dtype=float);mean=float(volume[:-1].mean())
        if not np.isfinite(volume).all() or (volume<0).any():raise ValueError('invalid_trade_activity')
        if mean<=0 or volume[-1]<=0 or np.count_nonzero(volume[-16:]>0)<12:
            pending=None;continue
        reference=float(trade['close'][i]);a=float(atr[j])
        if min(a,reference,c[j])<=0 or not math.isfinite(a) or not math.isfinite(reference) or abs(reference/c[j]-1)>.001:
            pending=None;continue
        ratio=float(volume[-1]/mean);span=float(h[j]-l[j]);body=float(c[j]-o[j])
        emitted=None
        if pending is not None:
            p=pending;sign=p['sign'];age=j-p['opened']
            invalid=(age>4 or sign*(c[j]-p['level'])<-.25*a
                or any(r['trend']==-sign for r in market))
            if p['family']=='range_sweep_confirm':
                invalid=invalid or not four['range'] or (l[j]<p['wick'] if sign>0 else h[j]>p['wick'])
            else:invalid=invalid or four['trend']!=sign or one['trend']!=sign
            if invalid:pending=None;continue
            position=(c[j]-l[j] if sign>0 else h[j]-c[j])/span if span>0 else .5
            if p['stage']=='wait_retest':
                touched=l[j]<=p['level']+.15*a if sign>0 else h[j]>=p['level']-.15*a
                if (touched and sign*(c[j]-p['level'])>=.1*a and sign*body>0 and position>=.6):
                    p.update(stage='confirm',confirmation_close=float(c[j]),wick=float(l[j] if sign>0 else h[j]),retest=j)
                continue
            if (sign*(c[j]-p['confirmation_close'])>0 and sign*body>0 and position>=.6 and ratio>=.75):
                wick=min(p['wick'],l[j]) if sign>0 else max(p['wick'],h[j])
                emitted=dict(p,wick=float(wick));pending=None
            elif age>=2 or p.get('retest') is not None:pending=None
            if emitted is None:continue
        else:
            prior_high=float(max(h[j-20:j]));prior_low=float(min(l[j-20:j]))
            if four['range'] and one['efficiency']<=.35:
                sign=1 if l[j]<prior_low-.1*a and c[j]>prior_low+.1*a and body>0 else (
                    -1 if h[j]>prior_high+.1*a and c[j]<prior_high-.1*a and body<0 else 0)
                if sign:
                    position=(c[j]-l[j] if sign>0 else h[j]-c[j])/span if span>0 else .5
                    if ratio>=1.2 and position>=.65 and not any(r['trend']==-sign for r in market):
                        pending=dict(family='range_sweep_confirm',sign=sign,opened=j,stage='confirm',
                            level=prior_low if sign>0 else prior_high,confirmation_close=float(c[j]),
                            wick=float(l[j] if sign>0 else h[j]),opposite=prior_high if sign>0 else prior_low)
            elif four['trend'] and one['trend']==four['trend']:
                sign=four['trend'];level=prior_high if sign>0 else prior_low
                if sign*(c[j]-level)>.1*a and sign*body>=.5*a and ratio>=1.2 and not any(r['trend']==-sign for r in market):
                    pending=dict(family='trend_retest_confirm',sign=sign,opened=j,stage='wait_retest',level=level)
            continue
        sign=emitted['sign'];entry=reference*(1+sign*entry_stress)
        stop=emitted['wick']-sign*.15*a
        risk=max(MINIMUM_STOP,sign*(entry-stop)/entry)
        if risk>MAXIMUM_STOP:continue
        stop=entry*(1-sign*risk);opposite=emitted.get('opposite')
        ahead=[v for v in (one['pivot_highs'] if sign>0 else one['pivot_lows']) if sign*(v-entry)>0]
        if opposite is not None and sign*(opposite-entry)>0:ahead.append(opposite)
        target_fraction=TARGET2
        if ahead:
            level=min(ahead) if sign>0 else max(ahead)
            target_fraction=min(target_fraction,sign*(level-entry)/entry-.1*a/entry)
        if target_fraction<MINIMUM_FINAL_MOVE or target_fraction/risk<1.5:continue
        records.append(dict(symbol=symbol,entry_time=entry_time,entry_index=i+1,entry=entry,sl=stop,
            tp1=entry*(1+sign*TARGET1),tp2=entry*(1+sign*target_fraction),risk_pct=risk,
            direction='LONG' if sign>0 else 'SHORT',family=emitted['family'],
            signal_strength=ratio*abs(body)/a,setup_started=int(t[emitted['opened']])+900,
            level=emitted['level'],signal_known_at=entry_time,h4_known_at=four['known_at'],h1_known_at=one['known_at'],
            reference_known_at=[r['known_at'] for r in market],expire_at=expire,
            target1_fraction=TARGET1,target2_fraction=target_fraction,hold_bars=HORIZON))
    return records
