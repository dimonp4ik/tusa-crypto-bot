"""Dense closed-session proposals and explicit directional features; no labels."""
from bisect import bisect_right
import math
import numpy as np
from stock_session_entries_nov19 import SessionContext
from stock_session_extended_nov19 import MarkSessionMacro
from price_grid_bracket_nov19 import rounded_bracket

FEATURES=('atr_fraction','long','inverse','leveraged_etf','risk_fraction','target1_fraction','target2_fraction',
    'body_atr','close_location','favorable_wick_atr','adverse_wick_atr','basis_now',
    'session_return_atr','distance_vwap_atr','vwap_slope3_atr','session_range_atr','session_position',
    'opening_range_atr','opening_ahead_atr','opening_behind_atr','prior_ahead_atr','prior_behind_atr',
    'gap_from_prior_mark_atr','regular_eff20','regular_z20','volume_ratio','active_session_share',
    'minutes_from_open','minutes_to_close','weekday_sin','weekday_cos',
    *(f'own_regular_return_{n}' for n in (1,2,4,16)),
    *(f'{name}_{feature}' for name in ('QQQ','SPY')
        for feature in ('session_return_atr','distance_mean_atr','atr_fraction','regular_return4','regular_return16')),
    'peer_count','peer_median_session_atr','peer_support','own_minus_qqq_session_atr','qqq_minus_spy_session_atr')


class FeatureContext:
    def __init__(self,contexts):
        self.contexts=contexts;self.macro={s:MarkSessionMacro(c) for s,c in contexts.items()}

    def regular(self,symbol,known_at,minimum=21):
        c=self.contexts[symbol];i=c.lookup.get(int(known_at)-900)
        if i is None:return None
        h=c.regular[:bisect_right(c.regular,i)]
        if len(h)<minimum:return None
        m=[c.mark_lookup.get(c.trade['time'][k]) for k in h]
        if any(j is None for j in m[-minimum:]):return None
        return c,h,m

    def row(self,row,point):
        symbol=row['symbol'];known_at=row['entry_time'];sign=1 if row['direction']=='LONG' else -1
        inverse=symbol=='SOXSUSDT';exposure=sign*(-1 if inverse else 1)
        data=self.regular(symbol,known_at)
        if data is None:raise ValueError('missing_closed_regular_features')
        c,history,mids=data;mark=c.mark;trade=c.trade;i=point['index'];m=point['mark_index'];atr=point['atr']
        closed=[float(mark['close'][j]) for j in mids[-21:]];close=float(mark['close'][m])
        o,h,l=[float(mark[k][m]) for k in ('open','high','low')];width=h-l
        day=[c.mark_lookup[trade['time'][k]] for k in point['indices']]
        high=max(float(mark['high'][j]) for j in day);low=min(float(mark['low'][j]) for j in day)
        initial=day[:2];or_high=max(float(mark['high'][j]) for j in initial);or_low=min(float(mark['low'][j]) for j in initial)
        prior=point['previous']
        if prior is None or prior['known_at']>known_at:raise ValueError('missing_known_prior_session')
        first=point['indices'][0];prior_indices=[k for k in history if k<first]
        if not prior_indices:raise ValueError('missing_prior_session_close')
        previous_mark=c.mark_lookup[trade['time'][prior_indices[-1]]]
        older=c.at(known_at-2700)
        if older is None or older['start']!=point['start']:raise ValueError('missing_prior_closed_session_vwap')
        path=float(np.sum(abs(np.diff(closed[-20:]))));std=float(np.std(closed[-20:]))
        from datetime import datetime
        from us_cash_session_nov19 import ET
        weekday=datetime.fromtimestamp(known_at,ET).weekday()
        f=dict(atr_fraction=atr/close,long=float(sign>0),inverse=float(inverse),
            leveraged_etf=float(symbol in ('SOXSUSDT','SOXLUSDT')),risk_fraction=row['risk_pct'],
            target1_fraction=row['target1_fraction'],target2_fraction=row['target2_fraction'],
            body_atr=sign*(close-o)/atr,close_location=(close-l if sign>0 else h-close)/width if width>0 else .5,
            favorable_wick_atr=(h-max(o,close) if sign>0 else min(o,close)-l)/atr,
            adverse_wick_atr=(min(o,close)-l if sign>0 else h-max(o,close))/atr,
            basis_now=sign*(point['close']/close-1),session_return_atr=sign*point['session_return_atr'],
            distance_vwap_atr=sign*(close-point['vwap'])/atr,vwap_slope3_atr=sign*(point['vwap']-older['vwap'])/atr,
            session_range_atr=(high-low)/atr,session_position=(close-low if sign>0 else high-close)/(high-low) if high>low else .5,
            opening_range_atr=(or_high-or_low)/atr,
            opening_ahead_atr=(or_high-close if sign>0 else close-or_low)/atr,
            opening_behind_atr=(close-or_low if sign>0 else or_high-close)/atr,
            prior_ahead_atr=(prior['high']-close if sign>0 else close-prior['low'])/atr,
            prior_behind_atr=(close-prior['low'] if sign>0 else prior['high']-close)/atr,
            gap_from_prior_mark_atr=sign*(point['opening']-mark['close'][previous_mark])/atr,
            regular_eff20=abs(closed[-1]-closed[-20])/path if path>0 else 0.,
            regular_z20=sign*(close-float(np.mean(closed[-20:])))/std if std>0 else 0.,
            volume_ratio=float(point['volume_ratio']),
            active_session_share=sum(trade['volume'][k]>0 for k in point['indices'])/len(point['indices']),
            minutes_from_open=(known_at-point['start'])/60,minutes_to_close=(point['end']-known_at)/60,
            weekday_sin=math.sin(2*math.pi*weekday/7),weekday_cos=math.cos(2*math.pi*weekday/7))
        for n in (1,2,4,16):f[f'own_regular_return_{n}']=sign*(close/closed[-n-1]-1)
        refs={}
        for name in ('QQQ','SPY'):
            s=name+'USDT';p=self.macro[s].at(known_at);regular=self.regular(s,known_at)
            if p is None or regular is None:raise ValueError('missing_closed_market_reference')
            cc,_,mm=regular;prices=[float(cc.mark['close'][j]) for j in mm[-17:]]
            refs[name]=p
            f.update({f'{name}_session_return_atr':exposure*p['session_return_atr'],
                f'{name}_distance_mean_atr':exposure*(p['close']-p['vwap'])/p['atr'],
                f'{name}_atr_fraction':p['atr']/p['close'],
                f'{name}_regular_return4':exposure*(prices[-1]/prices[-5]-1),
                f'{name}_regular_return16':exposure*(prices[-1]/prices[-17]-1)})
        peers=[]
        for s,macro in self.macro.items():
            if s==symbol:continue
            p=macro.at(known_at)
            if p is not None:peers.append(p['session_return_atr']*(-1 if s=='SOXSUSDT' else 1))
        if len(peers)<4:raise ValueError('insufficient_closed_session_peers')
        f.update(peer_count=float(len(peers)),peer_median_session_atr=exposure*float(np.median(peers)),
            peer_support=float(np.mean([exposure*p>0 for p in peers])),
            own_minus_qqq_session_atr=sign*point['session_return_atr']-exposure*refs['QQQ']['session_return_atr'],
            qqq_minus_spy_session_atr=exposure*(refs['QQQ']['session_return_atr']-refs['SPY']['session_return_atr']))
        if set(f)!=set(FEATURES) or not all(math.isfinite(v) for v in f.values()):
            raise ValueError('invalid_dense_session_feature_vector')
        return [float(f[k]) for k in FEATURES]


def proposals(context,symbol,spec,features,*,diagnostics=None):
    diag=diagnostics if diagnostics is not None else {};diag.clear();records=[]
    def note(key):diag[key]=diag.get(key,0)+1
    mark=context.mark;trade=context.trade
    for i,t in enumerate(trade['time']):
        known_at=int(t)+900;p=context.at(known_at)
        if p is None:continue
        if not p['start']+1800<=known_at<min(p['start']+5*3600,p['end']-1800):continue
        if len(p['indices'])<4 or any(trade['volume'][k]<=0 for k in p['indices'][:2]):continue
        mids=[context.mark_lookup.get(int(t)-900*k) for k in (2,1,0)]
        if any(m is None for m in mids):note('missing_stop_structure');continue
        for sign in (1,-1):
            entry=p['close']*(1+sign*.0005)
            extreme=min(mark['low'][m] for m in mids) if sign>0 else max(mark['high'][m] for m in mids)
            stop=extreme-sign*.15*p['atr'];risk=max(.003,sign*(entry-stop)/entry)
            if risk>.01:note('structure_risk_rejected');continue
            raw=dict(symbol=symbol,direction='LONG' if sign>0 else 'SHORT',family='session_dense',
                entry_time=known_at,entry_index=i+1,entry=entry,sl=entry*(1-sign*risk),
                tp1=entry*(1+sign*.005),tp2=entry*(1+sign*.015),risk_pct=risk,risk_cap=.01,
                expire_at=p['end'],signal_strength=abs(mark['close'][p['mark_index']]-mark['open'][p['mark_index']])/p['atr'],
                signal_known_at=known_at,session_open=p['start'],session_close=p['end'])
            row=rounded_bracket(raw,spec)
            if row is None:note('grid_bracket_rejected');continue
            try:values=features.row(row,p)
            except ValueError as exc:
                if str(exc) not in ('missing_closed_regular_features','missing_known_prior_session','missing_prior_session_close',
                    'missing_prior_closed_session_vwap','missing_closed_market_reference','insufficient_closed_session_peers'):raise
                note(str(exc));continue
            records.append(dict(row,**{'feature__'+k:v for k,v in zip(FEATURES,values)}))
    return records
