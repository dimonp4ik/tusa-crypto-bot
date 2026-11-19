"""Explicit causal native-candle features, independent of the deployed classifier."""
import math
from datetime import datetime, timezone
import numpy as np

FAMILIES=('range_reversion','trend_pullback','breakout')
HORIZONS=(1,2,4,16,64)
FEATURES=('atr_fraction','eff20','eff96','dir_z20','mean20_minus96_atr',
    'body_atr','close_location','favorable_wick_atr','adverse_wick_atr',
    'ahead16_atr','behind16_atr','ahead96_atr','behind96_atr',
    'range4_atr','range16_atr','volume_ratio16','zero_volume_share16',
    'risk_fraction','long','hour_sin','hour_cos','weekday_sin','weekday_cos',
    *(f'own_return_{n}' for n in HORIZONS),
    *(f'{s}_return_{n}' for s in ('BTC','ETH') for n in (4,16,64)),
    *(f'{s}_{name}' for s in ('BTC','ETH') for name in ('trend_atr','eff20','z20')),
    'peer_count','peer_median4_atr','peer_median16_atr','peer_support4','peer_support16',
    'relative_to_btc16','relative_to_eth16',*(f'family_{f}' for f in FAMILIES))


class NativeContext:
    def __init__(self, datasets):
        self.data={};self.memo={}
        for symbol,candle in datasets.items():
            c={k:np.asarray(v,dtype=np.int64 if k=='time' else float)
                for k,v in candle.items() if k in ('time','open','high','low','close','volume')}
            if set(c)!={'time','open','high','low','close','volume'}:raise ValueError('missing_candle_fields')
            if len({len(v) for v in c.values()})!=1 or (np.diff(c['time'])<=0).any():
                raise ValueError('invalid_candle_alignment')
            if any(not np.isfinite(v).all() for v in c.values()) or (c['volume']<0).any():
                raise ValueError('invalid_candle_values')
            if (np.minimum(c['open'],c['close'])<c['low']).any() or (
                    np.maximum(c['open'],c['close'])>c['high']).any() or (c['low']<=0).any():
                raise ValueError('invalid_ohlc')
            previous=np.r_[c['close'][0],c['close'][:-1]]
            tr=np.maximum.reduce((c['high']-c['low'],abs(c['high']-previous),abs(c['low']-previous)))
            atr=np.r_[np.full(13,np.nan),np.convolve(tr,np.ones(14)/14,mode='valid')]
            self.data[symbol]=dict(c=c,lookup={int(t):i for i,t in enumerate(c['time'])},atr=atr)

    def point(self,symbol,stamp,minimum=100):
        item=self.data[symbol];c=item['c'];j=item['lookup'].get(stamp)
        if j is None or j<minimum-1 or not np.array_equal(c['time'][j-minimum+1:j+1],
                np.arange(stamp-(minimum-1)*900,stamp+900,900)):
            return None
        key=(symbol,stamp)
        if key not in self.memo:
            atr=float(item['atr'][j]);recent=c['close'][j-19:j+1];slow=c['close'][j-95:j+1]
            std=float(np.std(recent));move20=float(np.sum(abs(np.diff(recent))))
            move96=float(np.sum(abs(np.diff(slow))))
            if min(atr,std,move20,move96)<=0:return None
            self.memo[key]=dict(j=j,atr=atr,eff20=abs(float(recent[-1]-recent[0]))/move20,
                eff96=abs(float(slow[-1]-slow[0]))/move96,z20=float((recent[-1]-recent.mean())/std),
                trend=float((recent.mean()-slow.mean())/atr),
                returns={n:float(c['close'][j]/c['close'][j-n]-1) for n in HORIZONS},
                return4_atr=float((c['close'][j]-c['close'][j-4])/atr),
                return16_atr=float((c['close'][j]-c['close'][j-16])/atr))
        return self.memo[key]

    def row(self,row):
        symbol=row['symbol'];stamp=int(row['entry_time'])-900
        if row['direction'] not in ('LONG','SHORT') or row['family'] not in FAMILIES:
            raise ValueError('unknown_entry_category')
        sign=1 if row['direction']=='LONG' else -1
        point=self.point(symbol,stamp);btc=self.point('BTCUSDT',stamp);eth=self.point('ETHUSDT',stamp)
        if any(p is None for p in (point,btc,eth)):raise ValueError('missing_closed_feature_context')
        c=self.data[symbol]['c'];j=point['j'];atr=point['atr']
        o,h,l,close=[float(c[k][j]) for k in ('open','high','low','close')]
        volume=c['volume'][j-15:j+1];mean=float(volume.mean())
        if mean<=0:raise ValueError('missing_trade_activity')
        peers=[self.point(s,stamp) for s in self.data if s!=symbol]
        peers=[p for p in peers if p is not None]
        if len(peers)<4:raise ValueError('insufficient_available_peers')
        now=datetime.fromtimestamp(row['entry_time'],timezone.utc);minute=now.hour*60+now.minute
        f=dict(atr_fraction=atr/close,eff20=point['eff20'],eff96=point['eff96'],
            dir_z20=sign*point['z20'],mean20_minus96_atr=sign*point['trend'],
            body_atr=sign*(close-o)/atr,close_location=(close-l if sign>0 else h-close)/(h-l) if h>l else .5,
            favorable_wick_atr=(h-max(o,close) if sign>0 else min(o,close)-l)/atr,
            adverse_wick_atr=(min(o,close)-l if sign>0 else h-max(o,close))/atr,
            range4_atr=float((max(c['high'][j-3:j+1])-min(c['low'][j-3:j+1]))/atr),
            range16_atr=float((max(c['high'][j-15:j+1])-min(c['low'][j-15:j+1]))/atr),
            volume_ratio16=float(volume[-1]/mean),zero_volume_share16=float(np.mean(volume==0)),
            risk_fraction=float(row['risk_pct']),long=float(sign>0),
            hour_sin=math.sin(2*math.pi*minute/1440),hour_cos=math.cos(2*math.pi*minute/1440),
            weekday_sin=math.sin(2*math.pi*now.weekday()/7),weekday_cos=math.cos(2*math.pi*now.weekday()/7))
        for n in (16,96):
            high=float(max(c['high'][j-n:j]));low=float(min(c['low'][j-n:j]))
            f[f'ahead{n}_atr']=(high-close if sign>0 else close-low)/atr
            f[f'behind{n}_atr']=(close-low if sign>0 else high-close)/atr
        f.update({f'own_return_{n}':sign*point['returns'][n] for n in HORIZONS})
        for name,p in (('BTC',btc),('ETH',eth)):
            f.update({f'{name}_return_{n}':sign*p['returns'][n] for n in (4,16,64)})
            f.update({f'{name}_trend_atr':sign*p['trend'],f'{name}_eff20':p['eff20'],f'{name}_z20':sign*p['z20']})
        f.update(peer_count=float(len(peers)),peer_median4_atr=sign*float(np.median([p['return4_atr'] for p in peers])),
            peer_median16_atr=sign*float(np.median([p['return16_atr'] for p in peers])),
            peer_support4=float(np.mean([sign*p['returns'][4]>0 for p in peers])),
            peer_support16=float(np.mean([sign*p['returns'][16]>0 for p in peers])),
            relative_to_btc16=sign*(point['returns'][16]-btc['returns'][16]),
            relative_to_eth16=sign*(point['returns'][16]-eth['returns'][16]))
        f.update({f'family_{family}':float(row['family']==family) for family in FAMILIES})
        if set(f)!=set(FEATURES) or not all(math.isfinite(v) for v in f.values()):raise ValueError('invalid_feature_vector')
        return [float(f[name]) for name in FEATURES]


def mature_before(frame,cutoff):
    return (frame.entry_time+48*900<cutoff)&frame.exit_time.lt(cutoff)
