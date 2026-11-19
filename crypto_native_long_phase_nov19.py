"""Long closed BTC/ETH phase ablation; never reads outcome fields."""
import math
import numpy as np
from src.crypto_venue_router import btc_context

PHASES=('bull','bear','bull_pullback','bear_rally')
LONG_FEATURES=tuple(f'{symbol}_{field}' for symbol in ('BTC','ETH') for field in
    ('long_gap_atr','return96_atr','eff96','atr_fraction',*(f'phase_{phase}' for phase in PHASES)))


def long_phase_features(candles,entry_time,direction):
    if direction not in ('LONG','SHORT') or not math.isfinite(entry_time) or entry_time%900:
        raise ValueError('invalid_phase_entry')
    sign=1 if direction=='LONG' else -1;features={}
    for name in ('BTC','ETH'):
        c=candles[name+'USDT'];i=int(np.searchsorted(c['time'],entry_time))
        if i<385 or list(c['time'][i-385:i])!=list(range(int(entry_time)-385*900,int(entry_time),900)):
            raise ValueError('missing_long_closed_context')
        context=btc_context({k:v[i-385:i] for k,v in c.items()})
        if context is None:raise ValueError('unusable_long_closed_context')
        features.update({f'{name}_long_gap_atr':sign*context['btc_slow_long_atr'],
            f'{name}_return96_atr':sign*context['btc_return96_atr'],
            f'{name}_eff96':context['btc_eff96'],f'{name}_atr_fraction':context['btc_atr_pct']})
        features.update({f'{name}_phase_{phase}':float(context['btc_regime']==phase) for phase in PHASES})
    if set(features)!=set(LONG_FEATURES) or not all(math.isfinite(v) for v in features.values()):
        raise ValueError('invalid_long_phase_vector')
    return [features[n] for n in LONG_FEATURES]


def attach_long_phase(frame,candles):
    result=frame.copy();memo={}
    for row in result.itertuples():
        key=(int(row.entry_time),row.direction)
        if key not in memo:memo[key]=long_phase_features(candles,*key)
    values=np.asarray([memo[(int(r.entry_time),r.direction)] for r in result.itertuples()])
    for i,name in enumerate(LONG_FEATURES):result['feature__'+name]=values[:,i]
    return result
