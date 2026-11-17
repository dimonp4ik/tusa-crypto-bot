"""Causal leave-one-out native-market breadth; research only."""
from bisect import bisect_right
import math
from statistics import median

POLICIES = ('baseline', 'opposition_1h', 'opposition_4h', 'opposition_either',
            'agreement_1h', 'family_context', 'opposition_both', 'trend_opposition_both')


def peer_features(candles, own_symbol, entry_time):
    if not math.isfinite(entry_time) or entry_time % 900:
        raise ValueError('entry must be an aligned 15m boundary')
    returns = []
    for symbol, data in candles.items():
        if symbol == own_symbol:
            continue
        stop = bisect_right(data['time'], entry_time - 900)
        if stop < 17:
            continue
        begin = stop - 17
        if list(data['time'][begin:stop]) != list(range(int(entry_time)-17*900, int(entry_time), 900)):
            continue
        close = [float(x) for x in data['close'][begin:stop]]
        high = [float(x) for x in data['high'][begin:stop]]
        low = [float(x) for x in data['low'][begin:stop]]
        if (not all(math.isfinite(x) and x > 0 for x in close+high+low)
                or any(l > c or c > h for l,c,h in zip(low,close,high))):
            raise ValueError('invalid closed peer prices')
        atr = sum(max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
                  for i in range(3,17))/14
        if atr <= 0:
            continue
        returns.append(((close[-1]-close[-5])/atr, (close[-1]-close[-17])/atr))
    out = dict(peers=len(returns), known_at=entry_time)
    if len(returns) < 6:
        return out
    for horizon,index in (('1h',0),('4h',1)):
        values = [r[index] for r in returns]
        out.update({f'median_{horizon}_atr':median(values),
                    f'up_{horizon}':sum(v>0 for v in values)/len(values),
                    f'down_{horizon}':sum(v<0 for v in values)/len(values)})
    return out


def allows(policy, row, features):
    if policy not in POLICIES:
        raise ValueError('unknown breadth policy')
    if policy == 'baseline':
        return True
    if features['peers'] < 6:
        return False
    direction = row['direction']
    if direction not in ('LONG','SHORT'):
        raise ValueError('unknown direction')
    sign = 1 if direction == 'LONG' else -1
    support,opposite = ('up','down') if sign>0 else ('down','up')
    def opposed(horizon):
        return (features[f'{opposite}_{horizon}'] >= 2/3
                and sign*features[f'median_{horizon}_atr'] <= -.5)
    if policy == 'opposition_1h':
        return not opposed('1h')
    if policy == 'opposition_4h':
        return not opposed('4h')
    if policy == 'opposition_either':
        return not (opposed('1h') or opposed('4h'))
    if policy == 'opposition_both':
        return not (opposed('1h') and opposed('4h'))
    if policy == 'trend_opposition_both':
        return row['family']=='range_reversion' or not (opposed('1h') and opposed('4h'))
    agreeing = features[f'{support}_1h'] >= .5 and sign*features['median_1h_atr'] >= 0
    if policy == 'agreement_1h':
        return agreeing
    # A contrarian family may retain its reversal if the fast basket already
    # supports it. Trend/breakout families require no strong opposing basket.
    return (not opposed('4h') or agreeing) if row['family']=='range_reversion' else not (opposed('1h') or opposed('4h'))
