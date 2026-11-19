"""Research brackets rounded by the production price-grid function."""
import math
from src.okx_trader import round_to_tick


def rounded_bracket(row,spec):
    if row['direction'] not in ('LONG','SHORT'):raise ValueError('unknown_grid_direction')
    tick=float(spec.get('tickSz') or 0)
    if not math.isfinite(tick) or tick<=0:raise ValueError('missing_price_grid')
    sign=1 if row['direction']=='LONG' else -1;entry=float(row['entry'])
    prices={k:round_to_tick(float(row[k]),tick) for k in ('sl','tp1','tp2')}
    if not all(math.isfinite(v) and v>0 for v in (entry,*prices.values())):return None
    # Publish already-quantized levels. If nearest-tick rounding crossed the
    # hard minimum, move one tick outward rather than accepting a tiny target
    # or a stop inside the declared noise floor. Broker rounding is idempotent.
    if sign*(entry-prices['sl'])/entry<.003-1e-12:
        prices['sl']=round_to_tick(prices['sl']-sign*tick,tick)
    if sign*(prices['tp1']-entry)/entry<.005-1e-12:
        prices['tp1']=round_to_tick(prices['tp1']+sign*tick,tick)
    risk=sign*(entry-prices['sl'])/entry
    first=sign*(prices['tp1']-entry)/entry;final=sign*(prices['tp2']-entry)/entry
    if not (.003-1e-12<=risk<=float(row.get('risk_cap',.015))+1e-12
            and first>=.005-1e-12 and final>=.008-1e-12 and final>first
            and final/risk>=1.5-1e-12):return None
    return dict(row,**prices,risk_pct=risk,target1_fraction=first,target2_fraction=final,tick_size=tick,
        price_grid_changed=any(abs(prices[k]-row[k])>1e-12 for k in prices),
        unrounded_sl=row['sl'],unrounded_tp1=row['tp1'],unrounded_tp2=row['tp2'])
