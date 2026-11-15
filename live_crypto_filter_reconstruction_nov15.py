"""Reconstruct frozen model inputs at actual signal times, without future bars."""
import json
import hashlib
from pathlib import Path
from bisect import bisect_right
from live_filter_loss_audit_nov15 import load_snapshot, ROOT
from src.crypto_profit_first_material_shadow_router import shadow_setups
from src.crypto_venue_router import BTC_CONTEXT_HISTORY_BARS,FAMILY_SIGNAL_HISTORY_BARS


def prefix(data,stamp,bars):
    stop=bisect_right(data['time'],stamp)
    return {k:v[max(0,stop-bars):stop] for k,v in data.items()}


def main():
    fresh=load_snapshot(ROOT/'crypto_15m/manifest.json')
    old=load_snapshot(Path('reports/audit_2026_09_08/xperp_15m_nov14_merged/manifest.json'))
    from append_candle_snapshot import merge
    data={s:merge(old[s],fresh[s])[0] for s in fresh}
    signals=json.loads((ROOT/'crypto_signals.json').read_text())['signals']
    rows=[]
    for row in reversed(signals):
        entry_time=int(row['strategy_entry_bar_ts'])
        candles=data[row['symbol']];index=candles['time'].index(entry_time)
        own=prefix(candles,entry_time-900,FAMILY_SIGNAL_HISTORY_BARS)
        btc=prefix(data['BTCUSDT'],entry_time-900,BTC_CONTEXT_HISTORY_BARS)
        diag={}
        found=shadow_setups(own,btc,entry_time=entry_time,
            market_price=row['entry_price'],model_price=candles['open'][index],
            symbol=row['symbol'],diagnostics=diag)
        setup=found[0] if found else None
        result=dict(id=row['id'],symbol=row['symbol'],direction=row['direction'],
            entry_time=entry_time,status=row['status'],r=row['realized_r'],
            reconstruction=setup,diagnostics=diag,
            direction_matches=bool(setup and setup['direction']==row['direction']))
        rows.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('reconstruction','diagnostics')}
            |dict(family=setup.get('family') if setup else None,
                quality=setup.get('model_quality') if setup else None,
                btc_regime=setup.get('btc_regime') if setup else None,
                btc_return20_atr=setup.get('btc_return20_atr') if setup else None)),flush=True)
    (ROOT/'crypto_reconstruction.json').write_text(json.dumps(dict(
        status='DIAGNOSTIC_NOT_NEW_FILTER_VALIDATION',rows=rows,
        limitations=['Reconstructed public candles are not the actual API responses at decision time.',
            'The classifier score is not a demonstrated future win rate.',
            'No client order execution or realized account return is inferred.'],
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'crypto_signals.json',ROOT/'crypto_15m/manifest.json',
            Path('reports/audit_2026_09_08/xperp_15m_nov14_merged/manifest.json'),
            Path('src/crypto_profit_first_material_shadow_router.py'),
            Path('src/frozen_crypto_profit_first_material_2r.json'),Path(__file__))}),indent=2))


if __name__=='__main__':main()
