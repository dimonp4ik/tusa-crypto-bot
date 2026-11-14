"""Bounded native-X-Perp execution-timing diagnostic, not strategy selection."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from minute_zone_entries import fetch_minutes
from crypto_profit_first_runner_protection_oct30_audit import replay_runner

ROOT = Path('reports/audit_2026_09_08')


def aggregate_traded_minutes(bars, volumes):
    """Exclude empty-minute previous-close placeholders from traded OHLC."""
    traded=[bar for bar,volume in zip(bars,volumes) if volume>0]
    if not traded:
        if any(bar!=bars[0] for bar in bars) or len(set(bars[0]))!=1:
            raise ValueError('non-flat all-empty candle')
        return tuple(bars[0])
    return (traded[0][0],max(b[1] for b in traded),min(b[2] for b in traded),traded[-1][3])


def main():
    source = ROOT/'crypto_frozen_nov14_expiry_mark_v2_be_0.5.pkl'
    specs_path = ROOT/'shared_xperp_specs_nov14.json'
    old_path = ROOT/'xperp_15m_nov11_merged/manifest.json'
    cache = ROOT/'minute_be_parity_nov15_cache'
    cache.mkdir(exist_ok=True)
    specs = json.loads(specs_path.read_text())
    previous = json.loads(old_path.read_text())
    start = max(v['end'] for v in previous['symbols'].values())+1
    rows = pd.read_pickle(source)
    rows = rows[rows.entry_time.ge(start)].sort_values(['entry_time','symbol']).head(12)
    report = dict(status='BOUNDED_MINUTE_DIAGNOSTIC_NOT_ACCOUNT_PNL',objective_verified=False,
        selection='First twelve chronologically admitted entries after the Nov11 snapshot, without outcome filtering',
        assumptions=[
            'Same selected entries and planned fills; no portfolio rerouting or profit extrapolation',
            'Only confirmed native X-Perp minute bars; all 720 minutes must exist',
            'Breakeven amendment after the activation minute; same-minute ordering remains uncertain',
            'Original stop wins ambiguous stop/target bars; resting targets assumed filled at their levels',
            'Initial order latency, actual fill prices, order-book liquidity and amendment acknowledgments not reconstructed',
            'Exclude zero-volume previous-close placeholders when reaggregating traded OHLC; any remaining 15-minute mismatch rejects the row',
            'Replay retains zero-volume minute last prices as observations, without asserting new execution liquidity',
        ],results=[],sources={})
    for row in rows.to_dict('records'):
        symbol=row['symbol']; entry_time=int(row['entry_time']); end=entry_time+48*900
        item=dict(symbol=symbol,direction=row['direction'],entry_time=entry_time,
            entry=float(row['entry']),sl=float(row['sl']),
            signal_runner_r=float(row['sequential_runner_r']),signal_outcome=row['runner_outcome'])
        try:
            inst=specs[symbol]['instId']
            minutes,digest=fetch_minutes(inst,entry_time,end,cache)
            minute_path=cache/f'{inst}_{entry_time}_{end}.json'
            report['sources'][str(minute_path)]=digest
            raw=json.loads(minute_path.read_text())
            volumes={int(v[0])//1000:float(v[5]) for v in raw['data'] if v[-1]=='1'}
            item['zero_volume_minutes']=sum(volumes[t]==0 for t in minutes)
            item['first_traded_minute']=next((t for t in sorted(minutes) if volumes[t]>0),None)
            path=ROOT/'xperp_15m_nov14_merged'/f'{symbol}_15min_current.pkl'
            candle=pd.read_pickle(path)
            report['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            lookup={int(t):i for i,t in enumerate(candle['time'])}
            for stamp in range(entry_time,end,900):
                bars=[minutes[t] for t in range(stamp,stamp+900,60)]
                aggregate=aggregate_traded_minutes(bars,[volumes[t] for t in range(stamp,stamp+900,60)])
                original=tuple(float(candle[k][lookup[stamp]]) for k in ('open','high','low','close'))
                if any(abs(a-b)>max(abs(b)*1e-8,1e-9) for a,b in zip(aggregate,original)):
                    raise ValueError(f'minute/15m OHLC mismatch at {stamp}: {aggregate} vs {original}')
            data={'time':list(range(entry_time,end,60))}
            for i,k in enumerate(('open','high','low','close')):
                data[k]=[minutes[t][i] for t in data['time']]
            kwargs=dict(direction=row['direction'],entry=float(row['entry']),stop=float(row['sl']),
                activation_r=.5,lock_r=0.,bar_seconds=60,max_bars=720)
            for name,stress in (('minute_callback',False),('minute_activation_stress',True)):
                outcome,runner,banked,exit_time=replay_runner(data,0,activation_bar_stress=stress,**kwargs)
                item[name]=dict(outcome=outcome,runner_r=runner,banked=banked,exit_time=exit_time)
        except Exception as exc:
            item['error']=f'{type(exc).__name__}: {exc}'
        report['results'].append(item)
        print(json.dumps(item),flush=True)
        for p in (source,specs_path,old_path,Path(__file__),
                  Path('crypto_profit_first_runner_protection_oct30_audit.py'),Path('minute_zone_entries.py')):
            report['sources'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        (ROOT/'crypto_minute_be_parity_nov15.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
