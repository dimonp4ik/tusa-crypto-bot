"""Verify loss paths using confirmed public minutes; never backfill entry knowledge."""
import json
import hashlib
import math
from pathlib import Path
from minute_zone_entries import fetch_minutes

ROOT=Path('reports/audit_2026_09_08/live_filter_review_nov15')


def main():
    cache=ROOT/'minute_cache';cache.mkdir(exist_ok=True)
    output=[]
    for bot in ('stock','crypto'):
        manifest=json.loads((ROOT/f'{bot}_15m/manifest.json').read_text())
        exported=json.loads((ROOT/f'{bot}_signals.json').read_text())
        selected=([r for r in exported['signals'] if r['id'] in (394,395,396)] if bot=='stock'
            else [r for r in exported['signals'] if r['closed_at'] and r['opened_at']>=1794528000])
        for row in reversed(selected):
            start=int(row['opened_at']//60)*60
            end=int(row['closed_at']//60)*60+60
            inst=manifest['symbols'][row['symbol']]['instrument']
            data,digest=fetch_minutes(inst,start,end,cache)
            # Exclude the publication minute: its intraminute pre-entry part
            # cannot be reconstructed from OHLC, and is never called a fill.
            usable={t:v for t,v in data.items() if t>=math.ceil(row['opened_at']/60)*60}
            direction=1 if row['direction']=='LONG' else -1
            entry=row['entry_price'];risk=abs(entry-row['sl'])
            favorable=lambda v:(v[1]-entry)/risk if direction==1 else (entry-v[2])/risk
            adverse=lambda v:(entry-v[2])/risk if direction==1 else (v[1]-entry)/risk
            stop=lambda v:v[2]<=row['sl'] if direction==1 else v[1]>=row['sl']
            tp=lambda v:v[1]>=row['tp1'] if direction==1 else v[2]<=row['tp1']
            stops=[t for t,v in usable.items() if stop(v)]
            targets=[t for t,v in usable.items() if tp(v)]
            first_stop=min(stops) if stops else None
            before={t:v for t,v in usable.items() if first_stop is None or t<first_stop}
            result=dict(bot=bot,id=row['id'],symbol=row['symbol'],direction=row['direction'],
                status=row['status'],r=row['realized_r'],entry=entry,stop=row['sl'],tp1=row['tp1'],
                first_stop_minute=first_stop,first_tp1_minute=min(targets) if targets else None,
                stop_minute_ohlc=usable.get(first_stop),
                mfe_before_stop_r=max([0.]+[favorable(v) for v in before.values()]),
                mae_before_stop_r=max([0.]+[adverse(v) for v in before.values()]),
                both_levels_same_minute=bool(stops and targets and min(stops)==min(targets)),
                publication_minute_touches_stop=stop(data[start]),
                publication_minute_touches_tp1=tp(data[start]),
                publication_minute_excluded=True,source_sha256=digest)
            output.append(result);print(json.dumps(result),flush=True)
    (ROOT/'minute_paths.json').write_text(json.dumps(dict(status='PATH_DIAGNOSTIC',rows=output,
        limitations=['OHLC does not show intraminute ordering or executable bid/ask.',
            'MFE excludes the stop minute; it is a conservative known pre-stop bound.',
            'Confirmed zero-volume candles can carry prior prices; not proof of a fill.'],
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'stock_signals.json',ROOT/'crypto_signals.json',Path(__file__))}),indent=2))


if __name__=='__main__':main()
