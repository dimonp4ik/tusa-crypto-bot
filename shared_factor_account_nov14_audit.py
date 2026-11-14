"""One lot-rounded account for crypto and stock signals with causal stock readiness."""
import bisect
import hashlib
import json
import math
from pathlib import Path
import pandas as pd
from lot_aware_tp1_audit import simulate

ROOT=Path('reports/audit_2026_09_08')
STOCK=Path('../tusa stocks/reports/audit_2026_09_08')


def stock_ready_rows(frame):
    closed=frame.sort_values(['exit_time','entry_time','symbol'])
    times=closed.exit_time.tolist();values=closed.net_r.tolist()
    kept=[]
    for row in frame.to_dict('records'):
        stop=bisect.bisect_left(times,row['entry_time'])
        if stop>=20 and sum(values[stop-20:stop])>0:
            kept.append(row)
    return pd.DataFrame(kept,columns=frame.columns)


def main():
    specs_path=ROOT/'shared_xperp_specs_nov14.json'
    specs=json.loads(specs_path.read_text())
    crypto_path=ROOT/'crypto_frozen_nov14_baseline.pkl'
    crypto=pd.read_pickle(crypto_path).assign(bot='crypto')
    cm=ROOT/'xperp_15m_nov14_merged/manifest.json'
    sm=STOCK/'stock_xperp_forward_nov14_merged/manifest.json'
    metas=[json.loads(p.read_text()) for p in (cm,sm)]
    cutoff=min(v['end'] for m in metas for v in m['symbols'].values() if 'end' in v)-64*900
    for m in metas:
        for symbol,meta in m['symbols'].items():
            if 'path' in meta and specs[symbol]['instId']!=meta['instrument']:
                raise ValueError('instrument changed; cannot reuse its contract specification')
    report=dict(status='SHARED_ACCOUNT_RESEARCH_NOT_LIVE_PNL',objective_verified=False,
        deposit=125.64,client_margin_pct_each=3,leverage=10,local_slots=dict(crypto=4,stocks=8),
        global_cap=12,simultaneous_writers=2,slot_release_delay=900,margin_cap=.9,
        stock_health='20 completed shadow outcomes strictly before entry, sum R positive; no inherited pre-snapshot history',
        limitations=['Month scenarios start flat at 125.64; they are not reconstructed client history',
            'Entry cohorts exclude inherited positions and exits at/after the next month boundary',
            'Candle-trigger fills; no spread, fees, latency reconstruction or open PnL in sizing',
            'Full entry margin stays reserved until delayed slot release; partial releases not modelled',
            'Current contract specifications assumed across the cohort',
            'Same-bar ordering is tested in two directions but real arrival times remain unknown',
            'Factor cap changes shadow signal admission and therefore its hypothetical readiness history',
            'No pristine unseen market era and no forecast of 30 percent monthly'],results={},sources={})
    paths=[crypto_path,specs_path,cm,sm,Path(__file__),Path('lot_aware_tp1_audit.py')]
    for name in ('baseline','etf_same_benchmark_cap1'):
        path=STOCK/f'stock_factor_nov14_{name}.pkl';paths.append(path)
        stock=pd.read_pickle(path)
        ready=stock_ready_rows(stock).assign(bot='stocks',sequential_banked=False,near_exit_time=math.nan)
        ready['sequential_runner_r']=ready.net_r
        frame=pd.concat([crypto,ready],ignore_index=True)
        months={}
        for month,end in (('07','2026-08-01'),('08','2026-09-01'),('09','2026-10-01'),
                          ('10','2026-11-01'),('11','2026-12-01')):
            a=pd.Timestamp(f'2026-{month}-01',tz='UTC').timestamp();b=pd.Timestamp(end,tz='UTC').timestamp()
            selected=frame[frame.entry_time.ge(a)&frame.exit_time.lt(b)&frame.entry_time.le(cutoff)]
            outcomes={}
            for priority in ('crypto_first','stocks_first'):
                favored='crypto' if priority=='crypto_first' else 'stocks'
                result=simulate(selected.to_dict('records'),specs,slots=12,
                    bot_slots=dict(crypto=4,stocks=8),simultaneous_writers=2,release_delay=900,
                    tie_key=lambda r,f=favored:(r['bot']!=f,r['symbol']))
                assert result['peak_positions']<=12 and result['peak_reserved_margin_pct']<=90+1e-9
                assert result['peak_by_bot'].get('crypto',0)<=4 and result['peak_by_bot'].get('stocks',0)<=8
                assert abs(sum(result['return_contribution_pct'].values())-result['return_pct'])<1e-5
                result.pop('ledger')
                outcomes[priority]=result
            months[f'2026-{month}']=outcomes
        report['results'][name]=dict(shadow_stock_signals=len(stock),stock_ready_signals=len(ready),months=months)
    report['sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (ROOT/'shared_factor_account_nov14.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['results'],indent=2))


if __name__=='__main__':
    main()
