"""Descriptive phase attribution of model signals; never used for selection."""
import hashlib
import json
from pathlib import Path
import pickle
import pandas as pd
from src.crypto_venue_router import btc_context

ROOT=Path('reports/audit_2026_09_08')


def main():
    chosen=ROOT/'crypto_native_price_nov19_later.pkl'
    manifest=ROOT/'xperp_15m_nov14_merged/manifest.json'
    frame=pd.read_pickle(chosen);item=json.loads(manifest.read_text(encoding='utf-8'))['symbols']['BTCUSDT']
    path=Path(item['path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('btc_attribution_source_changed')
    btc=pickle.loads(raw);lookup={int(t):i for i,t in enumerate(btc['time'])};memo={}
    for stamp in frame.entry_time.unique():
        i=lookup[int(stamp)]
        if list(btc['time'][i-385:i])!=list(range(int(stamp)-385*900,int(stamp),900)):
            raise ValueError('missing_closed_attribution_context')
        context=btc_context({k:v[i-385:i] for k,v in btc.items()})
        if context is None:raise ValueError('unknown_attribution_phase')
        memo[stamp]=context['btc_regime']
    frame['long_btc_regime']=frame.entry_time.map(memo)
    frame['month']=pd.to_datetime(frame.entry_time,unit='s',utc=True).dt.strftime('%Y-%m');rows=[]
    for keys,g in frame.groupby(['month','family','long_btc_regime']):
        rows.append(dict(month=keys[0],family=keys[1],long_btc_regime=keys[2],signals=len(g),
            price_sum=float(g.runner_price_return.sum()),mean_prediction=float(g.predicted_price_return.mean()),
            mean_actual=float(g.runner_price_return.mean()),wins=int(g.runner_price_return.gt(0).sum()),
            stops=int(g.sequential_outcome.eq('SL').sum()),locks=int(g.sequential_outcome.eq('LOCK').sum())))
    report=dict(basis='model signal paths, not account cash or private fills; descriptive, not selection evidence',
        rows=rows,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (chosen,manifest,path,Path(__file__))})
    (ROOT/'crypto_native_price_nov19_diagnostics.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('phase_attribution_groups',len(rows))


if __name__=='__main__':main()
