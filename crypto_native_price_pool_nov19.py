"""All family candidates, checked against closed production family generation."""
import hashlib
import json
import math
from pathlib import Path
import pickle
import pandas as pd

from crypto_native_price_features_nov19 import NativeContext,FEATURES
from crypto_runner_exit_path_nov16 import replay
from src.crypto_venue_router import family_signals

ROOT=Path('reports/audit_2026_09_08')
MANIFEST=ROOT/'xperp_15m_nov14_merged/manifest.json'
RAW=ROOT/'crypto_xperp_raw_router_audit_2_0_nov14_merged_raw_families.csv'
BASELINE=ROOT/'crypto_peer_breadth_nov17.json'
CACHE=ROOT/'crypto_native_price_nov19_pool.pkl'
META=ROOT/'crypto_native_price_nov19_pool_sources.json'


def load_pool():
    source_paths=(MANIFEST,RAW,BASELINE,Path(__file__),Path('crypto_native_price_features_nov19.py'),
        Path('crypto_runner_exit_path_nov16.py'),Path('src/crypto_venue_router.py'))
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    if CACHE.exists() or META.exists():
        if not CACHE.exists() or not META.exists():raise ValueError('incomplete_native_price_cache')
        meta=json.loads(META.read_text(encoding='utf-8'))
        if meta['sources']!=sources or hashlib.sha256(CACHE.read_bytes()).hexdigest()!=meta['sha256']:
            raise ValueError('native_price_cache_source_changed')
        return pd.read_pickle(CACHE),meta
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));candles={};data_hashes={}
    for symbol,item in manifest['symbols'].items():
        if item.get('status')!='OK':raise ValueError('incomplete_native_source')
        p=Path(item['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=item['sha256']:raise ValueError('native_candle_source_changed')
        candles[symbol]=pickle.loads(raw);data_hashes[str(p)]=digest
    context=NativeContext(candles);records=[];excluded={};baseline=json.loads(BASELINE.read_text(encoding='utf-8'))
    expected={(r['symbol'],r['entry_time'],r['direction'],r['family']):r for r in baseline['candidate_diagnostics']}
    verified=0;frame=pd.read_csv(RAW)
    if frame.duplicated(['symbol','entry_time','direction','family']).any():raise ValueError('duplicate_family_candidate')
    for symbol,group in frame.groupby('symbol',sort=True):
        c=candles[symbol];lookup=context.data[symbol]['lookup'];before=len(records)
        for row in group.to_dict('records'):
            stamp=int(row['entry_time']);i=lookup[stamp]
            if i<100 or i+48>len(c['time']):raise ValueError('incomplete_family_history')
            prefix={k:values[i-100:i] for k,values in c.items()}
            if prefix['time']!=list(range(stamp-100*900,stamp,900)):raise ValueError('family_history_gap')
            matching=[s for s in family_signals(prefix) if s['family']==row['family'] and s['direction']==row['direction']]
            if len(matching)!=1:raise ValueError('raw_candidate_not_reproduced_from_closed_runtime_prefix')
            signal=matching[0]
            if abs(float(c['open'][i])-row['entry'])>max(1e-8,abs(row['entry'])*1e-9):
                raise ValueError('raw_execution_bar_reference_mismatch')
            risk=2*signal['atr'];entry=float(c['close'][i-1]);risk_pct=risk/entry
            if not .003<=risk_pct<=.02:
                excluded['risk_fraction']=excluded.get('risk_fraction',0)+1;continue
            sign=1 if row['direction']=='LONG' else -1;stop=entry-sign*risk
            key=(symbol,stamp,row['direction'],row['family']);old=expected.get(key)
            out=dict(symbol=symbol,entry_time=stamp,direction=row['direction'],family=row['family'],
                entry=entry,sl=stop,tp1=entry+sign*risk*.5,risk_pct=risk_pct)
            try:features=context.row(out)
            except ValueError as exc:
                reason=str(exc)
                if old is not None or reason not in ('insufficient_available_peers','missing_trade_activity',
                        'missing_closed_feature_context'):raise
                excluded[reason]=excluded.get(reason,0)+1;continue
            path=replay(c,i,direction=row['direction'],entry=entry,stop=stop)
            net=.025+.95*path['runner_r'] if path['banked'] else path['runner_r']
            if old is not None:
                if abs(net-old['model_r'])>1e-9 or path['exit_time']!=old['exit_time']:
                    raise ValueError('frozen_candidate_path_mismatch')
                verified+=1
            out.update(exit_time=path['exit_time'],net_r=net,
                sequential_outcome=path['outcome'],sequential_runner_r=path['runner_r'],sequential_banked=path['banked'],
                near_exit_time=path['near_exit_time'] if path['near_exit_time'] is not None else math.nan,
                runner_price_return=risk_pct*path['runner_r'],is_frozen_candidate=old is not None)
            out.update({'feature__'+name:value for name,value in zip(FEATURES,features)})
            records.append(out)
        print('native_price_pool',symbol,len(records)-before,flush=True)
    if verified!=len(expected):raise ValueError('frozen_baseline_candidate_missing')
    pool=pd.DataFrame(records);pool.to_pickle(CACHE)
    meta=dict(sources=sources,data_hashes=data_hashes,sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        rows=len(pool),raw_rows=len(frame),excluded=excluded,feature_names=FEATURES,frozen_candidate_path_parity=verified,
        label='runner price return; no ideal partial bank assumed in fitting; cash rounds client TP1 separately')
    META.write_text(json.dumps(meta,indent=2),encoding='utf-8');return pool,meta


if __name__=='__main__':
    frame,meta=load_pool();print('native_price_pool_complete',len(frame),meta['frozen_candidate_path_parity'],flush=True)
