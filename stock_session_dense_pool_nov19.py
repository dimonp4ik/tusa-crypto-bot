"""Dense pre-entry session features and trade-price outcomes, unknowns retained."""
from pathlib import Path
import hashlib
import json
import math
import pandas as pd

from stock_session_entries_nov19 import SessionContext
from stock_session_dense_features_nov19 import FeatureContext,proposals,FEATURES
from liquidity_trap_joint_nov19_audit import load_frame,with_exact_tick,attach_outcome
from phase_scenarios_nov19_audit import ROOT,STOCK_ROOT,STOCK_MANIFEST

EXPANSION=ROOT/'native_expansion_nov19';SNAPSHOT=ROOT/'native_universe_nov19_snapshot.json'
CACHE=ROOT/'stock_session_dense_nov19_pool.pkl'
META=ROOT/'stock_session_dense_nov19_pool_sources.json'


def source_hashes():
    paths=[Path(__file__),Path('stock_session_dense_features_nov19.py'),Path('stock_session_entries_nov19.py'),
        Path('stock_session_extended_nov19.py'),Path('us_cash_session_nov19.py'),
        Path('liquidity_trap_joint_nov19_audit.py'),Path('price_grid_bracket_nov19.py'),
        Path('price_bracket_path_nov19.py'),Path('src/okx_trader.py'),STOCK_MANIFEST,
        EXPANSION/'manifest.json',EXPANSION/'selection.json',SNAPSHOT]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_pool():
    sources=source_hashes()
    if CACHE.exists() or META.exists():
        if not CACHE.exists() or not META.exists():raise ValueError('incomplete_dense_session_cache')
        meta=json.loads(META.read_text(encoding='utf-8'))
        if meta['sources']!=sources or hashlib.sha256(CACHE.read_bytes()).hexdigest()!=meta['sha256']:
            raise ValueError('dense_session_cache_source_changed')
        return pd.read_pickle(CACHE),meta
    instruments=json.loads(SNAPSHOT.read_text(encoding='utf-8'))['instruments']
    datasets={};specs={};data_sources={}
    for s,item in json.loads(STOCK_MANIFEST.read_text(encoding='utf-8'))['symbols'].items():
        mark=STOCK_ROOT/item['path'];trade=STOCK_ROOT/item['trade_path']
        datasets[s]=dict(mark=load_frame(mark,item['sha256']),trade=load_frame(trade,item['trade_sha256']),origin='base')
        data_sources.update({str(mark):item['sha256'],str(trade):item['trade_sha256']})
        specs[s]=with_exact_tick(item['spec'],instruments)
    expanded=json.loads((EXPANSION/'manifest.json').read_text(encoding='utf-8'))
    if not expanded['complete']:raise ValueError('dense_session_expansion_incomplete')
    for s,item in expanded['symbols'].items():
        if item['category']!='3':continue
        if s in datasets:raise ValueError('duplicate_dense_session_asset')
        mark=Path(item['mark']['path']);trade=Path(item['trade']['path'])
        datasets[s]=dict(mark=load_frame(mark,item['mark']['sha256']),trade=load_frame(trade,item['trade']['sha256']),origin='expansion')
        data_sources.update({str(mark):item['mark']['sha256'],str(trade):item['trade']['sha256']})
        specs[s]=with_exact_tick(item['spec'],instruments)
    contexts={s:SessionContext(d['mark'],d['trade']) for s,d in datasets.items()};features=FeatureContext(contexts)
    end=min(min(d['mark']['time'][-1],d['trade']['time'][-1])+900 for d in datasets.values())
    records=[];diagnostics={}
    for s,context in sorted(contexts.items()):
        diag={};events=proposals(context,s,specs[s],features,diagnostics=diag)
        for row in events:
            if row['entry_time']+32*900>end:continue
            out=attach_outcome(dict(row,origin=datasets[s]['origin']),context.trade,True)
            if not out['known']:
                # NaNs are deliberately not flat outcomes. They permit causal
                # admission checks; any admitted unknown invalidates cash score.
                out.update(signed_return=math.nan,net_r=math.nan,sequential_outcome='UNKNOWN',
                    sequential_banked=False,near_exit_time=math.nan,bank_price_return=math.nan,runner_price_return=math.nan)
            records.append(out)
        diagnostics[s]=diag
        print('dense_session_pool',s,'proposals',len(events),flush=True)
    frame=pd.DataFrame(records)
    if frame.empty or frame.duplicated(['symbol','entry_time','direction']).any():
        raise ValueError('empty_or_duplicate_dense_session_pool')
    frame.to_pickle(CACHE)
    meta=dict(sources=sources,data_sources=data_sources,specs=specs,feature_names=FEATURES,
        sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),rows=len(frame),known=int(frame.known.sum()),
        unknown_reasons=frame.loc[~frame.known,'unknown_reason'].value_counts().to_dict() if 'unknown_reason' in frame else {},
        diagnostics=diagnostics,first_entry=float(frame.entry_time.min()),last_entry=float(frame.entry_time.max()),
        objective_verified=False,live_activation_supported=False)
    META.write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8');return frame,meta
