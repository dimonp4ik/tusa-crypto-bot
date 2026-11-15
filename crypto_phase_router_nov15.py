"""Freeze a phase-conditioned entry router before later-year/shared-account checks."""
import argparse
import hashlib
import itertools
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import verify_crypto_profit_first_runtime_parity_nov3 as runtime
from append_candle_snapshot import merge
from asset_entry_features_research import FIELDS, attach
from crypto_complete_years_nov11_audit import CORE, ENDS
from crypto_expected_cash_edge_nov14 import metrics, timestamp
from crypto_walk_forward_family_model import FAMILIES, enrich, attach_btc_context
from entry_hypotheses import replay
from lot_aware_tp1_audit import simulate
from shared_factor_account_nov14_audit import stock_ready_rows

ROOT=Path('reports/audit_2026_09_08')
STOCK=Path('../tusa stocks/reports/audit_2026_09_08')
PHASES=('range','trend','shock')
POLICIES=[''.join(p) for p in itertools.product('ba',repeat=3)]+['intersection','union']


def phases(frame):
    """Fixed thresholds on features built solely from bars preceding entry."""
    required=['asset_eff96','asset_trend_atr','asset_volume_ratio','asset_body_atr']
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError('phase features unavailable')
    trend=(frame.asset_eff96>=.30)&(frame.asset_trend_atr.abs()>=1.)
    shock=(frame.asset_volume_ratio>=2.)&(frame.asset_body_atr.abs()>=1.)
    return np.where(shock,'shock',np.where(trend,'trend','range'))


def routed(frame,base_score,asset_score,cutoffs,policy,frozen_score=None):
    b=np.asarray(base_score)/cutoffs['b']-1
    a=np.asarray(asset_score)/cutoffs['a']-1
    values=dict(b=b,a=a)
    if frozen_score is not None:
        values['c']=np.asarray(frozen_score)/cutoffs['c']-1
    combined={'intersection':('ba',True),'union':('ba',False),
        'bc_intersection':('bc',True),'bc_union':('bc',False),
        'ac_intersection':('ac',True),'ac_union':('ac',False),
        'all_intersection':('bac',True),'all_union':('bac',False)}
    if policy in combined:
        keys,require_all=combined[policy]
        scores=np.column_stack([values[k] for k in keys])
        score=(np.min if require_all else np.max)(scores,axis=1)
        allowed=score>=0
    else:
        if len(policy)!=3 or set(policy)-set(values):
            raise ValueError('unknown router')
        codes=phases(frame)
        score=np.zeros(len(frame))
        allowed=np.zeros(len(frame),dtype=bool)
        for phase,expert in zip(PHASES,policy):
            mask=codes==phase
            allowed[mask]=values[expert][mask]>=0
            # Preserve the active classifier's first-passing family priority.
            rank=(-frame.family.map(runtime.FAMILY_ORDER).to_numpy(float)
                if expert=='c' else values[expert])
            score[mask]=rank[mask]
    accepted=frame.assign(router_score=score,router_allowed=allowed)
    accepted=accepted[accepted.router_allowed].sort_values(
        ['symbol','entry_time','router_score','family'],ascending=[True,True,False,True],kind='stable')
    accepted=accepted.drop_duplicates(['symbol','entry_time'])
    return runtime.runtime_gate(runtime.base.standardized(accepted,'phase_router_research'))


def choose(development):
    eligible=[v for v in development if min(v['selection']['n'],v['confirmation']['n'])>=100]
    return max(eligible,key=lambda v:(
        min(v['selection']['worst_observed_month_pct'],v['confirmation']['worst_observed_month_pct']),
        min(v['selection']['return_pct'],v['confirmation']['return_pct']),
        v['policy'])) if eligible else None


def fingerprint(paths):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def full_year(year):
    """Rebuild all three raw families, never a previous model's filtered pool."""
    folder=ROOT/f'phase_router_full_{year}_expiry_mark_v2'
    folder.mkdir(exist_ok=True)
    pool=folder/'pool.pkl'; meta=folder/'manifest.json'
    source_paths=[Path(p) for p in (
        'append_candle_snapshot.py','entry_hypotheses.py','crypto_walk_forward_family_model.py',
        'asset_entry_features_research.py','crypto_profit_first_runner_protection_oct30_audit.py')]
    source_paths.append(runtime.router.ARTIFACT_PATH)
    source_paths.extend(Path('backtest_cache')/f'{s}_15min_18000_end{anchor}.pkl'
        for s in sorted(CORE) for anchor in ENDS[year])
    sources=fingerprint(source_paths)
    if pool.exists() and meta.exists():
        saved=json.loads(meta.read_text())
        if saved['sources']==sources and saved['pool_sha256']==hashlib.sha256(pool.read_bytes()).hexdigest():
            return pd.read_pickle(pool),sources|{str(pool):saved['pool_sha256']}
    candles={};paths={}
    for symbol in sorted(CORE):
        chunks=[pickle.loads((Path('backtest_cache')/f'{symbol}_15min_18000_end{a}.pkl').read_bytes())
            for a in ENDS[year]]
        c,_=merge(chunks[0],chunks[1]);c,_=merge(c,chunks[2])
        if any(b-a!=900 for a,b in zip(c['time'],c['time'][1:])):
            raise ValueError('full-year candle gap')
        candles[symbol]=c
        path=folder/f'{symbol}.pkl';path.write_bytes(pickle.dumps(c))
        paths[(str(year),symbol)]=path
    parts=[]
    for family in FAMILIES:
        records=[]
        for symbol,c in candles.items():
            records.extend(replay(c,symbol,family,2.,max_cost_fraction=None,roundtrip_cost=0.))
        parts.append(attach_btc_context(enrich(pd.DataFrame(records),family),candles['BTCUSDT']))
    raw=pd.concat(parts,ignore_index=True).dropna(subset=['btc_regime'])
    model=runtime.router.MODEL
    raw=raw[raw.entry_time.ge(timestamp(f'{year}-01-01'))
        &raw.entry_time.lt(timestamp(f'{year+1}-01-01'))
        &raw.risk_pct.between(model['risk_pct_min'],model['risk_pct_max'])].assign(period=str(year))
    raw=attach(raw,paths)
    raw=runtime.sequential.apply_fraction(runtime.sequential.attach_current(raw,paths),.5)
    raw.to_pickle(pool)
    saved=dict(sources=sources,pool_sha256=hashlib.sha256(pool.read_bytes()).hexdigest())
    meta.write_text(json.dumps(saved,indent=2),encoding='utf-8')
    return raw,sources|fingerprint([pool,*paths.values()])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--with-frozen-classifier',action='store_true')
    args=parser.parse_args()
    policies=POLICIES
    if args.with_frozen_classifier:
        policies=[''.join(p) for p in itertools.product('bac',repeat=3)]+[
            'intersection','union','bc_intersection','bc_union',
            'ac_intersection','ac_union','all_intersection','all_union']
    sources={}
    pool=ROOT/'expected_cash_edge_literal_pool_expiry_mark_v2_nov14.pkl'
    frame=pd.read_pickle(pool).drop_duplicates(['symbol','entry_time','direction','family'])
    frame=runtime.sequential.apply_fraction(frame,.5)
    asset=attach(frame,runtime.sequential.exact.candle_paths(tuple(frame.period.unique())))
    sources.update(asset.attrs['asset_sources'])
    experts={};cutoffs={};fits={}
    for key,values,flag in (('b',frame,False),('a',asset,True)):
        path=ROOT/f"crypto_expected_cash_edge_{'asset_features_' if flag else ''}expiry_mark_v2_nov14.json"
        reference=json.loads(path.read_text());best=reference['selected']
        train=values[values.entry_time.ge(timestamp('2020-01-01'))
            &values.exit_time.lt(timestamp('2022-01-01'))]
        schema=runtime.base.design(train)
        def matrix(v,s=schema,extra=flag):
            x=runtime.base.matrix(v,s)
            return np.column_stack((x,v[list(FIELDS)].to_numpy(np.float32))) if extra else x
        estimator=HistGradientBoostingRegressor(max_leaf_nodes=best['leaves'],min_samples_leaf=300,
            max_iter=150,learning_rate=.05,l2_regularization=10,early_stopping=False,random_state=20261114)
        estimator.fit(matrix(train),(train.net_r*train.risk_pct).to_numpy(float))
        calibration=values[values.entry_time.ge(timestamp('2022-01-01'))
            &values.exit_time.lt(timestamp('2023-01-01'))]
        threshold=float(np.quantile(estimator.predict(matrix(calibration)),best['quantile']))
        if abs(threshold-best['cutoff'])>1e-12:
            raise ValueError('refit does not reproduce frozen expert cutoff')
        if threshold<=0:
            raise ValueError('normalization requires positive frozen cutoffs')
        experts[key]=(estimator,matrix);cutoffs[key]=threshold
        fits[key]=dict(training_rows=len(train),leaves=best['leaves'],cutoff=threshold,
            training_last_exit=float(train.exit_time.max()))
        sources.update(fingerprint([path]))

    def score(values):
        result={key:estimator.predict(matrix(values)) for key,(estimator,matrix) in experts.items()}
        if args.with_frozen_classifier:
            result['c']=runtime.portable_probability_batch(runtime.feature_matrix(values))
        return result

    if args.with_frozen_classifier:
        cutoffs['c']=float(runtime.router.MODEL['quality_threshold'])
        fits['c']=dict(kind='active_frozen_far_target_classifier',cutoff=cutoffs['c'],
            training='original 2020-2021 fit, original development threshold; no refit')

    def route(values,scores,policy):
        return routed(values,scores['b'],scores['a'],cutoffs,policy,scores.get('c'))

    historical=score(asset); development=[]; rows_by_policy={}
    for policy in policies:
        rows=route(asset,historical,policy)
        rows_by_policy[policy]=rows
        development.append(dict(policy=policy,
            selection=metrics(rows,timestamp('2022-01-01'),timestamp('2023-01-01')),
            confirmation=metrics(rows,timestamp('2023-01-01'),timestamp('2024-01-01'))))
    winner=choose(development)
    report=dict(status='RESEARCH_ONLY_NO_LIVE_ROUTER_CHANGE',objective_verified=False,
        label_revision='expiry_mark_v2',experts=fits,phase_order=PHASES,
        fixed_phase_rules='shock: volume ratio >=2 and abs(body/ATR)>=1; else trend: efficiency96>=.30 and abs(SMA20-SMA96)/ATR>=1; else range',
        development=development,selected=winner,results={},sources=sources,
        limitations=['All historical eras previously researched; no pristine unseen test',
            'Entry targets use ideal 50 percent first bank; current lot rounding only in direct joint-account tests',
            'Full-year candles outside native venue are transfer stress tests on five original assets',
            'Signal BE activates next 15m bar; actual within-bar amendment remains uncertain',
            'No historical spreads, fees, funding, book fills or live latency reconstruction',
            'Monthly cohorts start flat, exclude inherited positions and next-month exits; not client-account PnL',
            'Shock/trend/range labels are bounded price/volume proxies, not all possible market phases or news'])
    if not winner:
        raise ValueError('no development router meets the predeclared sample floor')
    selected=winner['policy']
    controls=['bbb','aaa','intersection','union']
    if args.with_frozen_classifier:
        controls+=['ccc','bc_intersection','bc_union','ac_intersection','ac_union',
            'all_intersection','all_union']
    inspected=list(dict.fromkeys([selected,*controls]))
    report['inspected_after_selection']=inspected
    report['later_archived_windows']={p:{str(y):metrics(rows_by_policy[p],timestamp(f'{y}-01-01'),
        timestamp(f'{y+1}-01-01')) for y in (2024,2025,2026)} for p in inspected}
    # Selection has finished. Every raw full-year family is now rebuilt independently.
    report['full_years']={}
    for year in (2023,2024,2025):
        print(f'full year {year}',flush=True)
        raw,year_sources=full_year(year);sources.update(year_sources)
        scores=score(raw)
        report['full_years'][str(year)]={p:metrics(route(raw,scores,p),
            timestamp(f'{year}-01-01'),timestamp(f'{year+1}-01-01')) for p in inspected}

    far=ROOT/'crypto_xperp_raw_router_audit_2_0_nov14_merged_raw_families.csv'
    near=ROOT/'crypto_xperp_raw_router_audit_0_5_nov14_merged_raw_families.csv'
    direct=runtime.joiner.join_direct_outcomes(far,near)
    m=runtime.router.MODEL
    direct=direct[direct.symbol.isin(runtime.base.ROBUST_SYMBOLS)
        &direct.risk_pct.between(m['risk_pct_min'],m['risk_pct_max'])].assign(period='direct')
    direct=attach(direct,runtime.exact.paths(ROOT/'xperp_15m_nov14_merged'))
    sources.update(direct.attrs['asset_sources'])
    direct=runtime.sequential.apply_fraction(runtime.sequential.attach_current(
        direct,runtime.exact.paths(ROOT/'xperp_15m_nov14_merged')),.5)
    scores=score(direct)
    stock_path=STOCK/'stock_factor_nov14_baseline.pkl'
    stock=stock_ready_rows(pd.read_pickle(stock_path)).assign(bot='stocks',sequential_banked=False,near_exit_time=math.nan)
    stock['sequential_runner_r']=stock.net_r
    sm=STOCK/'stock_xperp_forward_nov14_merged/manifest.json';cm=ROOT/'xperp_15m_nov14_merged/manifest.json'
    specs_path=ROOT/'shared_xperp_specs_nov14.json';specs=json.loads(specs_path.read_text())
    manifests=[json.loads(p.read_text()) for p in (cm,sm)]
    mature_end=min(v['end'] for meta in manifests for v in meta['symbols'].values() if 'end' in v)-64*900
    for meta in manifests:
        for symbol,v in meta['symbols'].items():
            if 'path' in v and specs[symbol]['instId']!=v['instrument']:
                raise ValueError('contract specification instrument mismatch')
    report['direct_last_entry_utc']=pd.Timestamp(mature_end,unit='s',tz='UTC').isoformat()
    policies={p:pd.DataFrame(route(direct,scores,p)) for p in inspected}
    baseline=ROOT/'crypto_frozen_nov14_expiry_mark_v2_be_0.5.pkl'
    policies['current_frozen']=pd.read_pickle(baseline)
    for policy,crypto in policies.items():
        frame=pd.concat([crypto.assign(bot='crypto'),stock],ignore_index=True)
        months={}
        for month in (7,8,9,10,11):
            start=timestamp(f'2026-{month:02}-01');end=timestamp(f'2026-{month+1:02}-01')
            rows=frame[frame.entry_time.ge(start)&frame.entry_time.le(mature_end)&frame.exit_time.lt(end)].to_dict('records')
            result={}
            for fraction in (5,50):
                cash=simulate(rows,specs,slots=12,bot_slots=dict(crypto=4,stocks=8),
                    simultaneous_writers=2,release_delay=900,tp1_pct=fraction)
                assert cash['peak_positions']<=12 and cash['peak_reserved_margin_pct']<=90+1e-9
                cash.pop('ledger');result[str(fraction)]=cash
            months[f'2026-{month:02}']=result
        report['results'][policy]=months
    source_paths=[pool,far,near,stock_path,sm,cm,specs_path,baseline,Path(__file__),
        Path(runtime.__file__),Path(runtime.base.__file__),Path('crypto_expected_cash_edge_nov14.py'),
        Path('asset_entry_features_research.py'),Path('lot_aware_tp1_audit.py'),
        Path('shared_factor_account_nov14_audit.py'),Path('crypto_profit_first_runner_protection_oct30_audit.py'),
        runtime.router.ARTIFACT_PATH]
    sources.update(fingerprint(source_paths))
    suffix='_with_frozen' if args.with_frozen_classifier else ''
    (ROOT/f'crypto_phase_router_nov15{suffix}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(selected=selected,full_years=report['full_years'],joint=report['results']),indent=2))


if __name__=='__main__':
    main()
