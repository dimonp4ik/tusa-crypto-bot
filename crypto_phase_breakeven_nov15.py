"""Conditional runner protection with frozen entries, selection before later checks."""
import hashlib
import itertools
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import verify_crypto_profit_first_runtime_parity_nov3 as runtime
from asset_entry_features_research import attach
from crypto_expected_cash_edge_nov14 import metrics, timestamp
from crypto_phase_router_nov15 import PHASES, choose, fingerprint, full_year, phases
from lot_aware_tp1_audit import simulate
from shared_factor_account_nov14_audit import stock_ready_rows

ROOT=Path('reports/audit_2026_09_08')
STOCK=Path('../tusa stocks/reports/audit_2026_09_08')
LEVELS=(.5,1.,1.25,99.)
KEYS=['period','symbol','entry_time','direction']
FIELDS={'outcome':'sequential_outcome','runner':'sequential_runner_r',
    'banked':'sequential_banked','exit':'sequential_exit_time'}


def prefix(level):
    return 'a'+str(float(level)).replace('.','p')


def frozen_candidates(frame):
    values=frame.copy()
    values['model_probability']=runtime.portable_probability_batch(runtime.feature_matrix(values))
    values=runtime.selected_candidates(values)
    return values


def with_phases(frame,paths):
    """Unknown phase preserves the original entry and its 0.5R protection."""
    available=attach(frame,paths)
    tags=available[KEYS].copy()
    tags['phase']=phases(available)
    result=frame.drop(columns=['phase'],errors='ignore').merge(tags,on=KEYS,how='left',validate='one_to_one')
    result['phase']=result['phase'].fillna('unknown')
    return result


def attach_outcomes(frame,paths,cache,*,stress=False):
    if frame.duplicated(KEYS).any():
        raise ValueError('duplicate frozen candidate')
    sources=fingerprint([*dict.fromkeys(paths.values()),runtime.router.ARTIFACT_PATH,
        Path('crypto_profit_first_runner_protection_oct30_audit.py'),
        Path('asset_entry_features_research.py'),Path('crypto_phase_router_nov15.py')])
    identity=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    expected=dict(input_sha256=identity,sources=sources,stress=stress,revision=2)
    manifest=cache.with_suffix('.manifest.json')
    if cache.exists() and manifest.exists():
        previous=json.loads(manifest.read_text())
        if (previous['inputs']==expected
                and previous['output_sha256']==hashlib.sha256(cache.read_bytes()).hexdigest()):
            return pd.read_pickle(cache),sources|fingerprint([cache,manifest])
    output=[]
    for (period,symbol),part in frame.groupby(['period','symbol'],sort=False):
        candle=pickle.loads(paths[str(period),str(symbol)].read_bytes())
        lookup={int(t):i for i,t in enumerate(candle['time'])}
        for row in part.to_dict('records'):
            index=lookup[int(row['entry_time'])]
            if index+47>=len(candle['time']):
                raise ValueError('incomplete runner window')
            result=dict(row)
            first_targets=[]
            for level in LEVELS:
                outcome,runner,banked,end=runtime.sequential.exact.replay_runner(candle,index,
                    direction=row['direction'],entry=float(row['entry']),stop=float(row['sl']),
                    activation_r=level,lock_r=0.,activation_bar_stress=stress,
                    on_first_target=first_targets.append if level==.5 else None)
                result.update({prefix(level)+'_'+k:v for k,v in zip(FIELDS,(outcome,runner,banked,end))})
            result['near_exit_time']=first_targets[0] if first_targets else math.nan
            output.append(result)
    result=pd.DataFrame(output)
    result.to_pickle(cache)
    manifest.write_text(json.dumps(dict(inputs=expected,
        output_sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    return result,sources|fingerprint([cache,manifest])


def policy_frame(frame,policy):
    if len(policy)!=len(PHASES) or set(policy)-set(LEVELS):
        raise ValueError('unknown protection policy')
    if set(frame.phase)-set(PHASES)-{'unknown'}:
        raise ValueError('unknown entry phase')
    result=frame.copy()
    for source,destination in FIELDS.items():
        dtype=object if source=='outcome' else (bool if source=='banked' else float)
        values=frame[prefix(.5)+'_'+source].to_numpy(dtype=dtype).copy()
        for phase,level in zip(PHASES,policy):
            mask=frame.phase.eq(phase).to_numpy()
            values[mask]=frame.loc[mask,prefix(level)+'_'+source].to_numpy()
        result[destination]=values
    result['protection_activation_r']=result.phase.map(dict(zip(PHASES,policy))).fillna(.5)
    return runtime.sequential.apply_fraction(result,.05)


def policy_rows(frame,policy):
    return runtime.runtime_gate(runtime.base.standardized(policy_frame(frame,policy),'phase_be_research'))


def key(policy):
    return '/'.join(f'{v:g}' for v in policy)


def shared_scenarios(crypto,stocks,specs,cutoff,after):
    frame=pd.concat([pd.DataFrame(crypto).assign(bot='crypto'),stocks],ignore_index=True)
    scopes=[(f'2026-{m:02}',timestamp(f'2026-{m:02}-01'),timestamp(f'2026-{m+1:02}-01'))
        for m in (7,8,9,10,11)]+[('after_nov11',after,timestamp('2026-12-01'))]
    output={}
    for scope,start,end in scopes:
        selected=frame[frame.entry_time.ge(start)&frame.entry_time.le(cutoff)&frame.exit_time.lt(end)]
        values={}
        for fraction in (5,50):
            result=simulate(selected.to_dict('records'),specs,slots=12,bot_slots=dict(crypto=4,stocks=8),
                simultaneous_writers=2,release_delay=900,tp1_pct=fraction)
            assert result['peak_positions']<=12 and result['peak_reserved_margin_pct']<=90+1e-9
            assert result['peak_by_bot'].get('crypto',0)<=4 and result['peak_by_bot'].get('stocks',0)<=8
            assert abs(sum(result['return_contribution_pct'].values())-result['return_pct'])<1e-5
            result.pop('ledger');values[str(fraction)]=result
        output[scope]=values
    return output


def main():
    source=ROOT/'expected_cash_edge_literal_pool_expiry_mark_v2_nov14.pkl'
    frame=pd.read_pickle(source).drop_duplicates(['symbol','entry_time','direction','family'])
    paths=runtime.sequential.exact.candle_paths(tuple(frame.period.unique()))
    frame=with_phases(frozen_candidates(frame),paths)
    labelled,sources=attach_outcomes(frame,paths,ROOT/'phase_be_development_expiry_mark_v2.pkl')
    development=[]
    for policy in itertools.product(LEVELS,repeat=3):
        rows=policy_rows(labelled,policy)
        development.append(dict(policy=policy,
            selection=metrics(rows,timestamp('2022-01-01'),timestamp('2023-01-01')),
            confirmation=metrics(rows,timestamp('2023-01-01'),timestamp('2024-01-01'))))
    winner=choose(development)
    if winner is None:
        raise ValueError('no policy meets development sample floor')
    selected=tuple(winner['policy']);baseline=(.5,)*3
    inspected=list(dict.fromkeys([selected,baseline,(1.,)*3,(1.25,)*3,(99.,)*3]))
    report=dict(status='RESEARCH_ONLY_NO_LIVE_EXIT_CHANGE',objective_verified=False,
        label_revision='expiry_mark_v2',phase_order=PHASES,levels=LEVELS,
        cash_timing_revision='runner_derived_tp1_v1',
        entry_model='unchanged frozen classifier and first-passing family order',
        signal_first_bank=.05,development=development,selected=winner,
        inspected_after_selection=inspected,full_years={},results={},sources=sources,
        archived_pool_phase_counts=labelled.phase.value_counts().to_dict(),
        limitations=['Development uses 2022 and partial archived 2023; all eras previously inspected',
            'Phase frozen at entry, uses only preceding closed candles; no news classification',
            'Unknown/missing phase keeps the original entry and baseline 0.5R protection',
            'Protection starts next strategy candle; real exchange amendment can occur within that candle',
            'Activation stress is an adverse ordering sensitivity, not account PnL or a portfolio lower bound',
            'Full-year tests use non-native historical candles and nominal partials on original five assets',
            'Direct shared-account cohorts use current contract specs and start flat at 125.64',
            'No reconstruction of spread, fees, funding, actual latency, fills or floating equity drawdown',
            'Simulated account quotas do not establish an atomic reservation between Railway databases'])
    for year in (2023,2024,2025):
        print(f'full year {year}',flush=True)
        raw,inputs=full_year(year);sources.update(inputs)
        candidates=frozen_candidates(raw)
        folder=ROOT/f'phase_router_full_{year}_expiry_mark_v2'
        paths={(str(year),s):folder/f'{s}.pkl' for s in candidates.symbol.unique()}
        candidates=with_phases(candidates,paths)
        annual,inputs=attach_outcomes(candidates,paths,ROOT/f'phase_be_full_{year}_expiry_mark_v2.pkl')
        sources.update(inputs)
        report['full_years'][str(year)]=dict(phase_counts=annual.phase.value_counts().to_dict(),
            policies={key(p):metrics(policy_rows(annual,p),timestamp(f'{year}-01-01'),
                timestamp(f'{year+1}-01-01')) for p in inspected})

    far=ROOT/'crypto_xperp_raw_router_audit_2_0_nov14_merged_raw_families.csv'
    near=ROOT/'crypto_xperp_raw_router_audit_0_5_nov14_merged_raw_families.csv'
    direct=runtime.joiner.join_direct_outcomes(far,near)
    model=runtime.router.MODEL
    direct=direct[direct.symbol.isin(runtime.base.ROBUST_SYMBOLS)
        &direct.risk_pct.between(model['risk_pct_min'],model['risk_pct_max'])].assign(period='direct')
    paths=runtime.exact.paths(ROOT/'xperp_15m_nov14_merged')
    direct=with_phases(frozen_candidates(direct),paths)
    normal,inputs=attach_outcomes(direct,paths,ROOT/'phase_be_direct_expiry_mark_v2.pkl');sources.update(inputs)
    # An unchanged protection policy must reproduce the current route before comparisons.
    reference=ROOT/'crypto_frozen_nov14_expiry_mark_v2_be_0.5.pkl'
    fields=['symbol','entry_time','direction'];check_fields=['net_r','exit_time']
    left=pd.DataFrame(policy_rows(normal,baseline)).sort_values(fields).reset_index(drop=True)
    right=pd.read_pickle(reference).sort_values(fields).reset_index(drop=True)
    if (not left[fields].equals(right[fields]) or not np.allclose(
            left[check_fields].to_numpy(float),right[check_fields].to_numpy(float),rtol=0,atol=1e-12)):
        raise ValueError('baseline route parity failed')
    report['baseline_parity_signals']=len(left)
    banked=left.sequential_banked.to_numpy(bool)
    report['baseline_first_target_time_corrections']=int((~np.isclose(
        left.loc[banked,'near_exit_time'].to_numpy(float),right.loc[banked,'near_exit_time'].to_numpy(float),
        rtol=0,atol=0)).sum())
    report['direct_phase_counts']=normal.phase.value_counts().to_dict()
    stock_path=STOCK/'stock_factor_nov14_baseline.pkl'
    stocks=stock_ready_rows(pd.read_pickle(stock_path)).assign(bot='stocks',sequential_banked=False,near_exit_time=math.nan)
    stocks['sequential_runner_r']=stocks.net_r
    specs_path=ROOT/'shared_xperp_specs_nov14.json';specs=json.loads(specs_path.read_text())
    cm=ROOT/'xperp_15m_nov14_merged/manifest.json';sm=STOCK/'stock_xperp_forward_nov14_merged/manifest.json'
    old=ROOT/'xperp_15m_nov11_merged/manifest.json'
    manifests=[json.loads(p.read_text()) for p in (cm,sm)]
    cutoff=min(v['end'] for m in manifests for v in m['symbols'].values() if 'end' in v)-64*900
    after=max(v['end'] for v in json.loads(old.read_text())['symbols'].values())+1
    for m in manifests:
        for symbol,v in m['symbols'].items():
            if 'path' in v and specs[symbol]['instId']!=v['instrument']:
                raise ValueError('contract specification mismatch')
    report['direct_last_entry_utc']=pd.Timestamp(cutoff,unit='s',tz='UTC').isoformat()
    for p in inspected:
        report['results'][key(p)]=shared_scenarios(policy_rows(normal,p),stocks,specs,cutoff,after)
    stressed,inputs=attach_outcomes(direct,paths,ROOT/'phase_be_direct_activation_stress_v2.pkl',stress=True)
    sources.update(inputs)
    report['activation_stress']={key(p):shared_scenarios(policy_rows(stressed,p),stocks,specs,cutoff,after)
        for p in dict.fromkeys((selected,baseline))}
    sources.update(fingerprint([source,far,near,reference,stock_path,specs_path,cm,sm,old,
        Path(__file__),Path(runtime.__file__),Path(runtime.base.__file__),
        Path('crypto_phase_router_nov15.py'),Path('crypto_expected_cash_edge_nov14.py'),
        Path('lot_aware_tp1_audit.py'),Path('shared_factor_account_nov14_audit.py')]))
    (ROOT/'crypto_phase_breakeven_nov15.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(selected=selected,full_years=report['full_years'],results=report['results']),indent=2))


if __name__=='__main__':
    main()
