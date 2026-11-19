"""One fixed session regressor, September selection and later shared-cash check."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import math
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from stock_session_dense_pool_nov19 import load_pool,source_hashes,ROOT,META
from stock_session_dense_features_nov19 import FEATURES
from stock_session_joint_nov19_audit import paper_portfolio,cash_cases
from phase_scenarios_nov19_audit import stock_ready,frame_of,CRYPTO_MANIFEST
from crypto_native_price_pool_nov19 import load_pool as crypto_pool
from crypto_minimum_bank_runner_nov19_audit import checked_row,admitted_records
from crypto_active_entry_runner_nov19_audit import activity_gate
from liquidity_trap_joint_nov19_audit import load_frame
from account_cash_guard_nov17 import stamp

PREFIX='stock_session_dense_nov19'
TRAIN_START=stamp('2026-07-01');TRAIN_END=stamp('2026-09-01')
SELECT_MID=stamp('2026-09-16');SELECT_END=stamp('2026-10-01')
THRESHOLDS=(0.,.0005,.001,.002,.004,.006)
MODEL=dict(max_leaf_nodes=7,min_samples_leaf=100,max_iter=150,learning_rate=.05,
    l2_regularization=5.,early_stopping=False,random_state=20261119)
COLUMNS=['feature__'+k for k in FEATURES]


def mature_before(frame,cutoff):
    return (frame.entry_time+32*900<cutoff)&frame.exit_time.lt(cutoff)


def train_model(frame,model=None):
    chronological=frame[frame.entry_time.ge(TRAIN_START)&mature_before(frame,TRAIN_END)]
    fit=chronological[chronological.known.astype(bool)].copy()
    if fit.empty or not np.isfinite(fit[COLUMNS+['signed_return']].to_numpy()).all():
        raise ValueError('invalid_dense_session_training_matrix')
    if model is None:
        if len(fit)<500:raise ValueError('insufficient_dense_session_training_rows')
        from sklearn.ensemble import HistGradientBoostingRegressor
        model=HistGradientBoostingRegressor(**MODEL)
    weights=1/fit.groupby('entry_time').entry_time.transform('size').to_numpy()
    weights*=len(weights)/weights.sum()
    with threadpool_limits(limits=2):
        model.fit(fit[COLUMNS].to_numpy(),fit.signed_return.to_numpy(),sample_weight=weights)
    return model,dict(rows=len(fit),unknown_labels_not_fit=len(chronological)-len(fit),
        first_entry=float(fit.entry_time.min()),last_entry=float(fit.entry_time.max()),last_exit=float(fit.exit_time.max()),
        full_horizon_purged=True,timestamp_balanced=True,target='nominal50% first-target/runner price return, not win flag',
        asset_id_feature=False,outcome_fields_used_as_features=False)


def rank_events(frame,threshold):
    if not np.isfinite(frame.predicted_price_return).all() or not math.isfinite(threshold):
        raise ValueError('invalid_dense_session_prediction')
    return frame[frame.predicted_price_return.ge(threshold)].sort_values(
        ['entry_time','symbol','predicted_price_return','direction'],ascending=[True,True,False,True],
        kind='stable').drop_duplicates(['symbol','entry_time']).copy()


def gate(frame,threshold):
    ranked=rank_events(frame,threshold)
    ranked['candle_signal_strength']=ranked['signal_strength']
    ranked['signal_strength']=ranked['predicted_price_return']
    # Unknowns remain in direction selection/admission with NaN outcome fields.
    # A rejected unknown never affects the gate. An admitted one invalidates the
    # scenario before readiness, metrics or cash are computed.
    accepted=paper_portfolio(ranked)
    unknown=accepted.loc[~accepted.known.astype(bool),['symbol','entry_time','unknown_reason']].to_dict('records') if len(accepted) else []
    return accepted,unknown


def paper_metrics(frame):
    if len(frame) and not frame.known.astype(bool).all():raise ValueError('unknown_dense_paper_metrics')
    return dict(signals=len(frame),sum_price_return=float(frame.signed_return.sum()),sum_r=float(frame.net_r.sum()),
        profitable_exit_pct=100*float(frame.signed_return.gt(0).mean()) if len(frame) else None)


def crypto_reference(start,end=None):
    frame,_=crypto_pool();rows=[];manifest=json.loads(CRYPTO_MANIFEST.read_text(encoding='utf-8'))
    frame=frame[frame.is_frozen_candidate&frame.entry_time.ge(start)]
    if end is not None:frame=frame[(frame.entry_time+48*900<end)&frame.exit_time.lt(end)]
    for symbol,g in frame.groupby('symbol'):
        item=manifest['symbols'][symbol];c=load_frame(Path(item['path']),item['sha256'])
        lookup={int(t):i for i,t in enumerate(c['time'])}
        for r in g.to_dict('records'):
            i=lookup[int(r['entry_time'])]
            if activity_gate(c,i):rows.append(checked_row(r,c,i,'be_tp1',.0005))
    admitted,unknown=admitted_records(rows)
    if unknown:raise ValueError('unknown_dense_crypto_reference')
    out=frame_of(admitted);out['signed_return']=out['risk_pct']*out['net_r'];return out


def trim_months(cases,start_month,end_month):
    for values in cases.values():
        for result in values.values():
            result['calendar_months']={m:v for m,v in result['calendar_months'].items() if start_month<=m<=end_month}
    return cases


def select_threshold(scored,crypto,specs):
    cohort=scored[scored.entry_time.ge(TRAIN_END)&mature_before(scored,SELECT_END)]
    diagnostics={};eligible=[]
    for threshold in THRESHOLDS:
        accepted,unknown=gate(cohort,threshold)
        if unknown:
            diagnostics[str(threshold)]=dict(status='UNSCORABLE_ADMITTED_UNKNOWN',unknown=unknown);continue
        halves=[paper_metrics(accepted[accepted.entry_time.ge(lo)&mature_before(accepted,hi)])
            for lo,hi in ((TRAIN_END,SELECT_MID),(SELECT_MID,SELECT_END))]
        ready=stock_ready(accepted);cases=trim_months(cash_cases(crypto,ready,specs,TRAIN_END),'2026-09','2026-09')
        outcomes=[];crypto_positive=True
        for values in cases.values():
            for result in values.values():
                item=result['calendar_months'].get('2026-09',{})
                contribution=item.get('return_contribution_pct',{}).get('stocks',0)
                outcomes.append((contribution,item.get('stocks_closed',0),item.get('cash_drawdown_pct',0)))
                crypto_positive &= item.get('return_contribution_pct',{}).get('crypto',0)>0 and item.get('crypto_closed',0)>=20
        valid=all(h['signals']>=15 and h['sum_price_return']>0 and h['sum_r']>0 for h in halves)
        valid &= all(p>0 and n>=20 for p,n,_ in outcomes)
        score=min(p-dd for p,_,dd in outcomes)
        diagnostics[str(threshold)]=dict(status='RESOLVED_SELECTION_CASH',paper_halves=halves,paper_signals=len(accepted),
            ready_entries=len(ready),qualified=bool(valid),crypto_prerequisite=bool(crypto_positive),score=score,cash=cases)
        if valid:eligible.append((score,threshold))
    return (max(eligible)[1] if eligible else None),diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args();sources=source_hashes()
    paths=[Path(__file__),Path('stock_session_joint_nov19_audit.py'),Path('phase_scenarios_nov19_audit.py'),
        Path('crypto_native_price_pool_nov19.py'),Path('crypto_minimum_bank_runner_nov19_audit.py'),
        Path('crypto_active_entry_runner_nov19_audit.py'),Path('banked_runner_path_nov19.py'),
        Path('account_cash_guard_nov17_audit.py'),Path('account_cash_guard_nov17.py'),Path('cash_tp1_policy_nov19.py'),
        Path('src/tp1_plan.py'),Path('../tusa stocks/stock_module_cluster_audit.py'),CRYPTO_MANIFEST,
        ROOT/'crypto_native_price_nov19_pool_sources.json',ROOT/'shared_xperp_specs_nov14.json']
    sources.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=dict(sources),features=FEATURES,
        model=MODEL,training_start=TRAIN_START,training_end=TRAIN_END,selection_end=SELECT_END,
        thresholds=THRESHOLDS,selection_halves=['September1-15','September16-30'],
        training_label='nominal50% TP1/runner signed price return; private banking verified separately with contract rounding',
        training_weight='one total weight per contemporaneous timestamp across both directions/all assets',
        entry='every quality-qualified closed session bar, both directions, no fixed regime family prerequisite',
        entry_quality='native current trade,75% active session bars, positive first two opening bars, close within10bp ofmark',
        clock='frozen2026 cash calendar; open+30min to min(open+5h,close-30min), requires prior3 closed session VWAP',
        stop='last3 completed mark-bar extreme +0.15 regular active ATR; floor0.3%, max1%',
        tp1=.005,tp2=.015,minimum_rr=1.5,price_grid='frozen exact instrument tick, production rounding',
        market_price_stress=.0005,horizon_bars=32,expiry='same underlying cash session close',
        stock_gate='2/scan,4/hour,2/direction/hour,4nominal/economic direction,3h cooldown,2losses,daily2R',
        cash_gate='last20 completed nominal50% paper exits positive in both price andR',
        threshold_qualification='>=15positive-aggregate paper signals each half; positive stock cash and>=20stock closes inSeptember across both TP1 modes/priorities',
        threshold_score='worst stock contribution minus shared realized cash drawdown',
        client_cash='130 shared,3% each entry margin,10x,actual lots,crypto4/stocks8/shared12,90% reserved margin,two writers,15m release',
        crypto_reference='fixed activity-qualified frozen native crypto, checkedBE-atTP1; fresh publication gate per validation start',
        unknown='never flat or removed from candidate ranking; admitted unknown makes scenario unscorable',
        later='score only after fixed threshold selection, no refit, freshOctober PAPER20 and one continuousOct/Nov account',
        previously_inspected_history=True,independent_holdout_claim=False,objective_verified=False,live_activation_supported=False)
    declaration=ROOT/f'{PREFIX}_protocol.json'
    if declaration.exists():
        previous=json.loads(declaration.read_text(encoding='utf-8'))
        if previous['sources']!=sources:raise ValueError('dense_session_declared_sources_changed')
        protocol=previous
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if args.declare_only:
        print('dense_session_declared',protocol['declared_at'],flush=True);return
    frame,meta=load_pool();model,training=train_model(frame)
    model_path=ROOT/f'{PREFIX}_model.pkl'
    model_path.write_bytes(pickle.dumps(dict(model=model,features=FEATURES,protocol=protocol),protocol=5))
    print('dense_session_fit',training,flush=True)
    specs=json.loads((ROOT/'shared_xperp_specs_nov14.json').read_text(encoding='utf-8'))|meta['specs']
    selection=frame[frame.entry_time.ge(TRAIN_END)&frame.entry_time.lt(SELECT_END)].copy()
    with threadpool_limits(limits=2):selection['predicted_price_return']=model.predict(selection[COLUMNS].to_numpy())
    chosen,diagnostics=select_threshold(selection,crypto_reference(TRAIN_END,SELECT_END),specs)
    decision=dict(threshold=chosen,selection=diagnostics,model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        pool_sha256=meta['sha256'],protocol_sha256=hashlib.sha256(declaration.read_bytes()).hexdigest())
    decision_path=ROOT/f'{PREFIX}_selection.json';decision_path.write_text(json.dumps(decision,indent=2,allow_nan=False),encoding='utf-8')
    print('dense_session_selected',chosen,flush=True);later=None
    if chosen is not None:
        forward=frame[frame.entry_time.ge(SELECT_END)].copy()
        with threadpool_limits(limits=2):forward['predicted_price_return']=model.predict(forward[COLUMNS].to_numpy())
        accepted,unknown=gate(forward,chosen)
        if unknown:later=dict(status='UNSCORABLE_ADMITTED_UNKNOWN',unknown=unknown)
        else:
            ready=stock_ready(accepted)
            cases=trim_months(cash_cases(crypto_reference(SELECT_END),ready,specs,SELECT_END),'2026-10','2026-11')
            later=dict(status='RESOLVED_RESEARCH_CASH',paper=paper_metrics(accepted),ready_entries=len(ready),cash=cases)
    sources.update({str(META):hashlib.sha256(META.read_bytes()).hexdigest(),
        str(model_path):decision['model_sha256'],str(decision_path):hashlib.sha256(decision_path.read_bytes()).hexdigest()})
    report=dict(status='DENSE_SESSION_PRICE_LEARNER_SHARED_CASH',protocol=protocol,sources=sources,training=training,
        pool=dict(rows=meta['rows'],known=meta['known'],unknown_reasons=meta['unknown_reasons'],sha256=meta['sha256']),
        chosen_threshold=chosen,selection=diagnostics,later=later,objective_verified=False,live_activation_supported=False,
        limitations=['Previously inspected history/current snapshot selection; chronology is enforced, no untouched holdout claimed.',
            'Training excludes unknown labels and may have censoring bias; ranking/admission never silently excludes unknown candidates.',
            'Nominal50% price label is not private fill PnL; client5% and minimum50% use actual lot rounding in separate cash replay.',
            'Known-close plus5bp stresses/mark features are proxies, not historical executable quotes, publication latency or actual fills.',
            'No full floating-equity guard,funding/fees,liquidation or live-account recovery reconstructed.',
            'Fresh September selection andOctober validation accounts, continuous within each stage; November partial.',
            'Both bots share cash/caps but current stock production remains PAPER and crypto reference is research only.',
            'No candidate can prove30% each bot each calendar month or justify live trading from this study alone.'])
    (ROOT/f'{PREFIX}.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
