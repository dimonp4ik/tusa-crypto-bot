"""Native all-family price learner, with shared-cash chronological selection."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from account_cash_guard_nov17 import calendar_months
from account_cash_guard_nov17_audit import runtime_gate,signal_portfolio
from crypto_native_price_features_nov19 import FEATURES,FAMILIES,mature_before
from crypto_native_price_pool_nov19 import load_pool,ROOT,META
from lot_aware_tp1_audit import simulate
from shared_factor_account_nov14_audit import stock_ready_rows


def stamp(s):return pd.Timestamp(s,tz='UTC').timestamp()

TRAIN_END=stamp('2026-07-01')
SELECT_END=stamp('2026-09-01')
THRESHOLDS=(0.,.0005,.001,.002,.004,.006)
MODEL=dict(max_leaf_nodes=7,min_samples_leaf=100,max_iter=150,learning_rate=.05,
    l2_regularization=5.,early_stopping=False,random_state=20261119)
COLUMNS=['feature__'+n for n in FEATURES]
STOCK_PATH=Path('../tusa stocks/reports/audit_2026_09_08/stock_close_anchor_nov16_causal_candidates_known_close.pkl')
SPEC_PATH=ROOT/'shared_xperp_specs_nov14.json'


def train_model(frame,model=None):
    fit=frame[mature_before(frame,TRAIN_END)]
    if fit.empty or not np.isfinite(fit[COLUMNS+['runner_price_return']].to_numpy()).all():
        raise ValueError('invalid_native_fit_matrix')
    if model is None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model=HistGradientBoostingRegressor(**MODEL)
    weights=1./fit.groupby('entry_time').entry_time.transform('size').to_numpy()
    weights*=len(weights)/weights.sum()
    with threadpool_limits(limits=2):
        model.fit(fit[COLUMNS].to_numpy(),fit.runner_price_return.to_numpy(),sample_weight=weights)
    return model,dict(rows=len(fit),first_entry=float(fit.entry_time.min()),last_entry=float(fit.entry_time.max()),
        last_exit=float(fit.exit_time.max()),target='runner_price_return',full_horizon_purged=True,
        timestamp_balanced=True,ideal_partial_profit_used_as_label=False)


def rank_events(frame,threshold):
    if not np.isfinite(frame.predicted_price_return).all():raise ValueError('invalid_prediction')
    ranked=frame[frame.predicted_price_return.ge(threshold)].copy()
    ranked['family_rank']=ranked.family.map({s:i for i,s in enumerate(FAMILIES)})
    return ranked.sort_values(['entry_time','symbol','predicted_price_return','family_rank','direction'],
        ascending=[True,True,False,True,True],kind='stable').drop_duplicates(['symbol','entry_time'])


def gated(frame,threshold=None):
    if threshold is None:
        rows=signal_portfolio(frame[frame.is_frozen_candidate].to_dict('records'))
    else:
        chosen=rank_events(frame,threshold)
        rows=[dict(r,module='native_price_research',source_priority=-r['predicted_price_return'],
            module_priority=r['family_rank'],signed_price_return=r['risk_pct']*r['net_r'],
            outcome='SL' if r['sequential_outcome']=='SL' else 'TP') for r in chosen.to_dict('records')]
        rows=runtime_gate(rows,latch_loss_pause=False)
    return pd.DataFrame(rows,columns=list(frame.columns)+[n for n in
        ('source_priority','module_priority','module','outcome','signed_price_return') if n not in frame.columns])


def cash(crypto,stock,specs,*,start,end=None,priority='crypto'):
    c=crypto[crypto.entry_time.ge(start)].assign(bot='crypto')
    s=stock[stock.entry_time.ge(start)].copy()
    if end is not None:
        c=c[mature_before(c,end)]
        s=s[(s.entry_time+64*900<end)&s.exit_time.lt(end)]
    rows=pd.concat([c,s],ignore_index=True).to_dict('records')
    result=simulate(rows,specs,capital=130.,slots=12,bot_slots=dict(crypto=4,stocks=8),
        simultaneous_writers=2,release_delay=900,capture_events=True,
        tie_key=lambda r:(r['bot']!=priority,r['symbol']))
    if result['peak_positions']>12 or result['peak_reserved_margin_pct']>90+1e-9:
        raise ValueError('shared_account_cap_violated')
    if any(result['peak_by_bot'].get(b,0)>cap for b,cap in (('crypto',4),('stocks',8))):
        raise ValueError('per_bot_cap_violated')
    return result


def crypto_metrics(value):
    ledger=[r for r in value['ledger'] if r['bot']=='crypto']
    pnl=[r['bank_pnl']+r['runner_pnl'] for r in ledger]
    events=sorted((r for r in value['cash_events'] if r['bot']=='crypto'),key=lambda r:r['time'])
    curve=np.r_[0.,np.cumsum([r['pnl'] for r in events])]
    dd=100*float(np.max(np.maximum.accumulate(curve)-curve))/130.
    return dict(closed=len(ledger),winrate_pct=100*sum(p>1e-12 for p in pnl)/len(pnl) if pnl else None,
        contribution_pct=value['return_contribution_pct']['crypto'],
        additive_crypto_cash_drawdown_pct=dd,account_cash_drawdown_pct=value['cash_drawdown_pct'])


def select_threshold(frame,stock,specs):
    periods=[(stamp('2026-07-01'),stamp('2026-08-01')),(stamp('2026-08-01'),SELECT_END)]
    diagnostics={};eligible=[]
    for threshold in THRESHOLDS:
        parts=[]
        for start,end in periods:
            cohort=frame[frame.entry_time.ge(start)&mature_before(frame,end)]
            rows=gated(cohort,threshold)
            parts.extend(crypto_metrics(cash(rows,stock,specs,start=start,end=end,priority=p))
                for p in ('crypto','stocks'))
        valid=all(p['closed']>=30 and p['contribution_pct']>0 for p in parts)
        score=min(p['contribution_pct']-p['additive_crypto_cash_drawdown_pct'] for p in parts)
        diagnostics[str(threshold)]=dict(period_priority_diagnostics=parts,qualified=valid,score=score)
        if valid:eligible.append((score,sum(p['contribution_pct'] for p in parts),threshold))
    return (max(eligible)[2] if eligible else None),diagnostics


def later_report(rows,stock,specs):
    results={}
    for priority in ('crypto','stocks'):
        value=cash(rows,stock,specs,start=SELECT_END,priority=priority)
        months={m:v for m,v in calendar_months(value).items() if m>='2026-09'}
        for month,metrics in months.items():
            start=stamp(month+'-01');end=stamp(f'2026-{int(month[-2:])+1:02}-01')
            for bot in ('crypto','stocks'):
                ledger=[r for r in value['ledger'] if r['bot']==bot and start<=r['exit_time']<end]
                pnl=[r['bank_pnl']+r['runner_pnl'] for r in ledger]
                metrics[bot+'_closed']=len(pnl)
                metrics[bot+'_winrate_pct']=100*sum(p>1e-12 for p in pnl)/len(pnl) if pnl else None
        summary={k:v for k,v in value.items() if k not in ('ledger','cash_events','guard_rejections')}
        results[priority]=dict(summary=summary,calendar_months=months)
    return results


def main():
    declaration=ROOT/'crypto_native_price_nov19_protocol.json'
    source_paths=(Path(__file__),Path('crypto_native_price_features_nov19.py'),
        Path('crypto_native_price_pool_nov19.py'),Path('account_cash_guard_nov17_audit.py'),
        Path('lot_aware_tp1_audit.py'),Path('shared_factor_account_nov14_audit.py'),STOCK_PATH,SPEC_PATH)
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=sources,
        model=MODEL,thresholds=THRESHOLDS,train_end=TRAIN_END,select_end=SELECT_END,
        selection='minimum July/August and both-priority crypto contribution minus additive crypto cash drawdown',
        minimum_cash_trades_per_selection_month=30,both_months_must_profit=True,
        risk_fraction_range=[.003,.02],later_cash_start=SELECT_END,later_refit=False)
    if declaration.exists():
        old=json.loads(declaration.read_text(encoding='utf-8'))
        if old['sources']!=sources:raise ValueError('native_price_declared_source_changed')
        protocol=old
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    frame,meta=load_pool();model,training=train_model(frame)
    stock=stock_ready_rows(pd.read_pickle(STOCK_PATH)).assign(bot='stocks',sequential_banked=False,near_exit_time=np.nan)
    stock['sequential_runner_r']=stock.net_r
    specs=json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    print('native_price_fit',training,flush=True)
    with threadpool_limits(limits=2):
        selection=frame[frame.entry_time.ge(TRAIN_END)&frame.entry_time.lt(SELECT_END)].copy()
        selection['predicted_price_return']=model.predict(selection[COLUMNS].to_numpy())
        chosen,selection_report=select_threshold(selection,stock,specs)
        print('native_price_selection',chosen,selection_report,flush=True)
        later=frame[frame.entry_time.ge(SELECT_END)].copy()
        baseline=gated(later)
        results={'frozen_baseline':later_report(baseline,stock,specs)}
        selected_path=None
        if chosen is not None:
            later['predicted_price_return']=model.predict(later[COLUMNS].to_numpy())
            accepted=gated(later,chosen)
            selected_path=ROOT/'crypto_native_price_nov19_later.pkl';accepted.to_pickle(selected_path)
            results['selected_native_regressor']=later_report(accepted,stock,specs)
    model_path=ROOT/'crypto_native_price_nov19_model.pkl'
    model_path.write_bytes(pickle.dumps(dict(model=model,features=FEATURES,protocol=protocol),protocol=5))
    sources[str(META)]=hashlib.sha256(META.read_bytes()).hexdigest()
    sources[str(model_path)]=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if selected_path:sources[str(selected_path)]=hashlib.sha256(selected_path.read_bytes()).hexdigest()
    report=dict(status='EXPLORATORY_NATIVE_PRICE_REGRESSOR',objective_verified=False,live_activation_supported=False,
        protocol=protocol,pool_rows=len(frame),frozen_candidate_path_parity=meta['frozen_candidate_path_parity'],
        training=training,feature_names=FEATURES,selection=selection_report,chosen_threshold=chosen,
        later=results,sources=sources,limitations=[
            'All historical eras have been inspected previously; chronological split is no new untouched holdout.',
            'Current matched native contracts and current specifications; survivorship and historical data vintages remain.',
            'Known prior close is not an executable bid/ask; 15m last-price stop-first/next-bar BE paths are public proxies.',
            'Runner price label has no ideal TP1 bank; shared cash applies actual current lot rounding for requested 5%.',
            'Selection uses two separate flat $130 months only; later September onward account is continuous without monthly resets.',
            'Stock feed is a frozen reference baseline with its existing last20 R PAPER gate, not current client execution.',
            'No full floating-equity guard or minute pause recovery,fees,funding,transfers,publication latency or private fills.',
            'May listing coverage and November end are partial; no monthly profit promise or automatic margin change.'])
    output=ROOT/'crypto_native_price_nov19.json';output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    for name,item in results.items():
        print('native_price_later',name,{m:dict(crypto=v['return_contribution_pct']['crypto'],
            stocks=v['return_contribution_pct']['stocks'],crypto_closed=v['crypto_closed'],
            crypto_wr=v['crypto_winrate_pct']) for m,v in item['crypto']['calendar_months'].items()},flush=True)


if __name__=='__main__':main()
