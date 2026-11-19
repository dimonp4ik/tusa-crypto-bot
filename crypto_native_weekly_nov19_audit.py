"""Shared-cash selection for fixed and weekly learners under client TP1 policies."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from account_cash_guard_nov17 import calendar_months
from crypto_native_price_pool_nov19 import load_pool,ROOT,META
from crypto_native_price_features_nov19 import FEATURES,mature_before
from crypto_native_price_nov19_audit import (TRAIN_END,SELECT_END,THRESHOLDS,MODEL,
    STOCK_PATH,SPEC_PATH,stamp,gated,crypto_metrics)
from crypto_native_weekly_nov19 import weekly_predictions,WINDOW_DAYS,MINIMUM_TRAINING_ROWS
from cash_tp1_policy_nov19 import simulate
from shared_factor_account_nov14_audit import stock_ready_rows

TP1_POLICIES={'requested5':dict(tp1_pct=5.,minimum_close=False),
    'half_minimum':dict(tp1_pct=50.,minimum_close=True)}


def shared_cash(crypto,stock,specs,*,start,end=None,priority='crypto',policy='requested5'):
    c=crypto[crypto.entry_time.ge(start)].assign(bot='crypto');s=stock[stock.entry_time.ge(start)].copy()
    if end is not None:
        c=c[mature_before(c,end)];s=s[(s.entry_time+64*900<end)&s.exit_time.lt(end)]
    value=simulate(pd.concat([c,s],ignore_index=True).to_dict('records'),specs,capital=130.,
        slots=12,bot_slots=dict(crypto=4,stocks=8),simultaneous_writers=2,release_delay=900,
        tie_key=lambda r:(r['bot']!=priority,r['symbol']),**TP1_POLICIES[policy])
    if value['peak_positions']>12 or value['peak_reserved_margin_pct']>90+1e-9:
        raise ValueError('weekly_shared_capacity_violation')
    if any(value['peak_by_bot'].get(bot,0)>cap for bot,cap in (('crypto',4),('stocks',8))):
        raise ValueError('weekly_local_capacity_violation')
    if abs(sum(value['return_contribution_pct'].values())-value['return_pct'])>1e-4:
        raise ValueError('weekly_contribution_accounting_mismatch')
    return value


def select(models,stock,specs):
    eligible=[];diagnostics={}
    periods=[(stamp('2026-07-01'),stamp('2026-08-01')),(stamp('2026-08-01'),SELECT_END)]
    for name,frame in models.items():
        for threshold in THRESHOLDS:
            parts=[]
            for start,end in periods:
                cohort=frame[frame.entry_time.ge(start)&mature_before(frame,end)]
                accepted=gated(cohort,threshold)
                for policy in TP1_POLICIES:
                    for priority in ('crypto','stocks'):
                        value=shared_cash(accepted,stock,specs,start=start,end=end,policy=policy,priority=priority)
                        parts.append(dict(start=start,policy=policy,priority=priority,**crypto_metrics(value)))
            valid=all(p['closed']>=30 and p['contribution_pct']>0 for p in parts)
            score=min(p['contribution_pct']-p['additive_crypto_cash_drawdown_pct'] for p in parts)
            key=name+'_'+str(threshold);diagnostics[key]=dict(parts=parts,qualified=valid,score=score)
            if valid:eligible.append((score,sum(p['contribution_pct'] for p in parts),name,threshold))
    best=max(eligible) if eligible else None
    return (dict(model=best[2],threshold=best[3],score=best[0]) if best else None),diagnostics


def later(rows,stock,specs):
    results={}
    for policy in TP1_POLICIES:
        results[policy]={}
        for priority in ('crypto','stocks'):
            value=shared_cash(rows,stock,specs,start=SELECT_END,policy=policy,priority=priority)
            months={m:v for m,v in calendar_months(value).items() if m>='2026-09'}
            for month,item in months.items():
                start=stamp(month+'-01');end=stamp(f'2026-{int(month[-2:])+1:02}-01')
                for bot in ('crypto','stocks'):
                    closed=[r for r in value['ledger'] if r['bot']==bot and start<=r['exit_time']<end]
                    pnl=[r['bank_pnl']+r['runner_pnl'] for r in closed]
                    item[bot+'_closed']=len(closed)
                    item[bot+'_winrate_pct']=100*sum(p>1e-12 for p in pnl)/len(pnl) if pnl else None
                    item[bot+'_tp1_hits_without_bank']=sum(r['tp1_hit'] and r['planned_partial']==0 for r in closed)
            summary={k:v for k,v in value.items() if k not in ('ledger','cash_events','guard_rejections')}
            results[policy][priority]=dict(summary=summary,calendar_months=months)
    return results


def main():
    model_path=ROOT/'crypto_native_price_nov19_model.pkl'
    paths=(Path(__file__),Path('crypto_native_weekly_nov19.py'),Path('cash_tp1_policy_nov19.py'),
        Path('crypto_native_price_nov19_audit.py'),Path('src/tp1_plan.py'),Path('src/okx_trader.py'),
        META,model_path,STOCK_PATH,SPEC_PATH)
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=dict(sources),model=MODEL,
        weekly_window_days=WINDOW_DAYS,minimum_training_rows=MINIMUM_TRAINING_ROWS,
        weekly_first_fit=TRAIN_END,subsequent_fits='Monday 00:00 UTC',thresholds=THRESHOLDS,
        tp1_policies=TP1_POLICIES,selection_end=SELECT_END,
        selection='minimum crypto contribution minus additive crypto cash DD across July/August, both client TP1 policies and tie priorities',
        required_minimum_cash_trades=30,require_all_selection_cases_profitable=True,
        later_threshold_retuning=False,objective_verified=False,live_activation_supported=False)
    declaration=ROOT/'crypto_native_weekly_nov19_protocol.json'
    if declaration.exists():
        prior=json.loads(declaration.read_text(encoding='utf-8'))
        if prior['sources']!=sources:raise ValueError('weekly_declared_sources_changed')
        protocol=prior
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    frame,meta=load_pool();weekly,audits=weekly_predictions(frame,start=TRAIN_END)
    scored_path=ROOT/'crypto_native_weekly_nov19_scored.pkl';weekly.to_pickle(scored_path)
    sources[str(scored_path)]=hashlib.sha256(scored_path.read_bytes()).hexdigest()
    fitted=pickle.loads(model_path.read_bytes())
    if tuple(fitted['features'])!=FEATURES:raise ValueError('unexpected_fixed_schema')
    fixed=frame[frame.entry_time.ge(TRAIN_END)].copy()
    with threadpool_limits(limits=2):
        fixed['predicted_price_return']=fitted['model'].predict(fixed[['feature__'+n for n in FEATURES]].to_numpy())
    models=dict(fixed=fixed,weekly=weekly)
    stock=stock_ready_rows(pd.read_pickle(STOCK_PATH)).assign(bot='stocks',sequential_banked=False,near_exit_time=np.nan)
    stock['sequential_runner_r']=stock.net_r;specs=json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    chosen,selection=select({name:f[f.entry_time.lt(SELECT_END)] for name,f in models.items()},stock,specs)
    print('weekly_robust_selection',chosen,flush=True)
    baseline=gated(frame[frame.entry_time.ge(SELECT_END)])
    results=dict(frozen_baseline=later(baseline,stock,specs))
    if chosen is not None:
        selected=models[chosen['model']];accepted=gated(selected[selected.entry_time.ge(SELECT_END)],chosen['threshold'])
        path=ROOT/'crypto_native_weekly_nov19_selected_later.pkl';accepted.to_pickle(path)
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        results['selected']=later(accepted,stock,specs)
    report=dict(status='NATIVE_WEEKLY_SHARED_CASH_RESEARCH',objective_verified=False,live_activation_supported=False,
        protocol=protocol,weekly_fits=audits,frozen_candidate_path_parity=meta['frozen_candidate_path_parity'],
        selection=selection,chosen=chosen,later=results,sources=sources,limitations=[
            'All eras were inspected before; new weekly rule is exploratory, not an untouched historical holdout.',
            'First fit July1, then fixed Monday schedule; 90 past days, full 48-bar maturity and exit strictly before fit.',
            'Model learns runner price return; client partial cash is separately lot-rounded, including indivisible full TP1.',
            'Client full TP1 releases account exposure earlier; the paper signal and its publication gates retain original lifetime.',
            'July/August selection uses separate flat accounts; later September onward account is continuous without monthly resets.',
            'Stock feed is a frozen native reference with last20 positive-R readiness; not current account or current stock PAPER result.',
            'Known-close entries and last-price paths are not historical executable bid/ask or confirmed private fills.',
            'Current specs assumed historically; no full-equity guard,minute recovery,floating PnL,funding,fees or latency.',
            'May listing coverage and November end partial; client setting not changed and no monthly profit promise.'])
    (ROOT/'crypto_native_weekly_nov19.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    for name,result in results.items():
        for policy,data in result.items():
            print('weekly_later',name,policy,{m:dict(crypto=v['return_contribution_pct']['crypto'],
                stocks=v['return_contribution_pct']['stocks'],crypto_n=v['crypto_closed'],crypto_wr=v['crypto_winrate_pct'])
                for m,v in data['crypto']['calendar_months'].items()},flush=True)


if __name__=='__main__':main()
