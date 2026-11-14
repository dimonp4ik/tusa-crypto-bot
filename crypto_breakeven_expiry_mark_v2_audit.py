"""Fresh labels for delayed-BE policies after the expiry accounting correction."""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
import pandas as pd
import verify_crypto_profit_first_runtime_parity_nov3 as runtime
from lot_aware_tp1_audit import simulate
from shared_factor_account_nov14_audit import stock_ready_rows

ROOT=Path('reports/audit_2026_09_08')
STOCK=Path('../tusa stocks/reports/audit_2026_09_08')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--activation-bar-stress',action='store_true')
    args=parser.parse_args()
    suffix='_activation_bar_stress' if args.activation_bar_stress else ''
    far=ROOT/'crypto_xperp_raw_router_audit_2_0_nov14_merged_raw_families.csv'
    near=ROOT/'crypto_xperp_raw_router_audit_0_5_nov14_merged_raw_families.csv'
    joined=runtime.joiner.join_direct_outcomes(far,near)
    model=runtime.router.MODEL
    joined=joined[joined.symbol.isin(runtime.base.ROBUST_SYMBOLS)
        & joined.risk_pct.between(model['risk_pct_min'],model['risk_pct_max'])].copy()
    joined['model_probability']=runtime.portable_probability_batch(runtime.feature_matrix(joined))
    selected=runtime.selected_candidates(joined).assign(period='direct')
    paths=runtime.exact.paths(ROOT/'xperp_15m_nov14_merged')
    candles={s:pickle.loads(p.read_bytes()) for (_,s),p in paths.items()}
    indexes={s:{int(t):i for i,t in enumerate(c['time'])} for s,c in candles.items()}
    stock_path=STOCK/'stock_factor_nov14_baseline.pkl'
    stocks=stock_ready_rows(pd.read_pickle(stock_path)).assign(bot='stocks',sequential_banked=False,
        near_exit_time=math.nan)
    stocks['sequential_runner_r']=stocks.net_r
    specs_path=ROOT/'shared_xperp_specs_nov14.json';specs=json.loads(specs_path.read_text())
    cm=ROOT/'xperp_15m_nov14_merged/manifest.json';sm=STOCK/'stock_xperp_forward_nov14_merged/manifest.json'
    metas=[json.loads(p.read_text()) for p in (cm,sm)]
    end=min(v['end'] for m in metas for v in m['symbols'].values() if 'end' in v)-64*900
    old_meta=json.loads((ROOT/'xperp_15m_nov11_merged/manifest.json').read_text())
    unseen_start=max(v['end'] for v in old_meta['symbols'].values())+1
    report=dict(status='RESEARCH_ONLY_NO_BE_POLICY_CHANGE',label_revision=runtime.sequential.exact.LABEL_REVISION,
        objective_verified=False,activation_bar_stress=args.activation_bar_stress,
        results={},cohort_last_entry_utc=pd.Timestamp(end,unit='s',tz='UTC').isoformat(),
        limitations=['Cohorts start flat on a hypothetical 125.64 shared account; no actual client PnL reconstruction',
            'Signal admission retains nominal 5 percent banking while client lot-rounded TP1 is a separate setting',
            'Current specifications, realized cash sizing and 900 second reservation-release assumption',
            'Historical spread, actual execution latency and fees omitted',
            'Signal protection starts next bar; stress scenario resolves unknown within-bar new-stop touches pessimistically',
            'No pristine unseen era or monthly-profit guarantee'])
    for activation in (.5,1.,1.25):
        values=[]
        for row in selected.to_dict('records'):
            outcome,runner,banked,exit_time=runtime.sequential.exact.replay_runner(
                candles[row['symbol']],indexes[row['symbol']][int(row['entry_time'])],
                direction=row['direction'],entry=row['entry'],stop=row['sl'],activation_r=activation,lock_r=0,
                activation_bar_stress=args.activation_bar_stress)
            values.append(dict(row,sequential_outcome=outcome,sequential_runner_r=runner,
                sequential_banked=banked,sequential_exit_time=exit_time))
        part=runtime.sequential.apply_fraction(pd.DataFrame(values),.05)
        admitted=runtime.runtime_gate(runtime.base.standardized(part,'expiry_mark_v2'))
        crypto=pd.DataFrame(admitted).assign(bot='crypto')
        crypto.to_pickle(ROOT/f'crypto_frozen_nov14_expiry_mark_v2_be_{activation}{suffix}.pkl')
        frame=pd.concat([crypto,stocks],ignore_index=True)
        scenarios={}
        for name,start in (('november',pd.Timestamp('2026-11-01',tz='UTC').timestamp()),('after_nov11',unseen_start)):
            rows=frame[frame.entry_time.ge(start)&frame.entry_time.le(end)].to_dict('records')
            scenarios[name]={}
            for fraction in (5,50):
                result=simulate(rows,specs,tp1_pct=fraction,slots=12,bot_slots=dict(crypto=4,stocks=8),
                    simultaneous_writers=2,release_delay=900)
                result.pop('ledger')
                assert result['peak_positions']<=12 and result['peak_reserved_margin_pct']<=90
                scenarios[name][str(fraction)]=result
        report['results'][str(activation)]=dict(crypto_signal_count=len(crypto),
            activation_ambiguous_exits=int(crypto.runner_outcome.eq('LOCK_ACTIVATION_AMBIGUOUS').sum()),
            marked_expired_runners=int(crypto.runner_outcome.eq('TP_EXPIRED').sum()),scenarios=scenarios)
    report['sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        far,near,stock_path,specs_path,cm,sm,Path(__file__),Path('lot_aware_tp1_audit.py'),
        Path('shared_factor_account_nov14_audit.py'),Path(runtime.__file__),Path(runtime.base.__file__),
        Path(runtime.exact.__file__),Path(runtime.joiner.__file__),Path(runtime.sequential.__file__),
        Path('crypto_profit_first_runner_protection_oct30_audit.py'),runtime.router.ARTIFACT_PATH)}
    for path in paths.values():
        report['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/f'crypto_breakeven_expiry_mark_v2_audit{suffix}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['results'],indent=2))


if __name__=='__main__':
    main()
