"""Full-calendar nominal-bank stress test of deferred breakeven, no live change."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import pandas as pd
from append_candle_snapshot import merge
from crypto_complete_years_nov11_audit import ENDS,CORE,compact
import verify_crypto_profit_first_runtime_parity_nov3 as runtime

ROOT=Path('reports/audit_2026_09_08')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--activation-bar-stress',action='store_true')
    args=parser.parse_args()
    suffix='_activation_bar_stress' if args.activation_bar_stress else ''
    report=dict(status='RESEARCH_ONLY_NO_BREAKEVEN_CHANGE',objective_verified=False,
        label_revision=runtime.sequential.exact.LABEL_REVISION,
        activation_bar_stress=args.activation_bar_stress,results={},sources={},
        limitations=['Nominal five percent bank; not lot-rounded historical client execution',
            'Standalone five-symbol crypto stress test; no native stock data in these years',
            'Annual cohorts start flat; monthly results use realized cash and can include next-January carry-out',
            'Old frozen entry model/cutoff retained; historical eras inspected previously',
            'No fees, actual publication latency or order-book reconstruction',
            'Signal protection starts next bar; optional stress resolves unknown activation-bar ordering pessimistically'])
    for year,anchors in ENDS.items():
        path=ROOT/f'crypto_complete_{year}_preportfolio_nov11.pkl'
        frame=pd.read_pickle(path);frame=frame[frame.symbol.isin(CORE)].copy()
        report['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        candles={};lookups={}
        for symbol in CORE:
            chunks=[]
            for anchor in anchors:
                p=Path('backtest_cache')/f'{symbol}_15min_18000_end{anchor}.pkl'
                raw=p.read_bytes();report['sources'][str(p)]=hashlib.sha256(raw).hexdigest()
                chunks.append(pickle.loads(raw))
            joined,_=merge(chunks[0],chunks[1]);joined,_=merge(joined,chunks[2])
            candles[symbol]=joined
            lookups[symbol]={int(t):i for i,t in enumerate(joined['time'])}
        values={}
        for activation in (.5,1.,1.25):
            labelled=[]
            for row in frame.to_dict('records'):
                result,runner,banked,end=runtime.sequential.exact.replay_runner(
                    candles[row['symbol']],lookups[row['symbol']][int(row['entry_time'])],
                    direction=row['direction'],entry=row['entry'],stop=row['sl'],
                    activation_r=activation,lock_r=0,activation_bar_stress=args.activation_bar_stress)
                labelled.append(dict(row,sequential_outcome=result,sequential_runner_r=runner,
                    sequential_banked=banked,sequential_exit_time=end))
            labels=runtime.sequential.apply_fraction(pd.DataFrame(labelled),.05)
            rows=runtime.runtime_gate(runtime.base.standardized(labels,'expiry_mark_v2'))
            rows=pd.DataFrame(rows)
            m=compact(rows)
            m['monthly_return_pct']=runtime.exact.metrics(rows,.03)['monthly_return_pct']
            m['activation_ambiguous_exits']=int(rows.runner_outcome.eq('LOCK_ACTIVATION_AMBIGUOUS').sum())
            m['first_entry_utc']=pd.Timestamp(rows.entry_time.min(),unit='s',tz='UTC').isoformat()
            m['last_entry_utc']=pd.Timestamp(rows.entry_time.max(),unit='s',tz='UTC').isoformat()
            values[str(activation)]=m
        report['results'][str(year)]=values
    for p in (Path(__file__),Path('crypto_profit_first_runner_protection_oct30_audit.py'),
              Path(runtime.__file__),Path(runtime.base.__file__),Path(runtime.exact.__file__),
              Path(runtime.sequential.__file__),Path('crypto_complete_years_nov11_audit.py'),
              Path('append_candle_snapshot.py')):
        report['sources'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/f'crypto_expiry_mark_v2_full_years{suffix}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['results'],indent=2))


if __name__=='__main__':
    main()
