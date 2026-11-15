"""Test simple loss mechanisms across fixed candidates, with unchanged exits."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from crypto_phase_breakeven_nov15 import policy_frame
import verify_crypto_profit_first_runtime_parity_nov3 as runtime
from crypto_expected_cash_edge_nov14 import metrics,timestamp

ROOT=Path('reports/audit_2026_09_08')


def policies(frame):
    sign=np.where(frame.direction.eq('LONG'),1.,-1.)
    return dict(baseline=np.ones(len(frame),dtype=bool),
        avoid_counter_btc_2atr=sign*frame.btc_return20_atr>=-2,
        avoid_long_falling_btc=~(frame.direction.eq('LONG') & frame.btc_return20_atr.lt(-2)),
        avoid_breakout_exhaustion=~(frame.family.eq('breakout') & (sign*frame.btc_return20_atr>4)),
        reversal_body_confirm=~frame.family.eq('range_reversion') | (sign*frame.asset_body_atr>0) | frame.asset_body_atr.isna(),
        raised_model_score=frame.model_probability.ge(.36))


def main():
    report=dict(status='EXPLORATORY_NOT_LIVE_APPROVAL',objective_verified=False,results={},sources={},
        limitations=['Previously inspected eras; no pristine new holdout.',
            '2023-2025 use five original assets and non-native market data.',
            'Returns are nominal 3% margin/10x closed entry cohorts, not client account statements.',
            'No historical spread/funding/fill reconstruction or contract lot rounding.',
            'Entry gates do not change exits or move stops after observing outcomes.'])
    for period,name in [('2023','phase_be_full_2023_expiry_mark_v2.pkl'),
        ('2024','phase_be_full_2024_expiry_mark_v2.pkl'),
        ('2025','phase_be_full_2025_expiry_mark_v2.pkl'),
        ('direct','phase_be_direct_expiry_mark_v2.pkl')]:
        path=ROOT/name;meta=path.with_suffix('.manifest.json')
        saved=json.loads(meta.read_text())
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==saved['output_sha256']
        for source,expected in saved['inputs']['sources'].items():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected,source
        report['sources'][str(path)]=digest
        frame=pd.read_pickle(path)
        if 'asset_body_atr' not in frame:
            from asset_entry_features_research import feature_table
            frame['asset_body_atr']=np.nan
            for symbol,part in frame.groupby('symbol'):
                source=ROOT/'xperp_15m_nov14_merged'/f'{symbol}_15min_current.pkl'
                table=feature_table(pickle.loads(source.read_bytes()))
                frame.loc[part.index,'asset_body_atr']=table.reindex(
                    part.entry_time.astype('int64')).asset_body_atr.to_numpy()
        values=policy_frame(frame,(.5,.5,.5))
        tests=policies(frame)
        report['results'][period]={}
        for name,mask in tests.items():
            rows=runtime.runtime_gate(runtime.base.standardized(values[mask],'loss_mechanism_audit'))
            if period=='direct' and name=='baseline':
                expected=pd.read_pickle(ROOT/'crypto_frozen_nov14_expiry_mark_v2_be_0.5.pkl')
                fields=['symbol','entry_time','direction','net_r','exit_time']
                left=pd.DataFrame(rows)[fields].sort_values(fields[:3]).reset_index(drop=True)
                right=expected[fields].sort_values(fields[:3]).reset_index(drop=True)
                pd.testing.assert_frame_equal(left,right,check_dtype=False,atol=1e-9,rtol=0)
            start=timestamp('2026-07-01') if period=='direct' else timestamp(period+'-01-01')
            end=timestamp('2026-11-15') if period=='direct' else timestamp(str(int(period)+1)+'-01-01')
            v=metrics(rows,start,end)
            report['results'][period][name]=v
            print(period,name,'n',v['n'],'return',round(v['return_pct'],2),'DD',round(v['dd_pct'],2),
                'worst_month',round(v['worst_observed_month_pct'],2),flush=True)
    report['sources'][str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT/'crypto_loss_mechanisms_nov15.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
