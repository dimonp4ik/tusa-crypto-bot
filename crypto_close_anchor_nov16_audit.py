"""Freeze model/exits and measure sensitivity to the observed entry reference."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import verify_crypto_profit_first_runtime_parity_nov3 as runtime
from crypto_expected_cash_edge_nov14 import metrics,timestamp

ROOT=Path('reports/audit_2026_09_08')


def reanchor(frame,candles):
    output=frame.copy()
    for symbol,part in output.groupby('symbol'):
        c=candles[symbol];lookup={int(t):i for i,t in enumerate(c['time'])}
        indices=np.asarray([lookup[int(t)] for t in part.entry_time])
        if min(indices)<1:raise ValueError('no prior close')
        original=part.entry.to_numpy(float)
        if not np.allclose(original,np.asarray(c['open'])[indices],rtol=1e-9,atol=1e-8):
            raise ValueError('source price disagrees with finalized candle open')
        entry=np.asarray(c['close'])[indices-1]
        risk=np.abs(original-part.sl.to_numpy(float))
        sign=np.where(part.direction.eq('LONG'),1.,-1.)
        output.loc[part.index,'entry']=entry
        output.loc[part.index,'sl']=entry-sign*risk
        output.loc[part.index,'tp1']=entry+sign*risk*.5
        output.loc[part.index,'risk_pct']=risk/entry
    return output


def scored_candidates(frame):
    model=runtime.router.MODEL
    eligible=frame[frame.risk_pct.between(model['risk_pct_min'],model['risk_pct_max'])].copy()
    eligible['model_probability']=runtime.portable_probability_batch(runtime.feature_matrix(eligible))
    return runtime.selected_candidates(eligible)


def resolved_rows(frame,paths):
    literal=runtime.sequential.attach_current(frame,paths)
    literal=runtime.sequential.apply_fraction(literal,.05)
    return pd.DataFrame(runtime.runtime_gate(runtime.base.standardized(literal,'entry_reference_audit')))


def main():
    cache=ROOT/'xperp_15m_nov14_merged';manifest=cache/'manifest.json'
    meta=json.loads(manifest.read_text());candles={};sources={}
    for symbol,item in meta['symbols'].items():
        if not item.get('path') or symbol not in runtime.base.ROBUST_SYMBOLS:continue
        p=Path(item['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=item['sha256']:raise ValueError('candle source hash mismatch')
        candles[symbol]=pickle.loads(raw);sources[str(p)]=digest
    far=ROOT/'crypto_xperp_raw_router_audit_2_0_nov14_merged_raw_families.csv'
    near=ROOT/'crypto_xperp_raw_router_audit_0_5_nov14_merged_raw_families.csv'
    raw=runtime.joiner.join_direct_outcomes(far,near)
    raw=raw[raw.symbol.isin(candles)].assign(period='direct')
    paths=runtime.exact.paths(cache)
    baseline_candidates=scored_candidates(raw)
    baseline=resolved_rows(baseline_candidates,paths)
    reference=ROOT/'crypto_frozen_nov14_expiry_mark_v2_be_0.5.pkl'
    expected=pd.read_pickle(reference)
    fields=['symbol','entry_time','direction','net_r','exit_time']
    pd.testing.assert_frame_equal(baseline[fields].sort_values(fields[:3]).reset_index(drop=True),
        expected[fields].sort_values(fields[:3]).reset_index(drop=True),check_dtype=False,atol=1e-9,rtol=0)
    corrected=reanchor(raw,candles)
    corrected_candidates=scored_candidates(corrected)
    same_candidates=reanchor(baseline_candidates,candles)
    outputs=dict(legacy_first_open=baseline,
        same_candidates_known_close=resolved_rows(same_candidates,paths),
        rescored_known_close=resolved_rows(corrected_candidates,paths))
    scopes=[(f'2026-{m:02}',timestamp(f'2026-{m:02}-01'),timestamp(f'2026-{m+1:02}-01'))
        for m in (7,8,9,10,11)]
    report=dict(status='ENTRY_REFERENCE_SENSITIVITY_NOT_ACCOUNT_PNL',objective_verified=False,
        baseline_parity_rows=len(baseline),raw_family_rows=len(raw),
        legacy_candidates=len(baseline_candidates),corrected_candidates=len(corrected_candidates),
        results={},limitations=['Known signal close is observable, not an executable bid/ask market fill.',
            'Frozen classifier is not retuned; replayed reference change is a sensitivity scenario.',
            '15m exits preserve stop-first ambiguity and next-bar BE amendment timing.',
            'Gross nominal 3% margin/10x standalone closed entry cohorts; no shared-account execution.',
            'Contract rounding, historical order books, funding and actual fills not reconstructed.',
            'November and initial listing coverage are partial; earlier periods were inspected before.'])
    for name,frame in outputs.items():
        rows=frame.to_dict('records')
        report['results'][name]={label:metrics(rows,start,end) for label,start,end in scopes}
        frame.to_pickle(ROOT/f'crypto_close_anchor_nov16_{name}.pkl')
        print(name,{label:round(v['return_pct'],3) for label,v in report['results'][name].items()},flush=True)
    report['sources']=sources|{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        manifest,far,near,reference,runtime.router.ARTIFACT_PATH,Path(__file__),
        Path('crypto_profit_first_runner_protection_oct30_audit.py'),
        Path('verify_crypto_profit_first_runtime_parity_nov3.py'))}
    (ROOT/'crypto_close_anchor_nov16.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
