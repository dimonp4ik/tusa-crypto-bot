"""Fixed leave-one-out breadth hypotheses, replaying admission and joint cash."""
import hashlib
import json
import math
import pickle
from pathlib import Path

import pandas as pd

import crypto_close_anchor_nov16_audit as anchor
from crypto_causal_runner_grid_nov16_audit import cash_months,portfolio
from crypto_runner_exit_path_nov16 import replay
from crypto_peer_breadth_nov17 import POLICIES,allows,peer_features
from crypto_causal_rule_rotation_nov17 import rotate
from shared_factor_account_nov14_audit import stock_ready_rows

ROOT=Path('reports/audit_2026_09_08')
STOCK=Path('../tusa stocks/reports/audit_2026_09_08')


def main():
    print('load_verified_native_sources',flush=True)
    manifest=ROOT/'xperp_15m_nov14_merged/manifest.json'
    meta=json.loads(manifest.read_text(encoding='utf-8'))
    candles={};sources={}
    for symbol,item in meta['symbols'].items():
        if not item.get('path') or symbol not in anchor.runtime.base.ROBUST_SYMBOLS:
            continue
        path=Path(item['path']);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('native candle hash mismatch')
        candles[symbol]=pickle.loads(raw);sources[str(path)]=item['sha256']
    far=ROOT/'crypto_xperp_raw_router_audit_2_0_nov14_merged_raw_families.csv'
    near=ROOT/'crypto_xperp_raw_router_audit_0_5_nov14_merged_raw_families.csv'
    raw=anchor.runtime.joiner.join_direct_outcomes(far,near)
    raw=raw[raw.symbol.isin(candles)].assign(period='direct')
    candidates=anchor.scored_candidates(anchor.reanchor(raw,candles))
    print('causal_candidates',len(candidates),flush=True)
    lookups={s:{int(t):i for i,t in enumerate(c['time'])} for s,c in candles.items()}
    records=[]
    for row in candidates.to_dict('records'):
        path=replay(candles[row['symbol']],lookups[row['symbol']][int(row['entry_time'])],
                    direction=row['direction'],entry=float(row['entry']),stop=float(row['sl']),
                    target_r=2.,activation_r=.5,trail_distance_r=None,stress=False)
        records.append(dict(row,peer=peer_features(candles,row['symbol'],row['entry_time']),
            sequential_outcome=path['outcome'],sequential_runner_r=path['runner_r'],
            sequential_banked=path['banked'],sequential_exit_time=path['exit_time'],
            exit_time=path['exit_time'],near_exit_time=path['near_exit_time'] if path['near_exit_time'] is not None else math.nan,
            net_r=.025+.95*path['runner_r'] if path['banked'] else path['runner_r']))
    frame=pd.DataFrame(records)
    expected_path=ROOT/'crypto_close_anchor_nov16_rescored_known_close.pkl'
    expected=pd.read_pickle(expected_path)
    baseline=pd.DataFrame(portfolio(frame,operational_pause=False))
    fields=['symbol','entry_time','direction','net_r','exit_time']
    pd.testing.assert_frame_equal(baseline[fields].sort_values(fields[:3]).reset_index(drop=True),
        expected[fields].sort_values(fields[:3]).reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=0)
    stock_path=STOCK/'stock_close_anchor_nov16_causal_candidates_known_close.pkl'
    stock=stock_ready_rows(pd.read_pickle(stock_path)).assign(bot='stocks',sequential_banked=False,near_exit_time=math.nan)
    stock['sequential_runner_r']=stock.net_r
    spec_path=ROOT/'shared_xperp_specs_nov14.json'
    specs=json.loads(spec_path.read_text(encoding='utf-8'))
    print('baseline_parity_passed',len(baseline),flush=True)
    months=[(f'2026-{m:02}',anchor.timestamp(f'2026-{m:02}-01'),anchor.timestamp(f'2026-{m+1:02}-01')) for m in (7,8,9,10,11)]
    report=dict(status='FIXED_BREADTH_HYPOTHESES_NOT_NEW_HOLDOUT',objective_verified=False,
        live_change_approved=False,candidates=len(records),baseline_parity_rows=len(baseline),
        policy_count=len(POLICIES),rotation_variants=1,results={},
        limitations=[
            'All native months were inspected previously; no untouched historical validation is claimed.',
            'Five fixed guards are evaluated after frozen first-family selection; no family fallback is added.',
            'Two narrower guards were added after inspecting the first five; rotation is a further exploratory hypothesis.',
            'Daily rotation uses 30 prior days with exits strictly before UTC midnight, minimum 100 candidates and 20 admitted.',
            'Rotation uses overlapping hypothetical payoffs; its dispersion penalty is not a statistical confidence bound.',
            'Native leave-one-out peers use strictly prior closed bars; current membership/specifications are assumed.',
            'Model prices use known prior close, which is observable but not a historical executable bid/ask.',
            '15m exits retain stop-first ambiguity and next-bar breakeven; private amendments are not reconstructed.',
            'Legacy baseline parity and operational pause proxy are distinct; publication and quote rejections are absent.',
            'Shared cash starts flat each month at $130 with 3% margin,10x,5% TP1 and contract rounding.',
            'Shared cash reserves two-writer headroom and 15m delayed release; configured coordinator uses one serialized writer.',
            'Actual latched account drawdown guards, floating equity, transfers, fees,funding and latency are not reproduced.',
            'Closed entry cohorts exclude inherited positions and month-crossing exits; July/November are partial.',
            'No claim of actual customer PnL or guaranteed monthly profitability.'])
    for policy in POLICIES:
        mask=[allows(policy,row,row['peer']) for row in records]
        kept=frame.loc[mask].copy()
        legacy=portfolio(kept,operational_pause=False)
        operational=portfolio(kept,operational_pause=True)
        result=dict(candidates_kept=len(kept),candidates_removed=len(records)-len(kept),
            missing_peer_context=sum(row['peer']['peers']<6 for row in records),
            legacy_model={label:anchor.metrics(legacy,start,end) for label,start,end in months},
            operational_pause_proxy={label:anchor.metrics(operational,start,end) for label,start,end in months},
            joint_cash=cash_months(operational,stock,specs))
        report['results'][policy]=result
        print(policy,dict(kept=len(kept),monthly={label:round(x['return_pct'],3) for label,x in result['operational_pause_proxy'].items()}),flush=True)
    rotated,decisions=rotate(records)
    kept=pd.DataFrame(rotated)
    legacy=portfolio(kept,operational_pause=False)
    operational=portfolio(kept,operational_pause=True)
    report['results']['daily_causal_rotation']=dict(candidates_kept=len(kept),
        candidates_removed=len(records)-len(kept),
        legacy_model={label:anchor.metrics(legacy,start,end) for label,start,end in months},
        operational_pause_proxy={label:anchor.metrics(operational,start,end) for label,start,end in months},
        joint_cash=cash_months(operational,stock,specs))
    report['daily_rotation_decisions']=decisions
    print('daily_causal_rotation',dict(kept=len(kept),monthly={label:round(x['return_pct'],3)
        for label,x in report['results']['daily_causal_rotation']['operational_pause_proxy'].items()}),flush=True)
    report['candidate_diagnostics']=[dict(symbol=r['symbol'],direction=r['direction'],family=r['family'],
        entry_time=r['entry_time'],exit_time=r['exit_time'],risk_pct=r['risk_pct'],
        btc_regime=r['btc_regime'],model_r=r['net_r'],peer=r['peer'],
        allowed={p:allows(p,r,r['peer']) for p in POLICIES}) for r in records]
    for p in (manifest,far,near,expected_path,stock_path,spec_path,Path(__file__),
              Path('crypto_peer_breadth_nov17.py'),Path('crypto_runner_exit_path_nov16.py'),
              Path('crypto_causal_rule_rotation_nov17.py'),
              Path('crypto_causal_runner_grid_nov16_audit.py'),Path('crypto_runtime_shadow_portfolio_oct3_audit.py'),
              Path('lot_aware_tp1_audit.py'),Path('shared_factor_account_nov14_audit.py'),
              Path(anchor.__file__),anchor.runtime.router.ARTIFACT_PATH):
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report['sources']=sources
    (ROOT/'crypto_peer_breadth_nov17.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
