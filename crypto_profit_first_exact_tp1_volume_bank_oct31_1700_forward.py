"""Append-only frozen TP1-volume-bank forward check through Oct-31 17:00 UTC."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import crypto_profit_first_exact_oct31_1700_forward as refresh
import crypto_profit_first_exact_tp1_volume_bank_oct30_audit as frozen_bank

ROOT=Path('reports/audit_2026_09_08')
OLD_ROWS=ROOT/'crypto_profit_first_exact_tp1_volume_bank_oct30.pkl'
NEW_BASE=refresh.ROWS_OUT
OUT=ROOT/'crypto_profit_first_exact_tp1_volume_bank_oct31_1700_forward.json'
ROWS_OUT=ROOT/'crypto_profit_first_exact_tp1_volume_bank_oct31_1700.pkl'
NEW_ROWS_OUT=ROOT/'crypto_profit_first_exact_tp1_volume_bank_oct31_1700_new.pkl'
THRESHOLD=1.5
EXTRA=.75

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(f): return set(zip(f.symbol.astype(str),f.entry_time.astype('int64'),f.direction.astype(str),f.family.astype(str)))
def metrics(f,cutoff):
    octo=f[f.entry_time.ge(pd.Timestamp('2026-10-01',tz='UTC').timestamp())]
    unseen=f[f.entry_time.gt(cutoff)]
    return {k:{'margin_3pct':refresh.metrics(x,.03),'margin_5pct':refresh.metrics(x,.05)} for k,x in [('october',octo),('after_oct16',unseen)]}

def main():
    source=json.loads(refresh.SOURCE_REPORT.read_text(encoding='utf-8'))
    cutoff=int(pd.Timestamp(source['fresh_direct_coverage']['last_entry']).timestamp())
    base=pd.read_pickle(NEW_BASE)
    attached=frozen_bank.attach_tp1_path(base,refresh.paths(refresh.NEW_MANIFEST.parent),direct=True)
    candidate=frozen_bank.apply_policy(attached,volume_threshold=THRESHOLD,extra_close_fraction=EXTRA,direct=True)
    old=pd.read_pickle(OLD_ROWS)
    old_cutoff=int(old.entry_time.max())
    prefix=candidate[candidate.entry_time.le(old_cutoff)]
    keys=['symbol','entry_time','direction','family']
    l=old.set_index(keys); r=prefix.set_index(keys); common=l.index.intersection(r.index)
    delta=max((abs(float(l.loc[k,'net_r'])-float(r.loc[k,'net_r'])) for k in common),default=0.0)
    parity=bool(identity(old)==identity(prefix) and delta<=np.finfo(float).eps*8)
    if not parity: raise ValueError({'old':len(old),'prefix':len(prefix),'matched':len(common),'delta':delta})
    old_ids=identity(old)
    incremental=candidate[[k not in old_ids for k in zip(candidate.symbol.astype(str),candidate.entry_time.astype('int64'),candidate.direction.astype(str),candidate.family.astype(str))]].copy()
    baseline_incremental=base[[k not in old_ids for k in zip(base.symbol.astype(str),base.entry_time.astype('int64'),base.direction.astype(str),base.family.astype(str))]].copy()
    candidate.to_pickle(ROWS_OUT); incremental.to_pickle(NEW_ROWS_OUT)
    base_metrics=metrics(base,cutoff); cand_metrics=metrics(candidate,cutoff)
    inc={f'margin_{int(m*100)}pct':refresh.metrics(incremental,m) for m in (.03,.05)}
    base_inc={f'margin_{int(m*100)}pct':refresh.metrics(baseline_incremental,m) for m in (.03,.05)}
    b3=base_inc['margin_3pct']; c3=inc['margin_3pct']
    forward_pass=bool(c3['return_pct']>=b3['return_pct'] and c3['dd_pct']<=b3['dd_pct'] and c3['negative_months']<=b3['negative_months'])
    report={'status':('FROZEN_TP1_VOLUME_BANK_FORWARD_PASSED' if forward_pass else 'FROZEN_TP1_VOLUME_BANK_FORWARD_FAILED'),'ready_for_live_orders':False,'orders_possible':False,'runtime_change_allowed':False,'created_at':datetime.now(timezone.utc).isoformat(),'frozen_policy':{'volume_ratio_below':THRESHOLD,'extra_close_fraction_at_hit_bar_close':EXTRA,'total_closed_after_weak_tp1_bar':.8},'integrity':{'accepted_prefix_rows':len(prefix),'matched_rows':len(common),'maximum_net_r_delta':delta,'passes':parity},'new_accepted_rows':len(incremental),'baseline_new_forward':base_inc,'candidate_new_forward':inc,'forward_gate':{'return_noninferior':c3['return_pct']>=b3['return_pct'],'drawdown_noninferior':c3['dd_pct']<=b3['dd_pct'],'negative_months_noninferior':c3['negative_months']<=b3['negative_months'],'passes':forward_pass},'baseline':base_metrics,'candidate':cand_metrics,'sources':{str(p):digest(p) for p in (OLD_ROWS,NEW_BASE,refresh.NEW_MANIFEST,*refresh.paths(refresh.NEW_MANIFEST.parent).values(),Path(__file__).resolve())},'limitations':['The new forward interval is still too short to validate a replacement policy.','The candidate remains shadow-only and failed its first return non-inferiority gate.','Gross results exclude fees at user request.']}
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps({'status':report['status'],'integrity':report['integrity'],'new_accepted_rows':len(incremental),'baseline_new_forward':base_inc,'candidate_new_forward':inc,'forward_gate':report['forward_gate'],'baseline_october':base_metrics['october'],'candidate_october':cand_metrics['october']},indent=2))
if __name__=='__main__': main()
