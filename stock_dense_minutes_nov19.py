"""Immutable one-minute source for every nonnegative-scored September proposal."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import pickle
import math
import time
import pandas as pd
from threadpoolctl import threadpool_limits
from stock_session_dense_pool_nov19 import load_pool,ROOT,META
from stock_session_dense_nov19_audit import rank_events,mature_before,COLUMNS,TRAIN_END,SELECT_END
from public_candle_window import fetch_window
from liquidity_trap_joint_nov19_audit import load_frame

FOLDER=ROOT/'stock_dense_minutes_nov19'


def reconcile(minutes,parent,start,end):
    if minutes['time']!=list(range(start,end,60)):raise ValueError('minute_source_not_contiguous')
    lookup={int(t):i for i,t in enumerate(parent['time'])};positive=zero=0
    for t in range(start,end,900):
        if t not in lookup:raise ValueError('missing_parent_session_bar')
        i=lookup[t];j=(t-start)//60;indices=list(range(j,j+15))
        volume=math.fsum(float(minutes['volume'][k]) for k in indices)
        if not math.isclose(volume,float(parent['volume'][i]),rel_tol=1e-8,abs_tol=1e-8):
            raise ValueError('minute_parent_contract_volume_mismatch')
        active=[k for k in indices if minutes['volume'][k]>0]
        if not active:zero+=1;continue
        positive+=1
        observed=dict(open=minutes['open'][active[0]],high=max(minutes['high'][k] for k in active),
            low=min(minutes['low'][k] for k in active),close=minutes['close'][active[-1]])
        if any(not math.isclose(float(v),float(parent[key][i]),rel_tol=1e-8,abs_tol=1e-10) for key,v in observed.items()):
            raise ValueError('minute_parent_positive_trade_ohlc_mismatch')
    return dict(positive_parent_bars=positive,zero_parent_bars=zero)


def plan():
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__),Path('public_candle_window.py'),Path('stock_session_dense_nov19_audit.py'),
        META,ROOT/'stock_session_dense_nov19_model.pkl',ROOT/'stock_session_dense_nov19.json')}
    path=FOLDER/'plan.json'
    if path.exists():
        body=json.loads(path.read_text(encoding='utf-8'))
        if body['sources']!=sources:raise ValueError('dense_minute_plan_sources_changed')
        return body
    frame,meta=load_pool();model=pickle.loads((ROOT/'stock_session_dense_nov19_model.pkl').read_bytes())['model']
    cohort=frame[frame.entry_time.ge(TRAIN_END)&mature_before(frame,SELECT_END)].copy()
    with threadpool_limits(limits=2):cohort['predicted_price_return']=model.predict(cohort[COLUMNS].to_numpy())
    chosen=rank_events(cohort,0.)
    sessions={}
    for r in chosen.to_dict('records'):
        key=f"{r['symbol']}:{int(r['session_open'])}"
        sessions[key]=dict(symbol=r['symbol'],instrument=meta['specs'][r['symbol']]['instId'],
            start=int(r['session_open']),end=int(r['session_close']))
    body=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=sources,
        cohort='all September direction-selected frozen-model predictions>=0, not only winners/losers/ambiguous paths',
        proposals=len(chosen),proposal_ids=[[r['symbol'],r['entry_time'],r['direction']] for r in chosen.to_dict('records')],
        sessions=sessions,source='public confirmed1m native contract candles, unadjusted',
        reconciliation='contract volume and positive-child-trade OHLC must match immutable15m source',
        future_replay_profiles=['amend_next15m_parent_bar','amend_next1m_bar'],
        minute_same_bar_order='stop-first; remaining intraminute ambiguity counted, no guessed profitable order',
        model_refit=False,new_threshold_search=False,objective_verified=False,live_activation_supported=False)
    FOLDER.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(body,indent=2),encoding='utf-8');return body


def main():
    body=plan();meta=json.loads(META.read_text(encoding='utf-8'));parent={}
    base_path=Path('../tusa stocks/reports/audit_2026_09_08/stock_mark_matched_nov18/manifest.json')
    base=json.loads(base_path.read_text(encoding='utf-8'))['symbols']
    extra=json.loads((ROOT/'native_expansion_nov19/manifest.json').read_text(encoding='utf-8'))['symbols']
    for s in {v['symbol'] for v in body['sessions'].values()}:
        if s in base:p=Path('../tusa stocks')/base[s]['trade_path'];digest=base[s]['trade_sha256']
        else:p=Path(extra[s]['trade']['path']);digest=extra[s]['trade']['sha256']
        parent[s]=load_frame(p,digest)
    manifest=dict(plan_sha256=hashlib.sha256((FOLDER/'plan.json').read_bytes()).hexdigest(),windows={},complete=False)
    output=FOLDER/'manifest.json';total=len(body['sessions']);errors=0
    print('dense_minute_plan',body['proposals'],'proposals',total,'sessions',flush=True)
    for n,(key,item) in enumerate(sorted(body['sessions'].items()),1):
        try:
            for attempt in range(3):
                try:
                    frame,digest=fetch_window(item['instrument'],item['start'],item['end'],60,FOLDER/'raw')
                    break
                except Exception:
                    if attempt==2:raise
                    time.sleep(attempt+1)
            checks=reconcile(frame,parent[item['symbol']],item['start'],item['end'])
            raw=FOLDER/'raw'/f"{item['instrument']}_1m_{item['start']}_{item['end']}.json"
            value=dict(status='OK',path=str(raw),sha256=digest,bars=len(frame['time']),**checks)
        except Exception as exc:
            errors+=1;value=dict(status='ERROR',error_type=type(exc).__name__,reason=str(exc))
        manifest['windows'][key]=value
        output.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        if n%10==0 or value['status']!='OK' or n==total:
            print('dense_minute_windows',n,'/',total,'errors',errors,flush=True)
        time.sleep(.2)
    manifest['complete']=errors==0 and len(manifest['windows'])==total
    output.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('dense_minute_complete',manifest['complete'],flush=True)


if __name__=='__main__':main()
