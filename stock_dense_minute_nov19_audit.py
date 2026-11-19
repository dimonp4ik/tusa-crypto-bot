"""Same frozen model/thresholds with reconciled minute paths; sensitivity only."""
from datetime import datetime,timezone
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import math
import pickle
import pandas as pd
from threadpoolctl import threadpool_limits
from stock_dense_minutes_nov19 import FOLDER
from stock_dense_minute_path_nov19 import resolve,PROFILES
from public_candle_window import validate_rows
from stock_session_dense_pool_nov19 import load_pool,ROOT,META
from stock_session_dense_nov19_audit import (rank_events,mature_before,COLUMNS,TRAIN_END,SELECT_END,
    select_threshold,crypto_reference,THRESHOLDS)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args();prefix=ROOT/'stock_dense_minute_nov19'
    paths=[Path(__file__),Path('stock_dense_minutes_nov19.py'),Path('stock_dense_minute_path_nov19.py'),
        Path('public_candle_window.py'),Path('stock_session_dense_nov19_audit.py'),META,
        ROOT/'stock_session_dense_nov19_model.pkl',ROOT/'stock_session_dense_nov19_protocol.json',FOLDER/'plan.json']
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=dict(sources),profiles=PROFILES,
        thresholds=THRESHOLDS,scope='all990 nonnegative frozen-model direction-selected September proposals',
        amendment='new stop participates only from next parent15m boundary or next1m bar',
        intraminute_order='stop-first, gap-aware; same-minute ambiguity counted rather than assigned a winning order',
        expiry='last minute before scheduled session close must have real trades, otherwise unknown',
        source='complete confirmed1m sessions reconciled with frozen15m contract volume and positive-trade OHLC',
        score='same existing six thresholds for sensitivity; no new threshold grid or model fit, no later-period promotion',
        objective_verified=False,live_activation_supported=False)
    declaration=prefix.with_name(prefix.name+'_protocol.json')
    if declaration.exists():
        previous=json.loads(declaration.read_text(encoding='utf-8'))
        if previous['sources']!=sources:raise ValueError('minute_replay_declared_sources_changed')
        protocol=previous
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if args.declare_only:
        print('minute_replay_declared',protocol['declared_at'],flush=True);return
    manifest_path=FOLDER/'manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest['complete']:raise ValueError('minute_source_download_incomplete')
    if manifest['plan_sha256']!=hashlib.sha256((FOLDER/'plan.json').read_bytes()).hexdigest():
        raise ValueError('minute_download_plan_mismatch')
    plan=json.loads((FOLDER/'plan.json').read_text(encoding='utf-8'));windows={}
    for key,item in plan['sessions'].items():
        m=manifest['windows'][key]
        if m['status']!='OK':raise ValueError('unverified_minute_window')
        raw=Path(m['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=m['sha256']:raise ValueError('minute_cache_hash_changed')
        body=json.loads(raw);windows[key]=validate_rows(body['data'],item['start'],item['end'],60)
    frame,meta=load_pool();model=pickle.loads((ROOT/'stock_session_dense_nov19_model.pkl').read_bytes())['model']
    selected=frame[frame.entry_time.ge(TRAIN_END)&mature_before(frame,SELECT_END)].copy()
    with threadpool_limits(limits=2):selected['predicted_price_return']=model.predict(selected[COLUMNS].to_numpy())
    chosen=rank_events(selected,0.).to_dict('records')
    if [[r['symbol'],r['entry_time'],r['direction']] for r in chosen]!=plan['proposal_ids']:
        raise ValueError('minute_direction_selected_cohort_changed')
    specs=json.loads((ROOT/'shared_xperp_specs_nov14.json').read_text(encoding='utf-8'))|meta['specs']
    crypto=crypto_reference(TRAIN_END,SELECT_END);results={}
    for profile in PROFILES:
        rows=[];transitions=Counter();ambiguities=0;unknown=Counter();changed=0
        for r in chosen:
            key=f"{r['symbol']}:{int(r['session_open'])}";out=resolve(windows[key],r,profile)
            if not out['known']:
                unknown[out['reason']]+=1
                new=dict(r,known=False,unknown_reason=out['reason'],exit_time=r['entry_time']+32*900,
                    net_r=math.nan,signed_return=math.nan,sequential_outcome='UNKNOWN',sequential_banked=False,
                    near_exit_time=math.nan,runner_price_return=math.nan,bank_price_return=math.nan)
            else:
                price=.5*(out['bank_price_return']+out['runner_price_return']) if out['banked'] else out['runner_price_return']
                new=dict(r,known=True,unknown_reason=None,exit_time=out['exit_time'],net_r=price/r['risk_pct'],
                    signed_return=price,sequential_outcome=out['outcome'],sequential_banked=out['banked'],
                    near_exit_time=out['near_exit_time'] if out['near_exit_time'] is not None else math.nan,
                    runner_price_return=out['runner_price_return'],bank_price_return=out['bank_price_return'])
                ambiguities+=out['intraminute_stop_target_ambiguities']
                if r['known']:
                    transitions[(r['sequential_outcome'],out['outcome'])]+=1
                    changed+=abs(r['signed_return']-price)>1e-12
            rows.append(new)
        refined=pd.DataFrame(rows);cache=ROOT/f'stock_dense_minute_nov19_{profile}.pkl'
        refined.to_pickle(cache);sources[str(cache)]=hashlib.sha256(cache.read_bytes()).hexdigest()
        _,same_threshold_cases=select_threshold(refined,crypto,specs)
        results[profile]=dict(proposals=len(rows),known=int(refined.known.sum()),unknown_reasons=dict(unknown),
            changed_known_price_returns=changed,intraminute_ambiguities=ambiguities,
            transitions=[dict(coarse=a,minute=b,count=n) for (a,b),n in sorted(transitions.items())],
            same_threshold_sensitivity=same_threshold_cases,selected_threshold=None)
        print('minute_replay',profile,'known',int(refined.known.sum()),'unknown',dict(unknown),
            'changed_price',changed,'intraminute_ambiguities',ambiguities,flush=True)
    sources[str(manifest_path)]=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    report=dict(status='FROZEN_SESSION_MODEL_MINUTE_SOURCE_SENSITIVITY',protocol=protocol,sources=sources,
        results=results,objective_verified=False,live_activation_supported=False,
        limitations=['Same previously inspected market period and same frozen model; no new fit or independent holdout.',
            'Minute resolution still cannot establish order within one minute or private amendment acceptance time.',
            'Parent-bar/one-minute amendments are declared scenarios, not measured live bot latency.',
            'Public trade crossings and5bp stress are not historical bid/ask or private fills.',
            'Scheduled close without a last-minute print is unknown; never assigned the stale last15m close.',
            'Sensitivity on the existing threshold grid does not select a new live policy or validate October/November.',
            'Cash is shared with a fixed crypto reference; no full equity guard,funding/fees,liquidation or live balances reconstructed.'])
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
