"""Compare fixed delayed spread retries for the two reported crypto contracts."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from okx_xperp_l2_entry_replay_nov19 import attempt
from okx_xperp_multiday_capture_nov20 import FOLDER, file_hash
from okx_xperp_multiday_replay_nov20 import key_for, pull_samples, resolve_all


ROOT=FOLDER.parents[1]
SYMBOLS={'AAVEUSDT','AVAXUSDT'}
OFFSETS={'current':(10_000,10_200,10_800),'within_deadline':(10_000,30_000,60_000,120_000)}


def admissible(row,quotes,spec,offsets):
    selected=[]
    for offset in offsets:
        at=row['epoch']*1000+offset
        result=attempt(quotes.get(at),at,row['direction'],spec,row['stop'],row['target'])
        selected.append(dict(ms=at,**result))
        if result['status']!='wide':
            break
    final=selected[-1]
    outgoing=dict(row,entry_ms=final['ms'],entry_status=final['status'],attempts=selected)
    if final['status']=='eligible':
        outgoing['reference_entry']=final['average']
        sign=1 if row['direction']=='LONG' else -1
        worst=final['worst']
        distance=sign*(worst-row['stop'])
        reward=sign*(row['target']-worst)
        tp1=row['entry']+.5*(row['entry']-row['stop'])
        outgoing['available_reward_to_risk']=reward/distance
        outgoing['already_past_tp1']=sign*(worst-tp1)>=0
    return outgoing


def main():
    inputs=[FOLDER/'plan.json',FOLDER/'capture.json',ROOT/'paired_contract_volume_nov19/manifest.json',
            ROOT/'native_universe_nov19_snapshot.json']
    plan,capture,ref,snapshot=(json.loads(x.read_bytes()) for x in inputs)
    by_id={r['instId']:float(r['tickSz']) for r in snapshot['instruments'] if r.get('tickSz')}
    ticks={s:by_id[plan['specs'][s]['instId']] for s in SYMBOLS}
    candidates=[row for row in plan['rows'] if row['symbol'] in SYMBOLS and row['bot']=='crypto']
    needed=defaultdict(set)
    for row in candidates:
        for offsets in OFFSETS.values():
            for offset in offsets:
                ms=row['epoch']*1000+offset
                needed[key_for(row['symbol'],ms)].add(ms)
    jobs=[]
    for name,points in sorted(needed.items()):
        item=capture['books'][name]
        jobs.append((item['symbol'],item['day'],item,sorted(points)))
    quotes,checks=pull_samples(jobs,stage='AAVE/AVAX books')
    pairs=[]
    flags=Counter()
    for row in candidates:
        symbol=row['symbol']
        compared={key:admissible(row,quotes[symbol],plan['specs'][symbol],offsets)
                  for key,offsets in OFFSETS.items()}
        current,delayed=compared['current'],compared['within_deadline']
        if current['entry_status']=='wide' and delayed['entry_status']=='eligible':
            passes_geometry=(delayed['available_reward_to_risk']>=1.5 and not delayed['already_past_tp1'])
            delayed['rescue_with_bracket_room']=passes_geometry
            flags[f'{symbol}:rescued']+=1
            flags[f'{symbol}:rescued_with_bracket_room']+=passes_geometry
        pairs.append(compared)
    evidence=resolve_all([r for pair in pairs for r in pair.values()],{'files':capture['trades']},ref,ticks)
    totals={s:{key:Counter(pair[key]['entry_status'] for pair in pairs if pair[key]['symbol']==s)
               for key in OFFSETS} for s in sorted(SYMBOLS)}
    result=dict(status='FIXED_SPREAD_RETRY_RESEARCH',inputs={str(p):file_hash(p) for p in inputs},
                offsets_ms=OFFSETS,counts=totals,rescues=flags,proposals=pairs,archive_checks=checks,
                tape_checks=evidence,objective_verified=False,live_activation_supported=False,
                limitations=['Different fixed times with the same spread/depth gates. No reduced guard threshold.',
                    'Retrospective on the fixed 9–14 November proposal cohort, no fresh holdout or private account eligibility.',
                    'A rescued quote needs room for at least 1.5R and must precede TP1. Trade tape reports triggers, no new cash execution estimate.'])
    path=ROOT/'okx_avax_aave_retry_nov21.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('AAVE/AVAX',json.dumps(totals),'rescues',dict(flags),flush=True)


if __name__=='__main__':main()
