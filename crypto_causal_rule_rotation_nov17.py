"""Daily policy decisions using only prior completed hypothetical candidates."""
import math
from crypto_peer_breadth_nov17 import POLICIES,allows


def choose_policy(history, *, day_start, regime):
    prior=[r for r in history if day_start-30*86400<=r['entry_time']<day_start
           and r['entry_time']<=r['exit_time']<day_start]
    phase=[r for r in prior if r['btc_regime']==regime]
    if len(phase)>=100:
        prior=phase;scope='same_btc_regime'
    else:
        scope='all_btc_regimes'
    if len(prior)<100:
        return dict(policy='baseline',scope='warmup_baseline',completed=len(prior),scores={})
    scores={}
    for policy in POLICIES:
        admitted=[allows(policy,r,r['peer']) for r in prior]
        if sum(admitted)<20:
            continue
        values=[r['risk_pct']*r['net_r'] if keep else 0. for r,keep in zip(prior,admitted)]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('invalid historical payoff')
        mean=sum(values)/len(values)
        variance=sum((v-mean)**2 for v in values)/(len(values)-1)
        # Dispersion penalty, not an iid confidence bound: candidates overlap.
        score=mean-math.sqrt(variance/len(values))
        scores[policy]=dict(score=score,mean_per_opportunity=mean,admitted=sum(admitted))
    best=max(scores,key=lambda p:(scores[p]['score'],-POLICIES.index(p))) if scores else None
    return dict(policy=best if best is not None and scores[best]['score']>0 else 'cash',
                scope=scope,completed=len(prior),scores=scores)


def rotate(records):
    decisions={};kept=[]
    for row in sorted(records,key=lambda r:(r['entry_time'],r['symbol'])):
        day_start=int(row['entry_time']//86400)*86400
        key=(day_start,row['btc_regime'])
        if key not in decisions:
            decisions[key]=choose_policy(records,day_start=day_start,regime=row['btc_regime'])
        choice=decisions[key]['policy']
        if choice!='cash' and allows(choice,row,row['peer']):
            kept.append(row)
    return kept,[dict(day_start=k[0],btc_regime=k[1],**v) for k,v in sorted(decisions.items())]
