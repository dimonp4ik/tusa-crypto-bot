"""Research-only causal confirmation of an already frozen native candidate."""
import math

RULES=('body','progress','strong_close','sweep_reclaim','pullback_turn')


def confirmation(candle,index,row,*,delay,rule):
    """Read only bars closed by the delayed decision, never its following open.

    The original structural stop remains fixed. Targets are then recomputed
    from the new known-close reference at 0.5R/2R, with a 1--2% risk gate.
    """
    if delay not in (1,2) or rule not in RULES:raise ValueError('unsupported confirmation')
    side=row['direction']
    if side not in ('LONG','SHORT'):raise ValueError('invalid direction')
    sign=1 if side=='LONG' else -1
    entry=float(row['entry']);stop=float(row['sl'])
    risk=sign*(entry-stop)
    if not all(math.isfinite(x) and x>0 for x in (entry,stop,risk)):
        raise ValueError('invalid original geometry')
    if index<1 or index+delay>len(candle['time']):return dict(reason='missing_confirmation')
    expected=[int(row['entry_time'])+(j-index)*900 for j in range(index-1,index+delay)]
    if list(candle['time'][index-1:index+delay])!=expected:
        return dict(reason='confirmation_gap')
    observed=[]
    for j in range(index,index+delay):
        o,h,l,c=(float(candle[k][j]) for k in ('open','high','low','close'))
        if not all(math.isfinite(x) and x>0 for x in (o,h,l,c)) or not l<=min(o,c)<=max(o,c)<=h:
            raise ValueError('invalid confirmation OHLC')
        observed.append((o,h,l,c))
    if any(sign*((l if sign>0 else h)-stop)<=0 for o,h,l,c in observed):
        return dict(reason='structural_stop_failed_before_entry')
    original_target=entry+sign*2*risk
    if any(sign*((h if sign>0 else l)-original_target)>=0 for o,h,l,c in observed):
        return dict(reason='original_target_exhausted_before_entry')
    o,h,l,new_entry=observed[-1]
    body=sign*(new_entry-o)/risk
    progress=sign*(new_entry-entry)/risk
    location=((new_entry-l) if sign>0 else (h-new_entry))/max(h-l,1e-15)
    sweep=any(sign*((lo if sign>0 else hi)-entry)<0 for op,hi,lo,cl in observed)
    allowed={
        'body':body>0,
        'progress':body>0 and progress>0,
        'strong_close':body>0 and progress>0 and location>=.7,
        'sweep_reclaim':body>0 and progress>0 and sweep,
        'pullback_turn':body>0 and progress<=0 and location>=.65,
    }[rule]
    if not allowed:return dict(reason='confirmation_rule_rejected')
    new_risk=sign*(new_entry-stop)/new_entry
    if not .01<=new_risk<=.02:return dict(reason='delayed_risk_outside_1_to_2pct')
    return dict(reason=None,entry=new_entry,sl=stop,risk_pct=new_risk,
        entry_time=float(row['entry_time'])+delay*900,confirmation_body_r=body,
        confirmation_progress_r=progress,confirmation_close_location=location,
        confirmation_sweep=sweep,confirmation_known_at=float(row['entry_time'])+delay*900)


def opportunity_key(row):
    return row['symbol'],row['direction'],row.get('setup_time',row['entry_time'])


def choose_policy(originals,variants,*,day,regime):
    """Only fully matured original horizons from the preceding 30 days.

    Skipped candidates have zero payoff per original opportunity. Overlapping
    hypothetical paths make the dispersion penalty a heuristic, never a CI.
    """
    prior=[r for r in originals if day-30*86400<=r['entry_time']
           and r['entry_time']+48*900<day]
    phase=[r for r in prior if r['btc_regime']==regime]
    if len(phase)>=100:prior=phase;scope='same_regime'
    else:scope='all_regimes'
    if len(prior)<100:return dict(policy='baseline',scope='warmup',opportunities=len(prior),scores={})
    keys=[opportunity_key(r) for r in prior];scores={}
    for name,mapping in variants.items():
        rows=[mapping.get(k) for k in keys]
        admitted=sum(r is not None for r in rows)
        if admitted<20:continue
        values=[r['risk_pct']*r['net_r'] if r is not None else 0. for r in rows]
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite mature payoff')
        mean=math.fsum(values)/len(values)
        variance=math.fsum((v-mean)**2 for v in values)/(len(values)-1)
        scores[name]=dict(score=mean-math.sqrt(variance/len(values)),mean=mean,admitted=admitted)
    best=max(scores,key=lambda name:(scores[name]['score'],name=='baseline',name)) if scores else None
    return dict(policy=best if best and scores[best]['score']>0 else 'cash',scope=scope,
        opportunities=len(prior),scores=scores)


def rotate(originals,variants):
    decisions={};selected=[]
    for original in sorted(originals,key=lambda r:(r['entry_time'],r['symbol'],r['direction'])):
        day=int(original['entry_time']//86400)*86400;key=(day,original['btc_regime'])
        if key not in decisions:
            decisions[key]=choose_policy(originals,variants,day=day,regime=original['btc_regime'])
        policy=decisions[key]['policy']
        if policy!='cash':
            row=variants[policy].get(opportunity_key(original))
            if row is not None:selected.append(row)
    return sorted(selected,key=lambda r:(r['entry_time'],r['setup_time'],r['symbol'])),[
        dict(day=key[0],btc_regime=key[1],**value) for key,value in sorted(decisions.items())]
