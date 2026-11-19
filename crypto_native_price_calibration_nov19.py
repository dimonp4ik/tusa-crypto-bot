"""Daily prequential residual correction from fully matured past candidate paths."""
import numpy as np
import pandas as pd

from crypto_native_price_features_nov19 import mature_before

HISTORY_DAYS=30
SHRINK_TIMESTAMPS=100.
CORRECTION_WEIGHT=.5


def calibrated_scores(frame,*,calibration_start):
    required=['entry_time','exit_time','raw_prediction','runner_price_return']
    if not np.isfinite(frame[required].to_numpy()).all():raise ValueError('invalid_calibration_inputs')
    if frame.duplicated(['symbol','entry_time','direction','family']).any():
        raise ValueError('duplicate_calibration_candidate')
    output=frame.copy();output['predicted_price_return']=output.raw_prediction
    output['calibration_group']='warmup';output['calibration_timestamps']=0
    decisions=[];days=sorted(set(int(t//86400)*86400 for t in output.entry_time if t>=calibration_start))
    for day in days:
        history=output[output.entry_time.ge(max(calibration_start,day-HISTORY_DAYS*86400))
            &mature_before(output,day)].copy()
        history['weight']=1./history.groupby('entry_time').entry_time.transform('size')
        history['residual']=history.runner_price_return-history.raw_prediction
        cells={}
        def add(key,rows,minimum):
            count=int(rows.entry_time.nunique())
            if count<minimum:return
            mean=float(np.average(rows.residual,weights=rows.weight))
            correction=CORRECTION_WEIGHT*count/(count+SHRINK_TIMESTAMPS)*mean
            cells[key]=(correction,count)
        add(('global',),history,200)
        for key,rows in history.groupby(['family','direction']):add(('family',*key),rows,100)
        for key,rows in history.groupby(['family','direction','long_btc_regime']):add(('phase',*key),rows,50)
        current=output[output.entry_time.ge(day)&output.entry_time.lt(day+86400)]
        selected_counts={}
        for row in current.itertuples():
            keys=[('phase',row.family,row.direction,row.long_btc_regime),
                ('family',row.family,row.direction),('global',)]
            key=next((key for key in keys if key in cells),None)
            if key is None:continue
            correction,count=cells[key]
            output.loc[row.Index,'predicted_price_return']=row.raw_prediction+correction
            output.loc[row.Index,'calibration_group']=key[0]
            output.loc[row.Index,'calibration_timestamps']=count
            selected_counts[key[0]]=selected_counts.get(key[0],0)+1
        decisions.append(dict(day=day,history_rows=len(history),history_timestamps=int(history.entry_time.nunique()),
            last_history_exit=float(history.exit_time.max()) if len(history) else None,
            last_history_full_horizon=float((history.entry_time+48*900).max()) if len(history) else None,
            groups_used=selected_counts))
    return output,decisions
