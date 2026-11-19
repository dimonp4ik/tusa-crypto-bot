"""Fixed weekly learning rule using only fully matured prior native paths."""
from datetime import datetime,timezone,timedelta
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from crypto_native_price_features_nov19 import FEATURES,mature_before
from crypto_native_price_nov19_audit import MODEL

WINDOW_DAYS=90
MINIMUM_TRAINING_ROWS=2000
COLUMNS=['feature__'+n for n in FEATURES]


def weekly_boundaries(start,end):
    first=datetime.fromtimestamp(start,timezone.utc)
    if first.hour or first.minute or first.second:raise ValueError('weekly_start_must_be_utc_midnight')
    values=[float(start)]
    monday=first+timedelta(days=(7-first.weekday())%7)
    if monday==first:monday+=timedelta(days=7)
    while monday.timestamp()<=end:
        values.append(monday.timestamp());monday+=timedelta(days=7)
    return values


def training_rows(frame,cutoff):
    return frame[frame.entry_time.ge(cutoff-WINDOW_DAYS*86400)&mature_before(frame,cutoff)]


def weekly_predictions(frame,*,start,model_factory=None):
    if model_factory is None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model_factory=lambda:HistGradientBoostingRegressor(**MODEL)
    output=frame[frame.entry_time.ge(start)].copy();output['predicted_price_return']=np.nan
    output['model_fit_cutoff']=np.nan;audits=[]
    cuts=weekly_boundaries(start,float(output.entry_time.max()))
    for index,cutoff in enumerate(cuts):
        end=cuts[index+1] if index+1<len(cuts) else float('inf')
        prediction=output.entry_time.ge(cutoff)&output.entry_time.lt(end)
        if not prediction.any():continue
        train=training_rows(frame,cutoff)
        if len(train)<MINIMUM_TRAINING_ROWS:raise ValueError('insufficient_weekly_training_rows')
        if not np.isfinite(train[COLUMNS+['runner_price_return']].to_numpy()).all():
            raise ValueError('invalid_weekly_training_data')
        weights=1./train.groupby('entry_time').entry_time.transform('size').to_numpy()
        weights*=len(weights)/weights.sum()
        model=model_factory()
        with threadpool_limits(limits=2):
            model.fit(train[COLUMNS].to_numpy(),train.runner_price_return.to_numpy(),sample_weight=weights)
            scores=model.predict(output.loc[prediction,COLUMNS].to_numpy())
        if not np.isfinite(scores).all():raise ValueError('invalid_weekly_prediction')
        output.loc[prediction,'predicted_price_return']=scores
        output.loc[prediction,'model_fit_cutoff']=cutoff
        audit=dict(cutoff=cutoff,train_rows=len(train),train_timestamps=int(train.entry_time.nunique()),
            first_train_entry=float(train.entry_time.min()),last_train_entry=float(train.entry_time.max()),
            last_train_exit=float(train.exit_time.max()),
            last_full_horizon=float((train.entry_time+48*900).max()),scored_rows=int(prediction.sum()))
        if audit['last_train_exit']>=cutoff or audit['last_full_horizon']>=cutoff:
            raise ValueError('weekly_label_leak')
        audits.append(audit)
        print('weekly_fit',cutoff,'rows',len(train),'scored',audit['scored_rows'],flush=True)
    if output.predicted_price_return.isna().any():raise ValueError('unscored_weekly_candidate')
    return output,audits
