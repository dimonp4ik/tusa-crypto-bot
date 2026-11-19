import numpy as np
import pandas as pd
from crypto_native_price_nov19_audit import COLUMNS,TRAIN_END,SELECT_END,train_model,rank_events,select_threshold
from crypto_native_price_features_nov19 import mature_before


class Capture:
    def fit(self,x,y,sample_weight):self.x=x;self.y=y;self.weight=sample_weight


def test_native_training_excludes_all_later_and_unmatured_horizon_labels():
    rows=[dict(entry_time=t,exit_time=e,runner_price_return=y,**{n:1. for n in COLUMNS})
        for t,e,y in [(TRAIN_END-60*900,TRAIN_END-40*900,.01),
            (TRAIN_END-900,TRAIN_END-1,-100),(TRAIN_END+900,TRAIN_END+1800,100),
            (SELECT_END+900,SELECT_END+1800,10000)]]
    model=Capture();train_model(pd.DataFrame(rows),model)
    np.testing.assert_array_equal(model.y,[.01])
    assert mature_before(pd.DataFrame(rows),TRAIN_END).tolist()==[True,False,False,False]


def test_family_selection_ignores_outcome_and_resolves_ties_in_fixed_family_order():
    frame=pd.DataFrame([dict(symbol='A',entry_time=1,direction='LONG',family=f,
        predicted_price_return=.003,runner_price_return=y)
        for f,y in [('breakout',100),('range_reversion',-100),('trend_pullback',1000)]])
    assert rank_events(frame,.001).family.tolist()==['range_reversion']
    frame['runner_price_return']*=-100
    assert rank_events(frame,.001).family.tolist()==['range_reversion']


def test_threshold_selection_cannot_see_september_labels_or_candidates():
    frame=pd.DataFrame([dict(symbol='A',entry_time=SELECT_END+900,exit_time=SELECT_END+1800,
        family='breakout',direction='LONG',predicted_price_return=.1,runner_price_return=1e9,
        net_r=1e9,risk_pct=.01,entry=100.,sequential_outcome='TP_RUNNER',
        sequential_banked=True,sequential_runner_r=2.,near_exit_time=SELECT_END+1800)])
    stocks=frame.iloc[:0].assign(bot='stocks')
    chosen,diagnostics=select_threshold(frame,stocks,{})
    assert chosen is None
    assert all(p['closed']==0 for d in diagnostics.values() for p in d['period_priority_diagnostics'])
