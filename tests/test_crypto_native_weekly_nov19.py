import numpy as np
import pandas as pd
from crypto_native_weekly_nov19 import weekly_boundaries,training_rows,weekly_predictions,COLUMNS


def test_weekly_schedule_keeps_first_cutoff_then_monday_midnights():
    start=pd.Timestamp('2026-07-01',tz='UTC').timestamp()
    end=pd.Timestamp('2026-07-20',tz='UTC').timestamp()
    assert [pd.Timestamp(t,unit='s',tz='UTC').strftime('%Y-%m-%d %H:%M') for t in weekly_boundaries(start,end)]==[
        '2026-07-01 00:00','2026-07-06 00:00','2026-07-13 00:00','2026-07-20 00:00']


def test_training_window_excludes_old_fast_unmatured_and_later_labels():
    cutoff=1000*86400
    frame=pd.DataFrame([dict(entry_time=t,exit_time=e) for t,e in [
        (cutoff-91*86400,cutoff-90*86400),(cutoff-60*900,cutoff-50*900),
        (cutoff-900,cutoff-1),(cutoff+900,cutoff+1800)]])
    assert training_rows(frame,cutoff).index.tolist()==[1]


def test_future_label_mutation_cannot_change_an_earlier_week_model(monkeypatch):
    import crypto_native_weekly_nov19 as module
    monkeypatch.setattr(module,'MINIMUM_TRAINING_ROWS',1)
    start=pd.Timestamp('2026-07-01',tz='UTC').timestamp()
    rows=[dict(entry_time=t,exit_time=e,runner_price_return=y,**{n:1. for n in COLUMNS})
        for t,e,y in [(start-60*900,start-50*900,.01),(start+900,start+1800,.02)]]
    frame=pd.DataFrame(rows)
    class Mean:
        def fit(self,x,y,sample_weight):self.value=float(np.average(y,weights=sample_weight))
        def predict(self,x):return np.full(len(x),self.value)
    first,audit=weekly_predictions(frame,start=start,model_factory=Mean)
    frame.loc[1,'runner_price_return']=-99999.
    second,_=weekly_predictions(frame,start=start,model_factory=Mean)
    np.testing.assert_array_equal(first.predicted_price_return,second.predicted_price_return)
    assert audit[0]['last_full_horizon']<start
