import math
import numpy as np
import pandas as pd
import pytest
from stock_session_dense_nov19_audit import train_model,rank_events,gate,paper_metrics,COLUMNS,TRAIN_END


class CaptureFit:
    def fit(self,x,y,sample_weight):self.x=x;self.y=y;self.weights=sample_weight


def test_fit_purges_fast_losses_at_boundary_unknown_labels_and_balances_timestamps():
    t=TRAIN_END-40*900
    rows=[dict(entry_time=a,exit_time=a+900,known=known,signed_return=label,
        **{name:float(i) for name in COLUMNS})
        for i,(a,known,label) in enumerate([(t,True,.01),(t,True,-.01),(t-900,True,.02),
            (TRAIN_END-900,True,-1.),(t-1800,False,math.nan)])]
    model,meta=train_model(pd.DataFrame(rows),CaptureFit())
    assert meta['rows']==3 and meta['unknown_labels_not_fit']==1
    np.testing.assert_allclose(model.y,[.01,-.01,.02])
    np.testing.assert_allclose(model.weights,[.75,.75,1.5])
    assert meta['last_entry']+32*900<TRAIN_END


def test_unknown_high_scoring_direction_is_never_removed_before_ranking():
    rows=pd.DataFrame([dict(symbol='A',entry_time=0,direction=d,known=k,predicted_price_return=p)
        for d,k,p in [('LONG',False,.004),('SHORT',True,.003)]])
    chosen=rank_events(rows,.002)
    assert len(chosen)==1 and chosen.iloc[0].direction=='LONG' and not chosen.iloc[0].known


def event(start=0,known=True):
    return dict(symbol='A',entry_time=start,exit_time=10000,direction='LONG',family='session_dense',
        risk_pct=.01,signed_return=.01 if known else math.nan,net_r=1 if known else math.nan,
        sequential_outcome='TARGET' if known else 'UNKNOWN',signal_strength=1,known=known,
        unknown_reason=None if known else 'stale_reference_at_time_exit',predicted_price_return=.005)


def test_rejected_unknown_never_enters_metrics_or_affects_future_admission():
    accepted,unknown=gate(pd.DataFrame([event(),event(900,False)]),0)
    assert not unknown and len(accepted)==1 and accepted.known.all()
    assert paper_metrics(accepted)['signals']==1


def test_admitted_unknown_invalidates_metrics_and_empty_threshold_has_no_fake_signals():
    accepted,unknown=gate(pd.DataFrame([event(known=False)]),0)
    assert len(unknown)==1 and unknown[0]['symbol']=='A'
    with pytest.raises(ValueError,match='unknown_dense_paper_metrics'):paper_metrics(accepted)
    accepted,unknown=gate(pd.DataFrame([event()]),.01)
    assert not unknown and paper_metrics(accepted)['signals']==0
