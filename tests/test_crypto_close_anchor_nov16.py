import pandas as pd
import pytest
from crypto_close_anchor_nov16_audit import reanchor


def test_reference_is_known_prior_close_and_absolute_stop_distance_is_preserved():
    row=pd.DataFrame([dict(symbol='TEST',entry_time=900,entry=100.,sl=99.,direction='LONG')])
    data={'TEST':dict(time=[0,900],open=[99,100],close=[99.5,101])}
    result=reanchor(row,data).iloc[0]
    assert result.entry==99.5 and result.sl==98.5 and result.risk_pct==pytest.approx(1/99.5)
    assert row.iloc[0].entry==100  # source evidence is immutable


def test_future_first_trade_price_cannot_change_the_reanchored_decision_inputs():
    earlier=pd.DataFrame([dict(symbol='TEST',entry_time=900,entry=100.,sl=99.,direction='LONG')])
    later=earlier.assign(entry=110.,sl=109.)
    a=reanchor(earlier,{'TEST':dict(time=[0,900],open=[99,100],close=[99.5,101])})
    b=reanchor(later,{'TEST':dict(time=[0,900],open=[99,110],close=[99.5,111])})
    pd.testing.assert_frame_equal(a,b)


def test_mismatched_price_source_is_rejected_instead_of_silently_repriced():
    row=pd.DataFrame([dict(symbol='TEST',entry_time=900,entry=100.,sl=99.,direction='LONG')])
    with pytest.raises(ValueError):
        reanchor(row,{'TEST':dict(time=[0,900],open=[99,110],close=[99.5,111])})
