from datetime import datetime, timezone

from okx_avax_aave_retry_nov21 import admissible
from okx_xperp_multiday_capture_nov20 import local_trade_days
from okx_xperp_multiday_replay_nov20 import cases_for


def test_trade_archive_dates_cover_the_utc8_boundary_without_future_overlap():
    before = int(datetime(2026, 11, 13, 15, 59, tzinfo=timezone.utc).timestamp())
    assert list(local_trade_days(before, before+60).values()) == ['2026-11-13']
    assert list(local_trade_days(before, before+61).values()) == ['2026-11-13', '2026-11-14']


def test_missing_entry_archive_cannot_be_treated_as_a_normal_market_rejection():
    result = cases_for([dict(symbol='AAVEUSDT',entry_ms=10,entry_status='unknown_quote')],{}, {}, {})
    assert result == dict(status='UNSCORABLE_ENTRY_ARCHIVE', unknown=[('AAVEUSDT',10)])


def test_delayed_spread_replay_requires_a_fresh_quote_and_bracket():
    row = dict(symbol='AVAXUSDT',bot='crypto',epoch=0,entry=100.,stop=99.,target=102.,direction='LONG')
    spec = dict(ctVal=1,lotSz=.01,minSz=.01)
    wide = dict(ts=9999,bids=[(99.9,2)],asks=[(100.1,2)])
    narrow = dict(ts=29999,bids=[(99.99,2)],asks=[(100.01,2)])
    result = admissible(row,{10000:wide,30000:narrow},spec,(10_000,30_000))
    assert result['entry_status']=='eligible'
    assert result['entry_ms']==30_000 and result['available_reward_to_risk']>1.5
    narrow['ts']=20000
    result = admissible(row,{10000:wide,30000:narrow},spec,(10_000,30_000))
    assert result['entry_status']=='stale order book'
