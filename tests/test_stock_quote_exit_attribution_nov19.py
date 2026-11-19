import pytest

from stock_quote_exit_attribution_nov19 import book_variant, minute_exit


def stock_row(*, receipt=0.0, side='LONG'):
    stop, target = (99., 102.) if side == 'LONG' else (101., 98.)
    return dict(receipt=receipt, bucket=0, geometry=dict(side=side, stop=stop, target=target),
                valid_book=True, spec=dict(ctVal='1', lotSz='.01', minSz='.01'))


def paths(opening, high, low, close, volume):
    return dict(time=list(range(0, len(close)*60, 60)), open=opening, high=high,
                low=low, close=close, volume=volume)


def test_receipt_minute_touch_cannot_be_attributed_to_hypothetical_position():
    c = paths([100., 100.], [102.1, 103.], [99.5, 100.], [100., 103.], [1., 1.])
    assert minute_exit(c, stock_row(receipt=30.5)) == {
        'known': False, 'reason': 'receipt_minute_touched_a_level'}


def test_short_gap_uses_open_even_when_same_minute_then_hits_target():
    c = paths([100., 103.], [100.1, 104.], [99.9, 97.], [100., 97.], [1., 1.])
    assert minute_exit(c, stock_row(side='SHORT', receipt=15)) == {
        'known': True, 'outcome': 'SL_GAP', 'fill_reference': 103., 'exit_time': 120}


def test_conflicting_traded_minute_stays_unresolved():
    c = paths([100.], [103.], [98.], [100.], [1.])
    assert minute_exit(c, stock_row()) == {
        'known': False, 'reason': 'stop_and_target_in_same_minute'}


@pytest.mark.parametrize(('bid', 'ask', 'expected'), [
    (99.99, 100.01, 'qualified_5bp'), (99.9, 100.1, 'wide'),
])
def test_public_book_quality_uses_full_order_at_fixed_client_margin(bid, ask, expected):
    row = stock_row()
    row['book'] = dict(bids=[[bid, 1.]], asks=[[ask, 1.]], exchange_at=0.,
                       spread_bps=20_000 * (ask-bid) / (ask+bid))
    check = book_variant(row)
    assert check['status'] == expected
    assert check['average'] == pytest.approx(ask)
    assert check['contracts'] == pytest.approx(.38)
