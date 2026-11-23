from okx_xperp_cash_tick_nov19 import simulate
from okx_xperp_tick_path_nov19 import Trade
from stock_spread_recovery_nov23 import recheck


def scenario():
    row = dict(symbol='TEST', bot='stocks', direction='LONG', epoch=0, entry_ms=1000,
               entry=100., stop=99., target=102.5, native_candidate={'backing_level': 99.5})
    first = dict(row, entry_status='wide', entry_attempts=[])
    spec = dict(ctVal=1., lotSz=.01, minSz=.01)
    return row, first, spec


def book(at, *, wide=False):
    prices = (99.9, 100.1) if wide else (99.99, 100.)
    return dict(ts=at - 20, bids=[(prices[0], 10)], asks=[(prices[1], 10)])


def test_an_intervening_wide_quote_resets_confirmation():
    row, first, spec = scenario()
    books = {at: book(at, wide=at == 11000) for at in (6000, 11000, 16000, 21000)}
    tapes = {'TEST': ([], [])}
    confirmation = recheck(row, first, spec, .01, books, tapes, lambda *_: True, 2)
    single = recheck(row, first, spec, .01, books, tapes, lambda *_: True, 1)
    assert single['entry_ms'] == 6000
    assert confirmation['entry_ms'] == 21000
    assert [v['status'] for v in confirmation['entry_attempts']] == [
        'waiting_for_second_quote', 'wide', 'waiting_for_second_quote', 'eligible',
    ]


def test_waiting_trade_cannot_cross_backing_before_entry():
    row, first, spec = scenario()
    books = {at: book(at) for at in (6000, 11000)}
    print_before_entry = Trade(10999, 1, 99.4, 1.)
    blocked = recheck(row, first, spec, .01, books, {'TEST': ([10999], [print_before_entry])},
                      lambda *_: True, 2)
    assert blocked['entry_status'] == 'backing_broken_while_waiting'


def test_cash_rechecks_the_research_band_and_geometry_at_actual_size():
    row, _, spec = scenario()
    row.update(entry_status='eligible', stock_reference_ready=True,
               path={60: dict(known=True, code='TARGET', end_ms=8000, be_level=100., bank=None)})
    quotes = {'TEST': {1000: dict(ts=995, bids=[(99.92, 10)], asks=[(100., 10)]), 8200: book(8200)}}
    calc = lambda value: simulate([value], {'TEST': spec}, {'TEST': .01}, quotes, clock=60, lag=200, setting='requested5')
    assert calc(row)['reason'] == 'spread exceeds limit'
    adjusted = dict(row, research_entry_spread=.001, research_minimum_reward_r=2., research_backing_level=99.5)
    assert calc(adjusted)['trades'] == 1
    failed = calc(dict(adjusted, research_backing_level=100.1))
    assert failed['skipped'] == {'stocks:research_geometry_at_actual_size': 1}
