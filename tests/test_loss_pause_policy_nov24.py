from loss_pause_policy_nov24 import HOUR, UTC_DAY, LossPauseState
from okx_xperp_cash_tick_nov19 import simulate


def test_daily_stocks_latch_and_crypto_resumes_after_nonstop_or_utc_reset():
    state = LossPauseState('daily')

    def confirm(bot, stamp, outcome='SL', pnl=-.5):
        state.confirm(bot=bot, direction='LONG', now_ms=stamp, outcome=outcome, pnl=pnl, risk=1.)

    for stamp in (10, 20):
        confirm('stocks', stamp)
    confirm('stocks', 30, 'TARGET', 1.)
    assert state.blocked('stocks', 'SHORT', 40)
    assert not state.blocked('crypto', 'SHORT', 40)
    for stamp in (50, 60, 70):
        confirm('crypto', stamp)
    assert state.blocked('crypto', 'LONG', 80)
    confirm('crypto', 90, 'BE', -.01)
    assert not state.blocked('crypto', 'LONG', 100)
    assert not state.blocked('stocks', 'LONG', UTC_DAY)


def test_direction_policy_only_pauses_clustered_negative_exits_in_one_direction():
    state = LossPauseState('direction_hour')
    for now in (0, HOUR, HOUR+100):
        state.confirm(bot='crypto', direction='SHORT', now_ms=now, outcome='TIME', pnl=-.1, risk=1.)
        assert not state.blocked('crypto', 'LONG', now)
        assert not state.blocked('stocks', 'SHORT', now)
    assert state.blocked('crypto', 'SHORT', HOUR+101)
    assert not state.blocked('crypto', 'SHORT', 2*HOUR+100)


def test_shared_pause_uses_the_confirmed_close_clock_without_looking_ahead():
    schedule = (('A','crypto',10000,60000,98.), ('B','stocks',70000,120000,98.),
                ('C','crypto',240200,300000,104.), ('D','crypto',240201,300000,104.),
                ('E','stocks',241000,300000,104.))
    rows, specs, books = [], {}, {}
    for symbol, bot, start, end, close in schedule:
        rows.append(dict(symbol=symbol, bot=bot, direction='LONG', entry_status='eligible',
                         entry_ms=start, entry=100., stop=98., target=104., stock_reference_ready=True,
                         path={60:dict(known=True, code='SL' if close < 100 else 'TARGET',
                                       end_ms=end, be_level=100., bank=None)}))
        specs[symbol] = dict(ctVal=.1, lotSz=1, minSz=1)
        books[symbol] = {start:dict(ts=start, bids=[(99.99,6)], asks=[(100.,6)]),
                         end+200:dict(ts=end+200, bids=[(close,6)], asks=[(close+.01,6)])}
    args = dict(clock=60, lag=200, setting='requested5', release_delay_ms=120_000)
    ticks = dict.fromkeys(specs, .01)
    shared = simulate(rows, specs, ticks, books, **args, loss_pause_policy='account_hour')
    local = simulate(rows, specs, ticks, books, **args, loss_pause_policy='bot_hour')
    assert [row['symbol'] for row in shared['closed']] == ['A','B','C']
    assert shared['skipped'] == {'crypto:verified_loss_pause':1, 'stocks:verified_loss_pause':1}
    assert [row['symbol'] for row in local['closed']] == [row[0] for row in schedule]
