import pytest

from okx_xperp_cash_tick_nov19 import simulate
from okx_xperp_tick_path_nov19 import Trade, reconcile, resolve_ticks
from post_close_replay_nov23 import PostCloseRule, PostCloseState


def signal():
    return dict(symbol='TEST',bot='crypto',direction='LONG',epoch=0,entry=100.,stop=98.,target=104.)


def test_two_causal_amendment_clocks_can_change_the_exit():
    tape=[Trade(10_000,1,98.,1.),Trade(50_000,2,101.1,1.),
          Trade(60_100,3,99.9,1.),Trade(120_000,4,104.1,1.)]
    kwargs=dict(signal=signal(),entered_at=10_000,entry_px=100.05,tick_size=.01)
    early=resolve_ticks(tape,[x.ms for x in tape],amend_seconds=60,**kwargs)
    late=resolve_ticks(tape,[x.ms for x in tape],amend_seconds=900,**kwargs)
    assert early['code']=='BE' and early['end_ms']==60_100
    assert late['code']=='TARGET' and late['end_ms']==120_000
    assert late['bank']==early['bank']==dict(ms=50_000,price=101.1)


def test_tape_reconciliation_detects_missing_contracts_even_at_same_price():
    frame=dict(time=[0,900],open=[100.,100.],high=[101.,100.],low=[100.,100.],
               close=[101.,100.],volume_contracts=[2.,0.])
    tape=[Trade(1,1,100.,1.),Trade(2,2,101.,1.)]
    assert reconcile(tape,frame,start=0,end=1800)=={}
    assert reconcile(tape[:-1],frame,start=0,end=1800)=={0:'volume'}


def test_shared_cash_uses_one_lot_when_client_requests_minimum_valid_bank():
    row=dict(signal(),entry_status='eligible',entry_ms=10_000,
             path={60:dict(known=True,code='BE',end_ms=60_000,be_level=100.,bank=dict(ms=30_000))})
    quotes={'TEST':{10_000:dict(ts=9999,bids=[(99.99,6)],asks=[(100.,6)]),
                    30_200:dict(ts=30199,bids=[(101.,6)],asks=[(101.01,6)]),
                    60_200:dict(ts=60199,bids=[(99.99,6)],asks=[(100.,6)])}}
    specs={'TEST':dict(ctVal=.1,lotSz=1,minSz=1)}
    data=[row],specs,{'TEST':.01},quotes
    old=simulate(*data,clock=60,lag=200,setting='requested5')
    new=simulate(*data,clock=60,lag=200,setting='half_minimum')
    assert old['closed'][0]['qty']==new['closed'][0]['qty']==3
    assert old['closed'][0]['bank_qty']==0
    assert new['closed'][0]['bank_qty']==1
    assert new['closed'][0]['bank_pnl']==pytest.approx(.1)
    assert new['closed'][0]['runner_pnl']==pytest.approx(-.002)
    assert new['peak_reservations']==1


def test_simultaneous_bots_share_twelve_slots_and_one_margin_balance():
    proposals, specs, ticks, quotes = [], {}, {}, {}
    for bot, count in (('crypto', 5), ('stocks', 9)):
        for index in range(count):
            symbol = f'{bot}{index}'
            path = dict(known=True, code='SL', end_ms=60_000, be_level=100., bank=None)
            proposals.append(dict(signal(), symbol=symbol, bot=bot, entry_ms=10_000,
                                  entry_status='eligible', stock_reference_ready=True, path={60:path}))
            specs[symbol] = dict(ctVal=.1, lotSz=1, minSz=1)
            ticks[symbol] = .01
            quotes[symbol] = {10_000:dict(ts=9999,bids=[(99.99,6)],asks=[(100.,6)]),
                              60_200:dict(ts=60199,bids=[(98.,6)],asks=[(98.01,6)])}
    result = simulate(proposals,specs,ticks,quotes,clock=60,lag=200,setting='requested5')
    assert result['trades']==result['peak_reservations']==12
    assert result['skipped']=={'crypto:position_cap':1,'stocks:position_cap':1}
    assert result['peak_margin_pct']==pytest.approx(100*36/130)


def test_loss_delay_only_starts_after_confirmation_and_is_local_to_one_bot_side():
    schedule = (
        ('A','crypto',10_000,60_000,98.),
        ('B','crypto',70_000,120_000,102.),
        ('C','crypto',1_100_000,1_160_000,102.),
        ('D','stocks',1_100_000,1_160_000,102.),
        ('E','crypto',1_860_200,1_920_000,102.),
    )
    proposals, specs, ticks, books = [], {}, {}, {}
    for symbol,bot,entry,end,sale in schedule:
        proposals.append(dict(signal(),symbol=symbol,bot=bot,entry_ms=entry,
            entry_status='eligible',stock_reference_ready=True,
            path={60:dict(known=True,code='SL' if sale<100. else 'TARGET',end_ms=end,be_level=100.,bank=None)}))
        specs[symbol] = dict(ctVal=.1,lotSz=1,minSz=1)
        ticks[symbol] = .01
        books[symbol] = {
            entry:dict(ts=entry,bids=[(99.99,6)],asks=[(100.,6)]),
            end+200:dict(ts=end+200,bids=[(sale,6)],asks=[(sale+.01,6)]),
        }
    score = simulate(proposals,specs,ticks,books,clock=60,lag=200,setting='requested5',
                     post_close_rule=PostCloseRule(direction_loss_seconds=1800))
    assert [trade['symbol'] for trade in score['closed']] == ['A','B','D','E']
    assert score['skipped'] == {'crypto:post_close_direction_loss':1}
    prompt = simulate(proposals,specs,ticks,books,clock=60,lag=200,setting='requested5',
                      post_close_rule=PostCloseRule(direction_loss_seconds=1800),release_delay_ms=0)
    assert [trade['symbol'] for trade in prompt['closed']] == ['A','D','E']
    assert prompt['skipped'] == {'crypto:post_close_direction_loss':2}


def test_optional_loss_floor_and_stock_scope_exclude_small_losses_and_crypto():
    state = PostCloseState(PostCloseRule(direction_loss_seconds=3600,
                                        direction_loss_bot='stocks',loss_margin_floor=.02))
    def closed(bot, pnl):
        state.confirm(symbol='A',bot=bot,direction='LONG',closed_at=60,pnl=pnl,margin=4.)
        return state.refusal(symbol='B',bot=bot,direction='LONG',now=960)

    assert closed('crypto',-.4) is None
    assert closed('stocks',-.03) is None
    assert closed('stocks',-.4) == 'post_close_direction_loss'
