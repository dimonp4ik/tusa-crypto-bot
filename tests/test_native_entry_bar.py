from types import SimpleNamespace
from unittest.mock import patch

import main
import pytest
from src.entry_window import EntryWindowUnavailable, clip_native_entry_bar
from test_monitor_intrabar_lifecycle import signal


def exchange_rows(start, end, size, events):
    rows = []
    for at in range(start,end,size):
        o,h,l,c,v = events.get(at, (100,100,100,100,0))
        rows.append([str(at*1000),str(o),str(h),str(l),str(c),str(v),str(v),'0','1'])
    return rows


def strategy_candle(events):
    traded = [x for x in events.values() if x[4] > 0]
    return dict(time=[1800], open=[traded[0][0]], high=[max(x[1] for x in traded)],
                low=[min(x[2] for x in traded)], close=[traded[-1][3]],
                volume_contracts=[sum(x[4] for x in traded)], confirmed=[1])


def test_clip_excludes_prepublication_target_and_stop_in_an_overlapping_minute():
    events = {1800:(100,104,98,100,2),1860:(100,100.2,99.8,100,1)}
    raw = exchange_rows(1800,2700,60,events)
    finer = exchange_rows(1800,1860,1,{1810:(100,104,98,100,1),1840:(100,100,100,100,1)})
    data = clip_native_entry_bar(strategy_candle(events), opened_at=1830.5,
        bar_seconds=900,direction='LONG',sl=99,tp1=101,tp2=102,now=2750,
        load_minutes=lambda start,end: raw,load_seconds=lambda minute: finer,
        load_second_trades=lambda second: pytest.fail('quiet boundary second has no crossing'))
    assert max(data['high']) == 100.2 and min(data['low']) == 99.8
    assert data['confirmed'].count(1) == 1 and data['confirmed'][-1] == 1
    assert all(t >=1830.5 for t in data['time'])


def test_partial_strategy_window_requires_complete_closed_minutes():
    events = {1800:(100,103,100,101,1)}
    raw = exchange_rows(1800,1920,60,events)
    coarse = strategy_candle(events)
    coarse['confirmed'] = [0]
    with pytest.raises(EntryWindowUnavailable, match='incomplete'):
        clip_native_entry_bar(coarse, opened_at=1830.5,bar_seconds=900,direction='LONG',
            sl=99,tp1=101,tp2=102,now=1950,load_minutes=lambda s,e:raw[:-1],
            load_seconds=lambda s:[],load_second_trades=lambda s:[])


def monitor_state(events, *, status='OPEN', now=2750., next_bar=False):
    row = signal(reason='Frozen X-Perp module: test',opened_at=1830.5, strategy_entry_bar_ts=1800.,
                 strategy_bar_seconds=900,status=status, runner_activation_bar_ts=1800.)
    coarse = strategy_candle(events)
    if next_bar:
        next_values = dict(time=2700,open=100,high=100,low=100,close=100,volume_contracts=1,confirmed=0)
        for key,values in coarse.items():
            values.append(next_values[key])
    with (patch.object(main, 'get_open_signals', return_value=[row]),
          patch.object(main, 'get_klines_xperp', return_value=coarse),
          patch('src.binance_client.get_xperp_entry_minutes', side_effect=lambda sm,s,e:exchange_rows(s,e,60,events)),
          patch('src.binance_client.get_xperp_entry_seconds') as seconds,
          patch.object(main, 'time', SimpleNamespace(time=lambda:now)),
          patch.object(main, 'send_signal_update'),
          patch.object(main.autotrader, 'mirror_transition') as mirror,
          patch.object(main, 'update_signal_status') as update):
        main._check_open_signals()
    seconds.assert_not_called()
    return update,mirror


def test_minute_tp1_does_not_advance_the_fifteen_minute_breakeven_clock():
    events = {1800:(100,100,100,100,1),1860:(100,101.2,100,101.1,2),
              1920:(101.1,101.1,99.5,100.1,1)}
    update,mirror = monitor_state(events)
    assert update.call_args.args[1] == 'TP1_PARTIAL'
    assert update.call_args.kwargs['runner_activation_bar_ts'] == 1800.
    assert mirror.call_args.args[1] == 'TP1_PARTIAL'
    update,mirror = monitor_state(events,status='TP1_PARTIAL')
    update.assert_not_called();mirror.assert_not_called()
    update,mirror = monitor_state(events,status='TP1_PARTIAL',next_bar=True)
    assert update.call_args.args[1] == mirror.call_args.args[1] == 'BREAKEVEN'
