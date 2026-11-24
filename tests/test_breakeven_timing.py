import logging
from unittest.mock import patch

from src import autotrader as at


def test_exchange_ack_and_protective_readback_are_separate(caplog):
    position = dict(id=7, direction='LONG', inst_id='TEST-X', sl_algo_id='algo', entry_px=100., sl_px=99.)
    signal = dict(id=8, entry_price=100.02, tp1_hit_at=1.)
    with (caplog.at_level(logging.INFO, logger=at.log.name),
          patch.object(at.okx, 'get_xperp_spec', return_value={'tickSz': .01}),
          patch.object(at.okx, 'get_live_protection_sl', side_effect=[(True, 99.), (True, 100.)]),
          patch.object(at.okx, 'amend_protection_sl', return_value=(True, {})) as amend,
          patch.object(at, 'at_update_position_sl') as write):
        at._move_stop_to_breakeven(position, signal, {})
        write.assert_not_called()
        at._move_stop_to_breakeven(position, signal, {})

    amend.assert_called_once_with({}, 'TEST-X', 'algo', 100.)
    write.assert_called_once_with(7, 100.)
    assert 'stage=acknowledged' in caplog.text and 'stage=confirmed' in caplog.text
