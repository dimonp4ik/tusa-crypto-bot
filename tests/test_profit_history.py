import time,unittest
from unittest.mock import patch
from src.okx_trader import get_realized_pnl_since
class ProfitHistoryTests(unittest.TestCase):
    def test_complete_closed_position_uses_net_value(self):
        now=time.time();row=dict(uTime=str((now-1)*1000),cTime=str((now-10)*1000),type='2',ccy='USDC',posId='1',instId='X',realizedPnl='-3.2')
        with patch('src.okx_trader._request',return_value=(True,[row])):
            self.assertEqual(get_realized_pnl_since({},now-20),(True,-3.2))
    def test_incomplete_or_ambiguous_history_never_returns_profit(self):
        now=time.time();base=dict(uTime=str((now-1)*1000),cTime=str((now-10)*1000),type='2',ccy='USDC',posId='1',instId='X',realizedPnl='20')
        variants=[[base]*100,[{**base,'realizedPnl':'nan'}],[{**base,'cTime':str((now-100)*1000)}],[{**base,'type':'1'}],[{**base,'ccy':'BTC'}],[base,base]]
        for rows in variants:
            with patch('src.okx_trader._request',return_value=(True,rows)):
                self.assertFalse(get_realized_pnl_since({},now-20)[0])
    def test_outside_history_window_does_not_query(self):
        with patch('src.okx_trader._request') as request:
            self.assertFalse(get_realized_pnl_since({},0)[0]);request.assert_not_called()
