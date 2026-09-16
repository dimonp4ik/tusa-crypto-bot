import unittest
from unittest.mock import patch
import minute_dual_feed as model

class DualFeedTests(unittest.TestCase):
    def row(self):return {'direction':'LONG','entry':100,'sl':90,'tp1':106,'tp2':120,'atr_at_signal':1,'size_mult':1}
    def test_callback_fills_on_execution_venue_not_signal_trailing_level(self):
        source={0:[100,108,99,107],60:[107,109,106,108],120:[108,109,107,108]}
        exchange={0:[100,104,99,102],60:[105,106,104,105],120:[105,106,104,105]}
        with patch.object(model.bt,'_post_tp1_trail_mult_bt',return_value=.006):
            r=model.simulate(self.row(),source,exchange,0,120)
        self.assertEqual(r['reason'],'signal_trail')
        self.assertEqual(r['exit_price'],105)
        self.assertEqual(r['exit_time'],60)

    def test_exchange_backstop_precedes_signal_callback(self):
        source={0:[100,108,99,107],60:[107,109,106,108],120:[108,109,107,108]}
        exchange={0:[100,104,70,102],60:[105,106,104,105],120:[105,106,104,105]}
        with patch.object(model,'STOP_CLOSE_CONFIRM',True),patch.object(model,'STOP_EXCHANGE_BACKSTOP_R',2):
            r=model.simulate(self.row(),source,exchange,0,120)
        self.assertEqual(r['reason'],'exchange_stop')
        self.assertEqual(r['exit_price'],80)

    def test_unclosed_15minute_stop_is_not_confirmed_by_one_minute(self):
        source={t:[100,101,88,89] for t in (0,60,120)}
        exchange={t:[100,101,98,99] for t in (0,60,120)}
        with patch.object(model,'STOP_CLOSE_CONFIRM',True),patch.object(model,'STOP_EXCHANGE_BACKSTOP_R',2):
            r=model.simulate(self.row(),source,exchange,0,120)
        self.assertEqual(r['reason'],'expiry')
