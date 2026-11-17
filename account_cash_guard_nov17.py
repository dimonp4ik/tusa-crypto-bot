"""Research wrapper for the real equity gate, using realized cash only."""
from datetime import datetime,timezone
from src.risk_limits import equity_guard


def stamp(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()


def calendar_months(result,*,capital=130.):
    events=result['cash_events'];out={}
    for month in (7,8,9,10,11):
        start,end=stamp(f'2026-{month:02}-01'),stamp(f'2026-{month+1:02}-01')
        initial=capital+sum(e['pnl'] for e in events if e['time']<start)
        if initial<=0:raise ValueError('nonpositive month opening cash')
        cash=peak=initial;dd=0.;by_bot=dict(crypto=0.,stocks=0.)
        for event in events:
            if not start<=event['time']<end:continue
            cash+=event['pnl'];by_bot[event['bot']]+=event['pnl']
            peak=max(peak,cash);dd=max(dd,(peak-cash)/peak)
        out[f'2026-{month:02}']=dict(opening_cash=initial,closing_cash=cash,
            return_pct=100*(cash/initial-1),cash_drawdown_pct=100*dd,
            return_contribution_pct={b:100*v/initial for b,v in by_bot.items()},
            entries=sum(start<=r['entry_time']<end for r in result['ledger']),
            rejected_by_account_guard=sum(start<=r['time']<end for r in result['guard_rejections']))
    return out


class RealizedCashGuard:
    def __init__(self,*,capital,max_daily_loss=.01,max_drawdown=.03):
        self.capital=capital
        self.max_daily_loss=max_daily_loss
        self.max_drawdown=max_drawdown
        self.states={}
        self.first_blocks={}
        self.first_drawdown_blocks={}

    def __call__(self,*,bot,now,equity):
        if equity<=0:
            return 'nonpositive_equity'
        day=datetime.fromtimestamp(now,timezone.utc).date().isoformat()
        state,reason=equity_guard(self.states.get(bot),equity=equity,day=day,
            max_daily_loss=self.max_daily_loss,max_drawdown=self.max_drawdown,
            peak_floor=self.capital)
        self.states[bot]=state
        if reason is not None and bot not in self.first_blocks:
            self.first_blocks[bot]=dict(time=now,equity=equity,reason=reason,**state)
        if reason=='drawdown limit' and bot not in self.first_drawdown_blocks:
            self.first_drawdown_blocks[bot]=dict(time=now,equity=equity,reason=reason,**state)
        return reason
