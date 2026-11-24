"""Daily and rolling loss policies for confirmed closes in portfolio research."""

from collections import defaultdict, deque


UTC_DAY = 86_400_000
HOUR = 3_600_000


class LossPauseState:
    def __init__(self, policy):
        if policy not in ('daily', 'bot_hour', 'direction_hour', 'account_hour'):
            raise ValueError('Unknown loss policy')
        self.policy = policy
        self.day = None
        self.streak = defaultdict(int)
        self.daily_r = defaultdict(float)
        self.latched = set()
        self.recent = defaultdict(deque)
        self.until = {}

    def _group(self, bot, direction):
        if self.policy == 'account_hour':
            return 'account'
        if self.policy == 'direction_hour':
            return bot, direction
        return bot

    def _roll_day(self, now_ms):
        day = now_ms // UTC_DAY
        if day != self.day:
            self.day = day
            self.streak.clear()
            self.daily_r.clear()
            self.latched.clear()

    def confirm(self, *, bot, direction, now_ms, outcome, pnl, risk):
        if self.policy == 'daily':
            self._roll_day(now_ms)
            if bot == 'crypto':
                self.streak[bot] = self.streak[bot]+1 if outcome == 'SL' else 0
            else:
                self.streak[bot] = self.streak[bot]+1 if pnl < -1e-9 else 0
                self.daily_r[bot] += pnl / risk
                if self.streak[bot] >= 2 or self.daily_r[bot] <= -2:
                    self.latched.add(bot)
            return
        if pnl >= -1e-9:
            return
        key = self._group(bot, direction)
        recent = self.recent[key]
        while recent and recent[0] <= now_ms-HOUR:
            recent.popleft()
        recent.append(now_ms)
        if len(recent) >= 2:
            self.until[key] = now_ms+HOUR

    def blocked(self, bot, direction, now_ms):
        if self.policy == 'daily':
            self._roll_day(now_ms)
            return bot in self.latched or bot == 'crypto' and self.streak[bot] >= 3
        return now_ms < self.until.get(self._group(bot, direction), 0)
