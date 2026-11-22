"""State for entry delays based on confirmed closes in cash replays."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PostCloseRule:
    instrument_seconds: int = 0
    direction_loss_seconds: int = 0
    direction_loss_bot: str | None = None
    loss_margin_floor: float = 0.

    def __post_init__(self):
        for duration in (self.instrument_seconds, self.direction_loss_seconds):
            if type(duration) is not int or duration < 0:
                raise ValueError('invalid post-close delay')
        if self.direction_loss_bot not in (None,'crypto','stocks'):
            raise ValueError('invalid post-close bot')
        if not math.isfinite(self.loss_margin_floor) or not 0 <= self.loss_margin_floor <= 1:
            raise ValueError('invalid post-close loss floor')


class PostCloseState:
    def __init__(self, rule):
        self.rule = rule
        self.instrument_until = {}
        self.direction_loss_until = {}

    def confirm(self, *, symbol, bot, direction, closed_at, pnl, margin):
        if self.rule.instrument_seconds:
            self.instrument_until[symbol] = closed_at + self.rule.instrument_seconds
        includes_bot = self.rule.direction_loss_bot in (None,bot)
        threshold = max(1e-9,self.rule.loss_margin_floor*margin)
        if self.rule.direction_loss_seconds and includes_bot and pnl < -threshold:
            direction_key = bot, direction
            previous = self.direction_loss_until.get(direction_key, 0)
            self.direction_loss_until[direction_key] = max(previous, closed_at + self.rule.direction_loss_seconds)

    def refusal(self, *, symbol, bot, direction, now):
        if now < self.instrument_until.get(symbol, 0):
            return 'post_close_instrument'
        if now < self.direction_loss_until.get((bot, direction), 0):
            return 'post_close_direction_loss'
        return None
