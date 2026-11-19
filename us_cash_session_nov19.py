"""Frozen 2026 underlying cash calendar for research, not X-Perp availability."""
from datetime import datetime,date,time,timezone
from zoneinfo import ZoneInfo

ET=ZoneInfo('America/New_York')
SOURCES=('https://www.nyse.com/trade/hours-calendars',
    'https://www.nasdaq.com/holiday-trading-hours')
HOLIDAYS=frozenset(date.fromisoformat(x) for x in (
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25',
    '2026-06-19','2026-07-03','2026-09-07','2026-11-26','2026-12-25'))
EARLY=frozenset(date.fromisoformat(x) for x in ('2026-11-27','2026-12-24'))


def session_bounds(stamp):
    d=datetime.fromtimestamp(stamp,timezone.utc).astimezone(ET).date()
    if d.year!=2026:raise ValueError('research_cash_calendar_year_not_supported')
    if d.weekday()>=5 or d in HOLIDAYS:return None
    return (datetime.combine(d,time(9,30),ET).timestamp(),
        datetime.combine(d,time(13 if d in EARLY else 16),ET).timestamp())
