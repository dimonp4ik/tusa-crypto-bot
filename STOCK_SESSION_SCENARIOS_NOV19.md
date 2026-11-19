# Native stock-session research: no live strategy qualification

The original two-bot/common-account objective remains unverified. No entry rule,
client preference, risk limit, or PAPER restriction changed in production.

## Declared scenarios and data

The study uses eleven existing and six expanded matched native X-Perp mark/trade
contracts, frozen specs and tick grids. It covers available 2026 contract history
through partial November. Current-snapshot universe/turnover selection is hindsight
for earlier months; these are not independent holdouts or a historical universe.

The underlying cash-session context follows the verified 2026 NYSE/Nasdaq holiday
and early-close schedules, 09:30 New York opening and DST. July2 has a regular
session; July3 is closed; November27 and December24 close13:00. This is a strategy
context, not an assertion that the X-Perp itself stops trading. Other years fail
explicitly. Sources: https://www.nyse.com/trade/hours-calendars and
https://www.nasdaq.com/holiday-trading-hours.

V1 declares three scenarios: a closed retest of the first30-minute opening range,
a failed breakout followed by a separate acceptance candle, and a pullback/reclaim
around the current session's native-volume-weighted typical15-minute price.
The latter is a candle approximation, not consolidated underlying-stock VWAP.
Both QQQ and SPY provide closed same-session context; self-confirmation is omitted.
SOXS has inverse macro sign and shares the economic-direction cap with opposite
nominal-direction positions. This is an imperfect market proxy, not a sector or
portfolio beta model.

Own entries require current native trades, at least75% active session candles,
last-trade close within10bp of mark, valid complete opening range, and known
regular-session ATR. Structural stops include0.15ATR buffer, minimum0.3% and an
adaptive maximum up to1.5%. First targets0.5–1%, final targets1–2% with known
opposing-level caps and minimum1.5 reward/risk. All published brackets satisfy
their floors after the production tick-rounding function. Entries are adverse5bp
known-close proxies, not limit orders or guaranteed market fills.

V1 produced20 candidate paths:13 opening retests and seven VWAP pullbacks, all
scorable on both trade and mark paths. Only seven pass the combined August/
September publication rails. All seven lack twenty prior completed PAPER exits.
Thus there are no cash-ready stock entries or qualified variants.

## Explicit extended hypothesis

After recording V1, a separately declared extension retains V1 source, declaration,
and report. It adds the first actual closed opening-range impulse crossing in
each direction/day: prior close inside, body>=0.5ATR, close location>=0.75,
volume>=1.5x and extension<=1ATR. It does not restart impulse entries repeatedly
while already outside the range. Existing scenarios retain priority on a collision.

Reference-only macro context uses closed mark session return and an unweighted
typical-candle session mean. This indicator needs no contemporaneous reference
trade and is explicitly not VWAP or an executable quote. Own entry trade-volume,
freshness, brackets and position constraints are unchanged. This is a declared
new hypothesis, not an independent validation of V1.

The extension produces61 paths:34 impulses,16 retests and eleven VWAP pullbacks.
There are59 known trade-price paths and two unknown stale trade references at
session expiry. Mark paths are all known, which does not prove those two market
exits were executable. Unknown paths are retained and make affected selection
cases unscorable.

The expanded trade-trigger union has26 selection proposals and21 admitted PAPER
signals in August/September. Twenty lack sufficient prior completed history;
the next has a nonpositive last20 PAPER R total. PAPER profitable-exit rates are
25% in August (eight closes) and38.46% in September (thirteen closes), with
negative price and R sums in both months. These are model signal diagnostics,
not client deposit returns. More signals did not establish an edge.

## Common-account validation and reporting

Publication rails include2/scan,4/hour,2/direction/hour,4/nominal direction,
4/economic direction, three-hour symbol/direction cooldown, two-loss streak and
daily2R pause. Cash-ready stock entries require the previous twenty completed
nominal50% PAPER outcomes to be positive in both price-return sum and R sum.
Private TP1 remains independently lot-rounded for requested5% and50%/minimum-lot.

The fixed crypto reference is the prior exploratory activity-qualified checked
BE-atTP1 proposal stream. Shared cash starts once with$130 in August, uses3%
client margin and10x, per-bot caps4/8 and common cap12/90% reserved margin, two
writer headroom and delayed margin release. Stocks remain absent from simulated
cash when PAPER is unready. This is not a newly deployed crypto strategy.

Selection compares base/expanded universe, trade/mark triggers and individual/
combined families across both TP1 modes and both bot ordering priorities. A
positive monthly contribution and twenty stock closes is only a research
prerequisite. The same is independently tested for crypto and both bots jointly;
none constitutes proof of the actual+30% monthly objective. No stock variant
qualifies, so October/November cash validation is not performed.

`stock_session_nov19_review.py` separates PAPER statistics from cash contribution
and explains readiness failures. It restricts cash tables to August/September:
the generic calendar helper's later zero rows in the original research files
are empty buckets, not evaluated holdout months. It verifies source cache hashes.

Neither source proves historical bid/ask fills, publication latency, floating
equity/current account guards, liquidation, fees/funding or private order PnL.
The latest TP1 minimum-lot production fix is documented separately.

## Reproduction

Run `stock_session_joint_nov19_audit.py`, then
`stock_session_extended_joint_nov19_audit.py`, then `stock_session_nov19_review.py`
with `python -X utf8 -u`. Each study checks its frozen declaration and input
fingerprints. Raw candles/pickle caches stay local; sources, declarations and
summary reports are committed. The sibling stock repository supplies its frozen
pure portfolio gate, identified by SHA256.

Sixteen focused tests cover official sessions/DST, causal non-vacuous opening
retests/impulses and VWAP reclaims, separate failure/acceptance, missing trade
activity, reference-only mark availability, inverse economic exposure caps,
full-horizon maturity, strictly completed PAPER history, and distinguishing a
stock-only prerequisite from joint objective achievement.
