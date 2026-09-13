# Turning the bank on: what changes, what to watch, what would make me turn it off again

Written while the bank is still off, so the decision is a decision and not a scramble. Nothing here
is done until the owner says so.

## What actually changes

One line of configuration:

    PULLBACK_LIVE_ENABLED   "0" -> "1"

That is all. The rules, the coin list, the regime gate, the take and stop, the 48h time exit and
the touch entry stay exactly as they were measured. `PULLBACK_GROUP_POSTS` is already 1, so signals
keep going to the group; `SMC_SIGNALS_ENABLED` stays 0, so the old logic stays silent.

The filter that was withdrawn on 12.09 is gone from the tree - there is nothing to switch on by
accident.

## What the owner picks

**Risk per trade.** The bot sizes each position from the owner's own setting. The numbers below
are not a simulation of the R stream - they are the actual historical path, compounding from $120,
with margin allowed to refuse a signal the account cannot fund (one position per coin, up to
sixteen at once, 10x).

| risk per trade | worst drawdown | days underwater | worst month | $120 after 4.3 years | first year |
|---|---|---|---|---|---|
| **1.5%** | **-16.4%** | 169 | -7.9% | $1,292 | $231 |
| 3% | -31.8% | 173 | -16.8% | $9,849 | $401 |
| 5% | -46.6% | 173 | -26.8% | $177,715 | $1,209 |

Three things that table does not say out loud.

**Margin only binds above 1.5%.** At 1.5% not one signal in 2,784 was ever refused, so that column
is the strategy exactly as it was measured. At 3% it refuses 73, at 5% it refuses 359 - and which
ones it refuses is decided by arrival order, not by quality.

**At 5% the result stops being about the edge.** The same period, the same rules: $773 in the first
year without the margin cap, $1,209 with it. A 56% difference produced by nothing but which trades
the account happened to be able to afford. That is not an argument that the cap helps; it is an
argument that at that size the outcome is decided by accident.

**Underwater for about 170 days at every setting.** Almost six months below a previous high is the
normal shape of this, not a rare episode. The difference between the settings is how deep the hole
is while you wait, and -46.6% from $120 is $64.

The bank is also flat 77.8% of the time and holds one position for another 11.5%; four or more
positions happen in 3.1% of hours, and all sixteen happened once, on 21.08.2026.

## Before flipping it

1. `python -m unittest discover -s tests -q` - 128 tests, all green as of 13.09.
2. Confirm the exchange has an X-Perp for each coin in `PULLBACK_SYMBOLS`. SEIUSDT and LABUSDT are
   in the list but not listed on OKX; the live module skips them silently (`if not inst: return`),
   so they cost nothing but never trade. The measured numbers are the other sixteen.
3. Check the OKX keys still work and the account is funded.

## What to watch in the first two weeks

The backtest says roughly 2 trades a day, 85% of them winners, one stop in eight.

* **Win rate below 75% after 40+ trades** - the live entry is not getting the price the model
  assumes. Compare the fills against the signal level; the bot already measures adverse slippage
  per coin and excludes a coin whose average exceeds `PULLBACK_LIVE_MAX_SLIP`.
* **Average slippage above 0.06%** - doubling slippage costs a third of the profit. That is the
  single most expensive thing that can go wrong, worth more than any strategy change tested in two
  nights.
* **Three losing months in a row** - never happened in five years of history (the longest run is
  two), so it would mean the edge has changed rather than that variance is doing its job.
* **Stops arriving in packs** is normal: 43% of them land within six hours of another, and that is
  what a losing month looks like. It is not a reason to intervene.

## What would justify turning it off

Not a losing week, and not a losing month - both are inside the measured distribution. The honest
triggers are: three consecutive losing months, a win rate under 75% over 100+ trades, or measured
slippage that stays above 0.06% after the per-coin exclusions have had a chance to work.

## One optional change, if the owner wants it

**Half size on BTC.** Measured on 13.09 and the only candidate of eleven to pass everything:

    ratio 0.32 -> 0.37, drawdown -11.1R -> -9.9R, better in four years of five,
    while halving any other coin (including AAVE, whose average trade is identical) does not
    reproduce it.

Two independent reasons: BTC takes part in 12.4% of the account's worst days against an 8.3%
average, and its execution book dips a quarter of a candle below the analysis feed in 15.8% of
bars against 1-4% for the rest. The same trim was already adopted in the signal bot in August for
its own symptoms, which is corroboration from a different engine.

It is a **position weight**, not a removal - BTC keeps trading, at half the money. There is no
config flag for per-coin weights today, so this needs a small addition to the live sizing path
(`_margin_for` is where a per-coin multiplier belongs), plus a test. Deliberately not written yet:
the bank is off, and nothing touches live code before the owner decides.

Without it the bank is exactly what every number in the reports describes. With it, expect roughly
a tenth less drawdown for the same profit.

## A second, smaller thing found in the venue data: BILL

Extending the venue comparison to all sixteen pinned coins turned up one that is not a strategy
problem but an execution one:

    BILLUSDT: basis +0.245% (ten times any other coin), and its X-Perp candle range is
    0.00x the analysis feed's - the instrument barely trades.

Every other coin sits within +-0.083% of basis and 0.66-1.00x of candle width.

In the backtest BILL is irrelevant - 18 trades in five years, and removing it moves the ratio from
0.32 to 0.33 (one year of five). But live, an order into an empty book pays slippage the model
never counted, and BILL already has the worst stop rate of any coin (28%).

**The bot already protects itself**: it measures adverse slippage per coin and excludes any coin
whose average exceeds `PULLBACK_LIVE_MAX_SLIP` for seven days. BILL would be excluded on its own
after the first few bad fills. Dropping it from `PULLBACK_SYMBOLS` up front just skips those first
few.

Combined with the BTC trim: ratio 0.37, average trade +0.0696R against +0.0684R for the BTC trim
alone - the same result, one less way to lose money to the order book.

## The honest numbers, priced from the live order book

Every figure above assumed a flat 0.03% of slippage per side. The execution venue's book was read
directly and priced for the order a $120 account actually sends (~$54 of notional at 1.5% risk):

    BTC, ETH        0.000%        SUI            0.014%        AAVE   0.077%
    ZEC             0.002%        TAO, NEAR      0.021%        XLM    0.106%
    XRP             0.004%        DOT            0.026%        BILL   0.386%
    SOL             0.005%        AVAX, LINK     0.027-0.029%
    HYPE            0.006%        ADA            0.033%

Average across all sixteen: 0.047% - half again the assumption. Without BILL: 0.025%, better than
assumed. BILL alone doubles the basket's execution cost, and 0.386% is nearly half the distance to
a 0.85% take, paid to the order book on entry.

Re-running the bank with each coin charged its own real cost:

| version | per trade | per month | drawdown | ratio |
|---|---|---|---|---|
| as reported (flat 0.03%) | +0.0670R | +3.6R | -11.1R | 0.32 |
| **real book costs** | **+0.0606R** | **+3.2R** | -11.2R | **0.29** |
| real, BILL dropped | +0.0625R | +3.3R | -11.2R | 0.30 |
| **real, BILL dropped + BTC at half** | **+0.0633R** | **+3.4R** | **-10.0R** | **0.34** |
| real, BILL and XLM dropped + BTC at half | +0.0670R | +3.3R | -10.1R | 0.33 |

**So the honest expectation is about a tenth below every profit figure in these reports** - at 1.5%
risk, roughly +4.8% a month rather than +5.4%. The two mechanical fixes more than pay that back:
+5.1% a month at a smaller drawdown.

One caveat on the measurement: the book was read at a single moment. Liquidity varies with the
hour and with volatility, so treat these as the shape of the cost rather than its exact value -
but the ranking of the coins, and BILL's position in it, will not change.
