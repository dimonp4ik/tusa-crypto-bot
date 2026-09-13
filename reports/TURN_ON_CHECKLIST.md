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

**Risk per trade.** The bot sizes each position from the user's own setting (percent of deposit or
a fixed dollar margin). From $120, compounding, 5000 paths with the clustering intact:

| risk per trade | median after 12 months | worst 5% | average drawdown |
|---|---|---|---|
| 1.5% | $224 | $147 | -13% |
| 3% | $401 | $177 | -25% |
| 5% | $863 | $215 | -39% |

Margin matters at this size: the bank holds up to 17 positions at once, each needing roughly
`risk / stop%` of notional at 10x. At 1.5% risk that is comfortable; above 3% the deposit starts to
limit how many signals can actually be taken, which quietly changes the strategy.

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
