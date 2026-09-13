# Turning the bank on: what changes, what to watch, what would make me turn it off again

> ## Where this stands, 13.09 — read this first
>
> This file grew as the work went, and two of its sections were overtaken by later measurements.
> Rather than delete them, they are marked and kept, because what was believed and why is part of
> the record. The current position is here; everything below is the working that led to it.
>
> **The proposal is two changes and one setting:**
>
> | | |
> |---|---|
> | take **1.0 ATR** instead of 0.75 | survived ten independent checks, including a run against real X-Perp candles |
> | drop **BILLUSDT** | measured order book: 0.386% to enter, 16 trades in five years, −4.5R |
> | risk **1.40%** | at 1.5% the drawdown reaches −14.3% against a latched 15% limit that never resumes |
>
> **What that is worth:** $120 becomes **$1,684** over 4.3 years against **$1,155** on today's
> settings — 46% more — at a drawdown of −13.4% against −13.1%, with 43 positive months of 52
> instead of 39 and a worst month of −6.6% instead of −7.1%.
>
> **Withdrawn during the work, do not act on:**
> - halving BTC — walk-forward fell to 4 of 9 once the take moved; BTC is second of fifteen, behind
>   AAVE. Sections "One optional change" and the retraction at the end.
> - shortening the BTC daily average to 32 — killed by the same walk-forward (in PARAMETER_AUDIT.md).
>
> **The edge, stated plainly:** over five years the bank returns **+0.0824R** a trade against
> **-0.0312R** for a random entry on the same bars with the same geometry and costs - a gap of
> **+0.1136R at 11.2 sigma**, steady in every year (+0.129, +0.131, +0.111, +0.090, +0.111).
>
> That also settles the win-rate question. A random entry already wins **74%** of the time, because
> a 1 ATR take against a 3 ATR stop is a bet that usually wins and still loses money. The bank wins
> 81%. Those seven points over free are the edge; the 88% available by widening the stop costs a
> third of the money. The number to ask for was never the win rate - it was the points above random.
>
> **Two things to hold in mind before reading the numbers:**
> - the backtest's 100% fill rate is an artefact; live fills ran 79% on real X-Perp bars, so expect
>   about 20% fewer trades than any projection here implies. The missed ones were below average.
> - the −13.4% drawdown happens *within* months, not between them. The chart will show that hole
>   while the month it falls in can still close positive.


Written while the bank is still off, so the decision is a decision and not a scramble. Nothing here
is done until the owner says so.

## What actually changes

Written first as one line - `PULLBACK_LIVE_ENABLED "0" -> "1"` - because at that point nothing else
was proposed. Two changes have since been accepted, so it is three:

    PULLBACK_LIVE_ENABLED   "0" -> "1"
    PULLBACK_RULES          -> a new set with the take at 1.0 ATR instead of 0.75
    PULLBACK_SYMBOLS        -> its own default, the 4h list minus BILLUSDT

and the owner's own risk setting at 1.40%. Exact edits are under "Exactly what changes, if the
answer is yes".

Everything else stays as measured: the rules themselves, the regime gate, the stop, the 48h time
exit, the touch entry. `PULLBACK_GROUP_POSTS` is already 1, so signals keep going to the group;
`SMC_SIGNALS_ENABLED` stays 0, so the old logic stays silent.

The filter that was withdrawn on 12.09 is gone from the tree - there is nothing to switch on by
accident. The BTC per-coin weight considered during this work was never written, and should not be.

## What the owner picks

**Risk per trade.** The bot sizes each position from the owner's own setting. The numbers below
are not a simulation of the R stream - they are the actual historical path, compounding from $120,
with margin allowed to refuse a signal the account cannot fund (one position per coin, up to
sixteen at once, 10x).

| risk | signals refused | worst drawdown | days underwater | worst month | $120 after 4.3y | first year |
|---|---|---|---|---|---|---|
| **1.5%** | **0 of 2,784** | -16.4% | 169 | -7.9% | $1,292 | $231 |
| 1.75% | 3 | -19.0% | 169 | -9.4% | $1,887 | $255 |
| **2%** | 9 (0.3%) | -21.6% | 169 | -10.9% | $2,706 | $280 |
| 2.5% | 32 (1.1%) | -26.8% | 173 | -14.4% | $5,238 | $332 |
| 3% | 73 (2.6%) | -31.8% | 173 | -16.8% | $9,849 | $401 |
| 4% | 198 (7.1%) | -42.0% | 173 | -24.1% | $38,082 | $665 |
| 5% | 359 (12.9%) | -46.6% | 173 | -26.8% | $177,715 | $1,209 |

Three things that table does not say out loud.

**The boundary of perfect funding is 1.55%.** Below it every one of the 2,784 signals is affordable
and the projection is the backtest exactly. Bisected on the same simulation; an account that
withdraws its profits instead of compounding gets 1.54%, essentially the same, because the crowding
that binds happens early while the deposit is still small.

**Above it the distortion is real but gradual, until it is not.** 2% refuses nine trades of 2,784,
which is nothing; 3% refuses seventy-three, which is a 2.6% distortion worth naming and not worth
panicking about. 5% refuses 359 - one signal in eight - and there the outcome stops being about the
edge: the same period and the same rules give $773 in the first year without the margin cap and
$1,209 with it, a 56% swing produced by nothing but which trades the account could afford.

**Underwater for about 170 days at every setting.** Almost six months below a previous high is the
normal shape of this, not a rare episode. What the risk setting buys is not a shorter wait - it is
a deeper or shallower hole while you wait, and -46.6% from $120 is $64.

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

> **SUPERSEDED 13.09 - do not act on this section.** Everything below was written under a
> 0.75 take and is now wrong in its central claim: the control at take 1.0 puts BTC second
> of fifteen, behind AAVE on both money and drawdown, and the walk-forward falls to four of
> nine. The trim is withdrawn - see "RETRACTED: halving BTC does not survive the take
> change" at the end of this file. Kept here as the record of what was believed and why.

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

## The risk setting and the drawdown latch are one decision, not two

Reading the live module against the backtest turned up three rules that exist only in
`src/pullback_live.py` and `config.py` and appear in none of the numbers above.

**Sizing is not equal-risk.** `pullback_live.py:254` shrinks the margin only when the stop is
*wider* than `PULLBACK_STOP_REF` (3.94%); a tighter stop keeps the full margin. Risk per trade is
therefore capped at the setting and falls proportionally for tight stops, while every report so far
assumed the same risk on every trade. Simulated on the same path from $120 at a 1.5% setting:

| sizing | $120 becomes | worst drawdown |
|---|---|---|
| equal risk (what the reports assume) | $1,292 | -16.4% |
| **the deployed rule** | **$1,185** | **-13.1%** |
| fixed margin with no shrink at all | $2,151 | -29.6% |

About 8% less money for 20% less drawdown - and that difference is not cosmetic, because it is what
keeps the account above the latch described next. With equal-risk sizing the live latch fires on
4 October 2023 and the account ends at $247.

**A latched drawdown pause at 15% of peak** (`PULLBACK_LIVE_MAX_DRAWDOWN`) and a 3% daily pause
(`PULLBACK_LIVE_MAX_DAILY_LOSS`). The daily one barely matters - ten skipped entries in 2,784. The
latched one decides everything, because it does not resume:

| risk setting | trades taken | entries lost to a pause | $120 becomes | latch fires |
|---|---|---|---|---|
| 1.50% | 2,774 | 10 | $1,154 | no |
| 1.75% | 2,769 | 15 | **$1,656** | no |
| **1.80%** | 1,623 | 1,161 | **$617** | **16.02.2025** |
| 2.00% | 810 | 1,974 | $320 | 04.10.2023 |
| 3.00% | 441 | 2,343 | $367 | 02.2023 |

The boundary is **1.779%**. Moving the setting from 1.75% to 1.80% - five hundredths of a
percentage point - costs two thirds of the money, not because the strategy got worse but because
the bank switches itself off and never comes back.

The two settings are therefore one decision:

| drawdown limit | at 1.5% risk | at 2% | at 3% |
|---|---|---|---|
| 15% (current) | $1,154 | $320, latched | $367, latched |
| 20% | $1,154 | $2,352 | $492, latched |
| 30% | $1,154 | $2,352 | $7,795 |

**This is not an argument for raising the limit.** The latch exists to stop a strategy that has
genuinely broken, and raising it means accepting a deeper real hole before anyone notices. It is an
argument that a risk setting above ~1.75% with the limit left at 15% is a decision to turn the bank
off in the first bad stretch - which is the worst of both, a large drawdown *and* no recovery.

**At 1.5% everything is consistent**: no signal is ever refused for margin, the daily pause costs
ten entries in four years, the latch never fires, and the drawdown lands at -13.1% against a 15%
limit. That is the only setting where the measured strategy and the deployed guards agree.

## The live universe is smaller than the backtest's, and that turns out to be fine

Two more gates exist only in the live path. `PULLBACK_LIVE_MAX_SPREAD` (0.05%) skips an entry whose
X-Perp spread is wider; `PULLBACK_LIVE_MAX_SLIP` (0.03%) excludes a coin for seven days once its
measured entry slippage averages above that. Neither is in any backtest number.

Measured on the venue's book (seven snapshots, 03-04 UTC - a thin hour, and the sampler now running
covers a full day, so treat this as the shape rather than the size):

| coin | median spread | snapshots above 0.05% | cost of a $54 order | above 0.03% |
|---|---|---|---|---|
| BILL | 0.453% | 100% | 0.327% | 100% |
| XLM | 0.195% | 100% | 0.106% | 100% |
| AAVE | 0.110% | 100% | 0.074% | 100% |
| NEAR | 0.085% | 57% | 0.043% | 57% |
| DOT | 0.040% | 0% | 0.039% | 71% |
| TAO / SUI / AVAX / ADA | 0.041-0.048% | 14-43% | 0.020-0.024% | 0-43% |
| LINK, HYPE, SOL, XRP, ZEC, ETH, BTC | 0.0001-0.026% | 0% | 0.0001-0.020% | 0% |

So the deployed bot would never enter BILL, XLM or AAVE at all, would miss over half of NEAR's
entries, and would periodically exclude NEAR and DOT for a week at a time. The backtest counts all
of them, and NEAR alone contributes +17.7R of the +168.6R total.

| universe | per trade | per month | drawdown | ratio |
|---|---|---|---|---|
| all sixteen | +0.0606R | +3.24R | -11.2R | 0.29 |
| the two proposed changes | +0.0633R | +3.37R | -10.0R | 0.34 |
| minus what the spread gate always blocks | +0.0724R | +3.29R | -10.0R | 0.33 |
| **the same, plus BTC at half** | **+0.0738R** | **+3.36R** | **-9.6R** | **0.35** |
| also minus NEAR | +0.0735R | +3.01R | -9.3R | 0.33 |
| also minus NEAR and DOT | +0.0764R | +2.85R | -8.4R | 0.34 |

**The bank survives its own safety gates**, and in the likeliest case is slightly better for them.
The reason is not luck: the coins the gates refuse are the ones already sitting at the break-even
line. XLM returns +2.1R across 177 trades and AAVE returns exactly nothing across 227, while the
edge lives in the liquid names.

The worst case - losing NEAR and DOT to the seven-day exclusions as well - costs about 15% of the
monthly R and buys a drawdown of -8.4R instead of -9.6R. Nothing here changes the decision; it
removes a way for the live result to diverge from the measured one without anyone noticing.

## The high win-rate requirement is not costing anything

The bank was chosen partly because it wins 85% of its trades. The engine it rides inside - the 4h
breakout in `src/trend4h.py` - wins about a third, and was set aside for that reason without the
two ever being put side by side under the same costs and the same account.

| engine | win rate | per trade | total R | R per month | drawdown | ratio |
|---|---|---|---|---|---|---|
| the bank | 85% | +0.061R | +168.6R | +3.24R | -11.2R | **0.29** |
| trend4h | 37% | +0.324R | **+689.6R** | **+13.52R** | -54.9R | 0.25 |
| both together | 64% | - | +858.2R | +16.50R | -61.4R | 0.27 |

trend4h makes four times the R. It also loses five times as much in its worst stretch, and there is
only one account. Sized so that each reaches the same 13% drawdown - the most the live latch
tolerates:

    the bank    risk 1.16% per trade  ->  +3.76% a month
    trend4h     risk 0.24% per trade  ->  +3.20% a month
    both        risk 0.21% per trade  ->  +3.49% a month

**At equal drawdown the bank earns more.** trend4h's larger R total is position size, not profit:
surviving its drawdown on a $120 account means cutting the risk fivefold, and the advantage
disappears. Running both does not help either - the bank trades inside trend4h's positions, so they
are correlated, and the pair lands between them at 0.27.

So the win-rate preference costs nothing here. It happens to coincide with the better engine on the
measure that decides how much an account can actually deploy. Worth knowing in the other direction
too: if the latch were raised and a much deeper hole accepted, trend4h would be the way to put more
money to work - it holds positions 81.8% of the time against the bank's 22.2%.

## Exactly what changes, if the answer is yes

Superseded once already - the BTC trim was in this list and has been withdrawn (see "RETRACTED:
halving BTC" below) - so this is the current version. Nothing here is applied; the bank is off until
the owner says otherwise.

**1. The take, 0.75 -> 1.0 ATR.** In `src/pullback_bank.py`, add a new named set rather than editing
the existing one, so every number ever reported under the old name stays reproducible:

    RULE_SETS["strict3+short2_wide10"] = [dict(r, tp=1.0) for r in RULES_STRICT + SHORT_RULES]

and point `PULLBACK_RULES` at it in `config.py`. The old `strict3+short2_wide` stays untouched.

**2. BILLUSDT out.** `PULLBACK_SYMBOLS` currently defaults to `TREND4H_SYMBOLS`. Give it its own
default - the same list minus BILLUSDT - so the 4h engine's universe is not affected. Code default,
no environment variable.

**3. `PULLBACK_LIVE_ENABLED` 0 -> 1**, with the risk setting at **1.40%** rather than 1.5%. Not
caution for its own sake: at 1.5% the take change puts the drawdown at -14.3% against a latched 15%
limit - seven tenths of a point of margin, and the latch does not resume. 1.40% gives -13.4%.

**No per-coin weight is needed.** An earlier version of this plan added a `coin_weight` lookup to
`src/pullback_live.py` for the BTC trim. That change is withdrawn and should not be written.

Then `python -m unittest discover -s tests -q` (128 tests, green as of this writing), commit, push.
The push triggers a Railway redeploy and a restart; it enables nothing by itself.

## What the next four years would have felt like, month by month

Summaries are what people agree to; months are what they live through, and a working system gets
switched off in the middle of a stretch the summary already promised. The full package - BILL
dropped, BTC at half, take 1.0 - on $120 at 1.5% risk with the live guards, read one month at a
time:

    52 months, 42 positive (81%), 10 negative
    longest run of losing months: ONE - two in a row never happened
    worst month: July 2024, -6.9% ($468 -> $436)
    best month: August 2026, +44.7%
    median month +4.0%; a quarter of months below +0.8%, a quarter above +8.4%
    longest wait for a new high: 4 months

**Two consecutive losing months has never happened in 4.3 years.** That sharpens the off-switch
rule: not "three losing months" but "a second losing month in a row is outside everything measured -
look immediately".

And a distinction worth holding onto before it is needed: the -13.3% drawdown is **within** months,
not between them. The account will show that hole on its own chart while the month it happens in can
still close positive. The bottom of a month and the end of a month are different numbers, and only
the second one is evidence about the strategy.

The ten losing months are the cost of the forty-two. Nothing in that list is a reason to intervene;
the triggers that are remain the ones above - a second consecutive losing month, a win rate under
75% over 100+ trades, or measured slippage staying above 0.06% after the per-coin exclusions have
had their chance.

## Measured on the venue it actually trades

Every number in these reports comes from OKX's global USDT swap. The orders go to the EU X-Perp, a
different instrument with its own prints. There are 18,000 real X-Perp 15m bars cached for six coins
- BTC, ETH, XRP, SOL, ADA, AVAX, from 27.05 to 26.08.2026 - so the bank can be run the way it
actually runs: signals and levels from the analysis feed, fills against the X-Perp's own highs and
lows.

| | signals | filled | take/stop/time | win rate | per trade | total |
|---|---|---|---|---|---|---|
| as the backtest does it | 96 | 96 (**100%**) | 83 / 11 / 2 | 86.5% | +0.1496R | +14.4R |
| **against real X-Perp bars** | 104 | 82 (**79%**) | 73 / 8 / 1 | **89.0%** | **+0.1814R** | +14.9R |

**The backtest's 100% fill rate is an artefact.** The entry level is the analysis feed's own hourly
close, so in that feed price starts exactly on it and the touch is guaranteed. On the X-Perp it is a
foreign price that has to be reached - and one signal in five is not.

So expect roughly 20% fewer trades live than the projections imply. The ones that do fill are
better: 89.0% against 86.5%, +0.1814R against +0.1496R, and the total over the window is the same
within noise. The miss behaves as a filter - if price never comes back to the level on the venue
where the money is, the move has already gone, and that trade was going to be below average.

Caveats stated plainly: six coins, three months, about a hundred trades. Enough to establish the
direction and the mechanism, not the size. But this is the first time the bank has been measured on
the exchange it will actually trade, and the answer is that the venue gap costs trades rather than
money.

### The missing fifth is a filter, not a loss

The obvious fix for a 79% fill rate is to take the level from the venue that fills it - the X-Perp's
own hourly close instead of the analysis feed's. The live bot already watches that price every
second, so it would cost nothing to implement. Same window, same signals:

| level from | filled on | fill rate | win rate | per trade | total |
|---|---|---|---|---|---|
| analysis feed | analysis feed | 100% | 86.5% | +0.1496R | +14.4R |
| analysis feed | X-Perp (what happens live) | 79% | **89.0%** | **+0.1814R** | **+14.9R** |
| X-Perp | X-Perp (the fix) | 92% | 86.8% | +0.1551R | +14.1R |

The fix does what it promises - 79% to 92% - and the recovered trades are worse than the ones
already being taken: +0.1551R against +0.1814R, and a smaller total from more trades.

So the miss is selection, not loss. If price never returns to the level on the exchange where the
money actually is, the move has gone, and that entry was going to be below average. The live bot is
already doing the right thing, for a different reason than assumed.

With a hundred trades over three months this is not proof the fix would hurt; it is the absence of
any reason to make it, against a real cost in live-code complexity. Rejected - the twenty-third
candidate.

### Partial retraction: the venue argument for the BTC trim is not confirmed

Halving BTC was argued on two grounds. First, BTC takes part in 12.4% of the account's worst days
against an 8.3% average, better in four years of five and seven forward half-years of nine, with
controls on five other coins failing to reproduce it. Second, its X-Perp feed dips a quarter of a
candle below the analysis feed in 15.8% of bars against 1-4% elsewhere - a rougher venue, therefore
more stops.

The second argument was a property of the candles, not of the trades. With real X-Perp bars the
question can be asked directly - do the same trades stop more often on the venue?

| coin | analysis: trades / stops | X-Perp: trades / stops | change |
|---|---|---|---|
| BTC | 17 / **0** (0.0%) | 13 / **0** (0.0%) | +0.0 pp |
| ETH | 15 / 3 (20.0%) | 13 / 2 (15.4%) | -4.6 pp |
| XRP | 13 / 1 (7.7%) | 11 / 0 (0.0%) | -7.7 pp |
| SOL | 19 / 1 (5.3%) | 15 / 0 (0.0%) | -5.3 pp |
| ADA | 18 / 3 (16.7%) | 18 / 3 (16.7%) | +0.0 pp |
| AVAX | 14 / 3 (21.4%) | 12 / 3 (25.0%) | +3.6 pp |
| **all** | 96 / 11 (11.5%) | 82 / 8 (**9.8%**) | **-1.7 pp** |

Two awkward facts. BTC takes zero stops on either venue in this window - 17 and 13 trades without
one - so the mechanism cannot be tested on the coin it was invented for. And across all six coins
the X-Perp produces **fewer** stops, not more.

**So the venue half of the case for halving BTC is withdrawn.** A wider candle is not the same thing
as a stop that fires, and when the claim was finally put to trades rather than to candles it did not
reproduce.

The trim itself stands, on the evidence that never depended on this: drawdown participation, four
years of five, seven forward half-years of nine, and controls that fail to reproduce it on other
coins. It is now a one-legged argument rather than a two-legged one, which is worth knowing when
deciding whether to take it.

Three months and ninety-odd trades cannot refute the venue claim either - but it was asserted on
candle geometry and has now failed its first direct test, which is the honest status to record.

## RETRACTED: halving BTC does not survive the take change

The trim's venue argument was withdrawn earlier today. What remained was its share of the account's
worst days and a walk-forward of seven forward half-years of nine - but that walk-forward was run
under a 0.75 take, and the take has moved.

Re-run at take 1.0, BILL dropped, real book costs:

| forward half-year | base | BTC halved | AAVE halved |
|---|---|---|---|
| 22-05..22-11 | +4.38 | +4.45 | +4.54 |
| 22-11..23-05 | +4.16 | +3.86 | +4.54 |
| 23-05..23-11 | +4.23 | +4.30 | +4.33 |
| 23-11..24-05 | +3.62 | +3.58 | +3.66 |
| 24-05..24-11 | +2.92 | +2.82 | +3.42 |
| 24-11..25-05 | +4.68 | +4.77 | +4.63 |
| 25-05..25-11 | +3.76 | +3.87 | +3.88 |
| 25-11..26-05 | +0.92 | +0.89 | +1.07 |
| 26-05..26-11 | +8.53 | +8.28 | +8.89 |
| **better than base** | - | **4 of 9** | 8 of 9 |

Four of nine is a coin flip. The control had already put BTC second of fifteen rather than unique,
behind AAVE on both money and drawdown. **The trim is withdrawn.**

Keeping it because it earns more would be exactly what twenty-three other candidates were refused
for. And AAVE is not adopted in its place: eight of nine looks strong until you remember it is the
winner of a fresh in-sample search over fifteen coins, where the best of fifteen will usually manage
eight of nine by luck. It is written down as a lead for future data, not taken now.

### What the package is now

| change | rests on | status |
|---|---|---|
| take 1.0 ATR | ten independent checks, including real X-Perp bars | **keep** |
| drop BILLUSDT | measured order book: 0.386% to enter, 16 trades in five years | **keep** |
| ~~BTC at half~~ | walk-forward 4 of 9 at the new take | **withdrawn** |

Without the trim the drawdown at 1.5% risk is -14.3%, which is seven tenths of a point from a latch
that never resumes. That is too close, and the honest instrument for it is the risk setting rather
than a coin weight dressed as a finding:

| setting | $120 becomes | drawdown | latch |
|---|---|---|---|
| take 1.0, no BILL, 1.50% risk | $2,015 | **-14.3%** | no, but close |
| **take 1.0, no BILL, 1.40% risk** | **$1,684** | **-13.4%** | no |
| take 1.0, no BILL, 1.35% risk | $1,539 | -12.9% | no |
| as deployed today, 1.50% | $1,155 | -13.1% | no |

**The recommendation is take 1.0, BILL dropped, risk 1.40%**: $1,684 against today's $1,155 - 46%
more money at the same drawdown - with every component validated rather than merely favourable.

## The current proposal against the current settings, month by month

The earlier month-by-month section was computed for a three-change package at 1.5% risk. That
package no longer exists - the BTC trim is withdrawn and the risk moved to 1.40% - so those figures
describe something that is not on the table. Recomputed for what is:

| | as deployed | proposed |
|---|---|---|
| settings | take 0.75, all coins, 1.5% | take 1.0, BILL dropped, 1.40% |
| $120 after 4.3 years | $1,155 | **$1,684** |
| worst drawdown | -13.1% | -13.4% |
| positive months | 39 of 52 | **43 of 52** |
| worst month | -7.1% | **-6.6%** |
| longest run of losing months | **2** | **1** |
| longest wait for a new high | 5 months | **4 months** |
| median month | +3.6% | +3.6% |
| months where it is better | - | **36 of 52** |

Better on almost every axis at once: more money, more positive months, a shallower worst month, a
shorter losing streak and a shorter wait for a new high. The single thing that is worse is the
drawdown, by three tenths of a point.

**A correction to an earlier claim.** "Two consecutive losing months has never happened" was written
about the three-change package. The **deployed** settings have had such a run - two months - and the
proposal has one. So the off-switch rule stands as written for the proposal, but it was never true
of what is running today.

The median month is identical at +3.6% in both. The gain is not a better typical month; it is fewer
bad ones and more room in the good ones.

## Do not pre-emptively drop expensive coins - and watch the slippage threshold

Twenty-three book snapshots per coin (hours 03-07 UTC) put six of the sixteen above the bot's own
`PULLBACK_LIVE_MAX_SLIP` of 0.03% by median cost of a $54 order:

    BILL 0.290%   XLM 0.112%   AAVE 0.073%   NEAR 0.043%   TAO 0.042%   DOT 0.037%
    -- below the line --
    SUI 0.025%  ADA 0.024%  LINK 0.020%  AVAX 0.014%  HYPE 0.008%  SOL 0.005%
    ZEC 0.004%  XRP 0.004%  ETH 0.0002%  BTC 0.0001%

The obvious move - drop them the way BILL is being dropped - is wrong. At take 1.0 through the live
guards, each coin charged its own measured cost:

| universe | $120 at 1.40% risk | drawdown | trades |
|---|---|---|---|
| all sixteen | $1,595 | -13.6% | 2,533 |
| **BILL dropped (proposed)** | **$1,672** | -13.5% | 2,517 |
| only the ten below the threshold | **$1,014** | -13.6% | 1,679 |

At the same drawdown the cheap ten return 40% less money. The five expensive-but-not-absurd coins
earn more than their cost consumes. BILL is the exception because it is an order of magnitude worse
than the next one - 0.290% against 0.112%.

**The warning that follows is more important than the finding.** The live threshold sits *below* the
measured cost of six coins. If realised slippage tracks the cost of walking the book, the bot will
repeatedly exclude NEAR, TAO and DOT for seven days at a time and give up a large part of the
result - not because those coins are bad, but because the threshold is set under the venue's own
price for them.

Stated carefully: the gate measures something different from this table. It compares the fill
against the best price seen just before the order, while these numbers are the cost of walking the
book for the whole size. Related, not equal. So this is not a change - it is the first thing to
check once real fills exist. If per-coin measured slippage comes in near these figures,
`PULLBACK_LIVE_MAX_SLIP` is too strict and should be raised rather than allowed to shrink the
universe by 40% of its earnings.

### Interim: how much the book cost moves between hours

The first order-book pricing was a single snapshot, with the caveat that "the ranking of the coins
will not change" but the level might. A sampler has since been running; five hours of twenty-four
are covered so far (03-07 UTC), twenty-three snapshots per coin.

| coin | 04h | 05h | 06h | 07h | spread across hours |
|---|---|---|---|---|---|
| BILL | 0.2893% | 0.2898% | 0.2551% | 0.2926% | +13% |
| XLM | 0.1071% | 0.1186% | 0.1123% | 0.1389% | +28% |
| AAVE | 0.0697% | 0.0779% | 0.0641% | 0.0627% | +23% |
| **DOT** | 0.0522% | 0.0424% | 0.0275% | 0.0245% | **+79%** |
| NEAR | 0.0319% | 0.0319% | 0.0323% | 0.0432% | +35% |
| **TAO** | 0.0318% | 0.0422% | 0.0211% | 0.0422% | **+57%** |
| ADA | 0.0242% | 0.0241% | 0.0241% | 0.0241% | +0% |
| BTC, ETH | 0.0001% | 0.0001% | 0.0001% | 0.0001% | +0% |

**The ranking is stable** - BILL, then XLM, then AAVE, then DOT/NEAR/TAO, then the rest, in every
hour measured. The caveat on the original snapshot holds up.

**The level moves about 37%**, and that matters for exactly the coins sitting on the 0.03%
threshold. DOT runs 0.0245-0.0522% and TAO 0.0211-0.0422%: in some hours the bot will accept them
and in others its own rule will exclude them for a week. That is the mechanism behind the warning
above, now with numbers.

Large relative swings on the cheap coins are misleading - ZEC moves +126% between hours, from
0.0004% to 0.0069%, which is nothing in money.

The bank's evening block (21-00 UTC), where a third of its entries happen, is not covered yet.

### The proposed lines verified against what was measured

Checked programmatically rather than by eye, because a plan that looks right and builds something
slightly different is how a measured result turns into an unmeasured one:

    [dict(r, tp=1.0) for r in RULES_STRICT + SHORT_RULES]
    == [dict(r, tp=1.0) for r in RULE_SETS["strict3+short2_wide"]]     -> True

Five rules, all at take 1.00 and stop 3.0, conditions untouched:

    btc_pump_evening        LONG   btc24 >= 0.04896   hour in (20,23)
    coin_run_evening        LONG   ret24 >= 3.864     hour in (20,23)
    btc_pump_night          LONG   btc24 >= 0.04896   hour in (0,3)
    short_btc_up_morning    SHORT  btc24 >= 0.01701   hour in (8,11)
    short_pop_in_downtrend  SHORT  rsi2 >= 71.02      rsi14 <= 35.22

And the symbol change: 18 to 17, removing exactly BILLUSDT, with `TREND4H_SYMBOLS` still at 18 so
the 4h engine's universe is untouched.

This matters because of a failure already on record here: a parity harness passing 300 of 300 while
the rule it was testing never fired in the real run, because the harness was fed something the live
path never sees. The check above is on the exact expressions that would be written.

### The tests are safe for this change - and do not cover the deployed rule set

Checked before applying anything, because a change that breaks the suite at application time is a
change made in a hurry.

**Safe.** Nothing in `tests/` hard-codes the take, the rule-set name, or BILLUSDT. The three bank
tests build their own sets - `bank9`, `bank9+short2`, `strict3+short2` - so changing the
`PULLBACK_RULES` default and the symbol list cannot break them.

**But that is also the gap.** None of those is the set actually deployed (`strict3+short2_wide`),
and the proposed `strict3+short2_wide10` would be equally uncovered. The 128 green tests exercise
the bank's machinery on rule sets it does not trade.

That is not a reason to delay - the machinery is what tests should cover, and it is covered - but
the missing test is worth naming:

    the deployed set has five rules, all take 1.0 ATR and stop 3.0 ATR after the change;
    a test that asserts that, and that the set is exactly RULES_STRICT + SHORT_RULES with the
    take replaced, would catch a future edit that changes the deployed geometry by accident.

Not written, like everything else here, until the owner decides.

## The bank against a random entry, on the venue's own candles

The signals that never fill were tested for a trade of their own - enter at market instead of
waiting for the pullback. Both directions came back profitable, which is the signature of a missing
control, so one was built: random hours on the same coins over the same window, same geometry, same
costs.

| | trades | win rate | per trade |
|---|---|---|---|
| random entry, random side | 2,350 | 72.5% | **-0.0424R** |
| random entry, always long | 2,347 | 72.3% | -0.0522R |
| random entry, always short | 2,354 | 75.7% | +0.0057R |
| **the bank (filled signals)** | 82 | **89.0%** | **+0.1814R** |
| missed signals, market entry same side | 21 | 95.2% | +0.2496R |
| missed signals, market entry opposite | 20 | 85.0% | +0.1118R |

**A hypothesis of mine, refuted.** I expected a 1 ATR take against a 3 ATR stop to make any entry
look profitable - a one-ATR move in some direction being near certain over 48 hours. It does not:
random entry returns -0.0424R at a 72.5% win rate, because the costs eat it. The harness is sound
and the geometry is not doing the work.

**Which gives the cleanest statement of the edge produced so far.** The bank returns +0.1814R a
trade where a random entry on the same candles returns -0.0424R - a gap of 0.22R per trade,
measured on the exchange where the orders actually go, with no normalisation or modelling in
between.

**The missed signals are not adopted.** +0.2496R is about two sigma above random on twenty-one
trades, and the opposite direction is above random too, which points at the period rather than at
an edge. Written down as a lead for when more X-Perp history exists, not taken.
