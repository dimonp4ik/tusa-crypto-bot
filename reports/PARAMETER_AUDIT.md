# Every fitted number in the bank, re-picked blind

The bank was validated as a whole: the mirror loses, the random side loses the cost, the bootstrap
never goes negative, the null model never reaches its ratio. What had never been done was to audit
the numbers *inside* it. There are thirteen of them, all produced by the same search on 2022-2024,
and a search that produces a four-decimal threshold is exactly the process that produces a number
which means nothing afterwards.

Method throughout: score the candidate on 2022-2024, pick the winner **there**, then read 2025-2026
once, as a consequence and never as a criterion. Real per-coin order-book costs, not the flat 0.03%
assumption. The acceptance bar is the one the BTC trim had to clear: better in at least four years
of five, and not dependent on which coins it is measured on.

## The four mined hour windows: all four hold

Three rules fire only inside a window of the day, and those windows were mined. Each was re-picked
against all 24 possible windows of the same width, every candidate executed in full.

| rule | window | rank on 2022-24 | rank on 2025-26 | window vs no window, on 2025-26 |
|---|---|---|---|---|
| btc_pump_evening | 20-23 | 1 of 24 | 7 of 24 | +0.1691R vs +0.1014R |
| coin_run_evening | 20-23 | 2 of 24 | **1 of 24** | +0.0702R vs +0.0173R |
| btc_pump_night | 0-3 | 7 of 24 | 6 of 24 | +0.1768R vs +0.1014R |
| short_btc_up_morning | 8-11 | 1 of 24 | 10 of 24 | +0.0356R vs +0.0067R |

Every window beats having no window at all on the years that chose nothing. The short morning rule
degrades most - from first to tenth - which matches what its own comment already says about being
weaker out of sample, and it stays positive.

One process note, because it nearly produced a false alarm. The first attempt stripped the hour
condition, ran the rule, then bucketed the resulting trades by hour. That is a different experiment:
without the window the rule fires all day, and one-position-per-coin means an 03:00 trade can hold
the coin when the 09:00 signal arrives, so the trades sitting in the 8-11 bucket are not the trades
the boxed rule takes. On that reading the short rule looked broken (-0.0966R in 2026). Generating
the signals once and executing each window separately is the correct comparison, and it says the
rule is fine. **A subset of a different run is not the same as a run of that subset.**

An observation deliberately not acted on: for the short rule, hours 22-02 are good on both windows
(+0.047/+0.056 fitting, +0.090/+0.085 exam) and were never chosen. Adopting them now would be
fitting to the exam. It is written down here so that future data can test it.

## The six thresholds: all six hold, and two attempts to improve them failed

Each condition was dropped from its rule, the feature's value recorded at every signal, and the
condition re-imposed at 25 thresholds spanning the observed range.

| rule | condition | rank on 2022-24 | rank on 2025-26 |
|---|---|---|---|
| btc_pump_evening | btc24 >= 0.04896 | 3 of 26 | 2 of 25 |
| coin_run_evening | ret24 >= 3.864 | 7 of 26 | 5 of 26 |
| btc_pump_night | btc24 >= 0.04896 | 2 of 26 | 2 of 25 |
| short_btc_up_morning | btc24 >= 0.01701 | 1 of 26 | 5 of 26 |
| short_pop_in_downtrend | rsi2 >= 71.02 | 3 of 26 | 2 of 26 |
| short_pop_in_downtrend | rsi14 <= 35.22 | 2 of 26 | 8 of 26 |

None is a spike. Two are monotone in both windows - stricter is better per trade, all the way out -
which raises the obvious question of whether the account would prefer a stricter cut. It would not:

| change | ratio on 2022-24 | ratio on 2025-26 | base on 2025-26 |
|---|---|---|---|
| ret24 3.864 -> 4.8 | 0.34 -> **0.38** | **0.26** | 0.28 |
| rsi2 71.02 -> 65.0 | 0.34 -> **0.38** | **0.19** | 0.28 |

Both looked better where they were chosen and were worse where they were not. For the other four
conditions the best value on the fitting window **is the value already in the code**. The deployed
thresholds are better than what a fresh fit would pick.

## Rule order: not a degree of freedom at all

The bank takes the first rule that matches, so the order of the five decides which one claims an
hour where two agree - a choice nobody ever recorded as a choice. All twelve permutations produce
an identical result: ratio 0.34, +0.0689R, drawdown -9.9R, 1553 trades, to the last digit. The
reason is that `strict3+short2_wide` gives every rule the same geometry (take 0.75 ATR, stop 3 ATR),
so which rule claims an hour cannot change what the trade does. One hidden knob turned out not to
exist.

## Time exit: 48h confirmed

| hold | ratio on 2022-24 |
|---|---|
| 12h | **0.37** |
| 24h | 0.28 |
| 36h | 0.28 |
| **48h (deployed)** | 0.34 |
| 60h | 0.32 |
| 72h | 0.33 |

The fitting window prefers 12 hours. On 2025-2026 that choice collapses to a 0.10 ratio against the
base's 0.28, with the win rate down to 76% and the drawdown out to -15.8R. 48h stands.

## The one change that survives: the BTC daily average, 50 -> 32

The regime filter blocks longs while BTC's last daily close sits under its own average. The length
is 50 and was never re-picked. On a grid of every length from 22 to 60:

    SMA25-26  0.17      SMA30-32  0.33  <- plateau
    SMA27     0.21      SMA33-35  0.32
    SMA28     0.26      SMA36     0.31
    SMA29     0.27      SMA37-39  0.29      SMA50 (deployed)  0.29

Six consecutive lengths (30-35) sit at 0.32-0.33 with the same -11.2R drawdown. That is a plateau,
not a spike; the deployed 50 sits past its far shoulder. The coarse first grid jumped from 26 to 30
and looked like a cliff - the intermediate lengths ramp smoothly, and the apparent cliff was an
artefact of the step size.

The collapse below 30 has a mechanism rather than being noise. SMA25-27's worst drawdown is
**January-March 2025**, a different episode from the one SMA30+ suffers (December 2025 - January
2026). Those short averages are too fast to keep longs out of the falling BTC of spring 2025 -
which is precisely what the filter exists to do, as its own comment says.

**The proposal is 32, the middle of the plateau, not 30, its best cell.** Taking the argmax of a
search is how an honest surface produces a fitted number. The centre costs a little measured
significance and cannot be the peak of noise.

SMA32 against SMA50, all sixteen coins:

| | 2022 | 2023 | 2024 | 2025 | 2026 | per trade | per month | drawdown | ratio |
|---|---|---|---|---|---|---|---|---|---|
| SMA50 | 0.64 | 0.35 | 0.32 | 0.24 | 0.37 | +0.0606R | +3.24R | -11.2R | 0.29 |
| **SMA32** | 1.43 | 0.40 | 0.33 | 0.27 | 0.43 | **+0.0675R** | **+3.67R** | -11.2R | **0.33** |

Five years of five. Better on both halves of the coin list (0.17 vs 0.15, and 0.15 vs 0.13). Where
the two disagree: 106 trades only SMA32 takes, averaging +0.1161R at a 91% win rate, against 127
only SMA50 takes, averaging -0.0343R at 77%. Bootstrapped by month so loss clustering survives, the
difference is at or below zero in 3.4% of draws; **with 2022 removed entirely, 4.8%** - marginal,
and reported as marginal.

The filter also earns its keep, which had never been checked either: without it the ratio is 0.17
at a -18.8R drawdown, against 0.33 at -11.2R with it.

## All three changes together

Adding SMA32 to the two execution fixes already proposed (drop BILL, BTC at half weight), priced at
real per-coin book costs:

| version | 2022 | 2023 | 2024 | 2025 | 2026 | per trade | per month | drawdown | ratio |
|---|---|---|---|---|---|---|---|---|---|
| as deployed | 0.64 | 0.35 | 0.32 | 0.24 | 0.37 | +0.0606R | +3.24R | -11.2R | 0.29 |
| BILL dropped, BTC half | 0.67 | 0.34 | 0.36 | 0.26 | 0.53 | +0.0633R | +3.37R | -10.0R | 0.34 |
| **all three** | 1.87 | 0.39 | 0.37 | 0.28 | 0.53 | **+0.0700R** | **+3.78R** | **-10.0R** | **0.38** |

Better in every one of the five years, and the three do not fight each other even though two of
them act on BTC. At 1.5% risk that is roughly **+5.7% a month** rather than +4.8%, at a tenth less
drawdown.

## What this audit changes about confidence in the bank

Thirteen fitted numbers, re-picked blind: twelve stand as they are, one moves. Two attempts to
improve thresholds and one to shorten the hold were rejected by the years that chose nothing -
bringing the night's count to fourteen rejected candidates against three accepted, all three
mechanical rather than strategic.

That ratio is the point. The eleven strategy ideas that died earlier died because they were fitted.
These thirteen numbers were produced by the same kind of search, and they did **not** die. The
bank's edge does not rest on its constants being lucky.

## The unpinned coins say nothing either way

SMA32 was chosen on the sixteen pinned coins. Twenty-two others sit in the same data set - DOGE,
BNB, PEPE, ARB, OP, BCH, LTC, SHIB and the rest - and contributed nothing to the plateau, the
bootstrap or the yearly table, which makes them the cleanest available test.

| coins | 2024 | 2025 | 2026 | per trade | per month | drawdown | ratio |
|---|---|---|---|---|---|---|---|
| unpinned 22, SMA50 | 0.06 | 0.01 | 0.40 | +0.0359R | +2.14R | -21.4R | 0.10 |
| unpinned 22, SMA32 | 0.05 | 0.05 | 0.34 | +0.0375R | +2.22R | **-18.0R** | 0.12 |

Every aggregate measure favours SMA32, including a drawdown 3.4R smaller. But the yearly count is
one of three, the two variants disagree on only 25 and 29 trades worth +1.4R between them, and
these coins have history only from 2024. The test has no power to speak with; it is reported as
neutral, not as support.

It is also a reminder that the unpinned half is simply a worse place to trade - ratio 0.10-0.12
against 0.29-0.33 for the pinned sixteen, which is the same gap the live universe showed in
September.

**So SMA32 is the weakest of the three proposed changes.** Dropping BILL and halving BTC rest on
measured execution facts. SMA32 rests on a plateau, five years of five, both halves of the pinned
list, and a bootstrap that is marginal once 2022 is removed. It is worth proposing and worth taking
last.

## RETRACTED: SMA32 does not survive walking forward

The evidence above was built one way: evaluate the whole history, split once, compare. That answers
"was 32 better than 50 over these five years". It does not answer the question that actually
matters - would someone standing at an arbitrary date, seeing only the past, have arrived anywhere
near 32, and would using it have helped from there on.

Walked forward with a 730-day trailing window and a 182-day step:

| applied to | chosen on trailing data | ratio of the choice | SMA50 | SMA32 |
|---|---|---|---|---|
| 24-05 .. 24-11 | SMA32 | 0.34 | 0.34 | 0.34 |
| 24-11 .. 25-05 | SMA22 | 0.02 | **0.40** | 0.33 |
| 25-05 .. 25-11 | SMA60 | 0.32 | 0.32 | **0.61** |
| 25-11 .. 26-05 | SMA60 | 0.01 | 0.05 | 0.05 |

    R per month going forward:  rolling choice +1.50 | SMA50 +2.47 | SMA32 +2.55
    fixed SMA32 beats SMA50 in 1 window of 4

Three things at once. The optimum does not stay put - the picks are 32, 22, 60, 60, a median of 46
across a 22-60 range, so there is no stable best length to find. Re-choosing actively destroys
money: +1.50R a month against +2.47R for leaving the parameter alone. And fixed SMA32 wins one
window of four, against five years of five on the full history.

That contradiction resolves badly for the change. The forward windows cover 2024-2026, and SMA32's
advantage is concentrated in 2022-2023 - 2022 alone was 42% of the total gain. **The advantage lives
in precisely the years that have been ruled out as a basis for decisions, and is absent from the
recent data.**

SMA32 is withdrawn. It is the fifteenth rejected candidate, and the only one that got as far as
being written up as a proposal before a further test killed it - which is the argument for running
the further test before proposing rather than after.

**Two changes remain, both resting on measured execution facts rather than on a search:** drop
BILLUSDT, and halve BTC's position size. Those give ratio 0.29 -> 0.34, +3.37R a month, drawdown
-11.2R -> -10.0R; roughly +5.1% a month at 1.5% risk.

A note on method for the next parameter that looks good: a plateau, a win in every year and a
monthly bootstrap were **not enough**. Only the walk-forward distinguished "better over this
history" from "better if you had deployed it". It is cheap and should come first, not last.
