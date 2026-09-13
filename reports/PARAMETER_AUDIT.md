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

## The stops, one by one - and what the win rate is actually worth

The bank pays -0.934R for a stop and earns +0.231R for a win, so one stop erases four wins. Nothing
else in the engine has that leverage, which is why the stops deserve looking at individually rather
than as a rate.

All 337 of them (12.1% of 2,790 trades):

    depth past the stop, in units of the stop's own distance
      median +0.074      58.8% of stops overshoot by less than 10%
      90th   +0.281       5.0% overshoot by more than 50%
    timing: median 10.8h after entry, 19% inside the first four hours
    price returns to the take inside the original 48h window in 33.5% of them

    booked, with the stop:      -1.0183R each, -343.2R in total
    the same trades, no stop:   -0.9217R each, -310.6R in total
    difference: +32.5R, which is 17% of everything the bank makes

Two facts at once. The stop is usually pierced by a sliver - a median of 7% of its own depth - and
two thirds of stopped trades genuinely never come back, so it is reading the situation correctly.
Removing it is not an option anyway: the deep group averages -2.29R unstopped, and on the exchange
a 10x position with no stop is the failure mode already documented in the money-path audit.

So the question is whether it belongs further out. Swept at constant risk - R is divided by each
trade's own stop distance, so a wider stop is a smaller position rather than a bigger bet:

| stop | fit 2022-24 | exam 2025-26 | stop rate | win rate on exam |
|---|---|---|---|---|
| 2.50 | 0.39 | 0.17 | 15.6% | 82% |
| **3.00 (deployed)** | 0.34 | **0.28** | 13.4% | 84% |
| 3.50 (fit's choice) | **0.41** | 0.26 | 11.1% | 86% |
| 4.00 | 0.38 | 0.27 | 9.6% | 87% |
| 5.00 | 0.37 | 0.20 | 7.2% | 88% |

The fitting window wants 3.50 and the exam refuses it - the sixteenth rejected candidate. The
deployed 3.00 has the best exam ratio of all eight widths.

**And the win rate is a dial, not an achievement.** Widening the stop to 5 ATR buys 88% on the exam
instead of 84%, cuts the stop rate nearly in half - and costs a third of the money per trade
(+0.0317R against +0.0501R). The same thing was established for the signal bot in August; here it
is proved again on the bank's own data. A win rate bought this way is bought with the money it was
supposed to represent.

## The take: the one geometry change that survives everything

Widening the stop buys win rate and loses money. The take is the mirror dial, and the deployed
0.75 ATR was itself a choice (0.5 -> 0.75 on 11.09), so it got the same treatment. Stop held at
3 ATR, risk constant, real book costs:

| take | fit 2022-24 | exam 2025-26 | exam win rate | average win |
|---|---|---|---|---|
| 0.50 | 0.26 | 0.18 | 89% | +0.148R |
| **0.75 (deployed)** | 0.34 | 0.28 | 84% | +0.231R |
| **1.00** | **0.35** | **0.33** | 80% | **+0.314R** |
| 1.25 | 0.26 | 0.34 | 76% | +0.395R |
| 2.00 | 0.22 | 0.30 | 65% | +0.628R |

The fitting window picks 1.00 and the exam agrees: +0.0685R a trade against +0.0501R, **37% more
per trade**, at a drawdown of -11.8R against -11.2R.

This time the walk-forward came first, because that is what killed the SMA change after it had
already been written up. Take 1.00 passes it decisively:

    forward half-years where 1.00 beats 0.75:   9 of 9
    the rolling choice on trailing data:        1.00, 1.00, 1.00, 1.00 - never wavers
    R per month forward:  rolling +3.10 | 0.75 +2.47 | 1.00 +3.10
    better in 4 of 5 years by ratio, 5 of 5 by R per trade
    monthly bootstrap: +35.6R, at or below zero in 0.0% of draws
    with 2022 removed: +30.0R, 0.1%

SMA32 managed one forward window of four and a 4.8% bootstrap. This is a different class of
evidence.

Pairs were also swept. The fitting window's favourite pair is take 1.00 with a 4 ATR stop (ratio
0.49) and the exam refuses it - 0.30 against 0.33 for take 1.00 with the stop left alone. The
single knob generalises; the pair overfits.

The cost is win rate: 84% -> 80% on the exam. The take was set to 0.75 partly to honour a 75%
floor, which 80% still clears.

## All three changes, as money, through the live guards

`$120`, the deployed sizing, the daily 3% pause, the latched 15% drawdown, margin refusing what it
cannot fund:

| version | risk | $120 becomes | drawdown | latch |
|---|---|---|---|---|
| as deployed (take 0.75) | 1.5% | $1,155 | -13.1% | no |
| drop BILL + BTC at half | 1.5% | $1,246 | -13.3% | no |
| take 1.0 alone | 1.5% | $1,908 | **-14.6%** | no |
| **all three** | **1.5%** | **$1,993** | **-13.3%** | **no** |
| all three | 1.25% | $1,273 | -11.1% | no |

**73% more money at the same drawdown**, and the changes repair each other: take 1.0 alone pushes
the drawdown to -14.6%, four tenths of a point from a latch that never resumes, and halving BTC
pulls it back to -13.3%. The latch boundary moves only from 1.779% to 1.742%, so the headroom at a
1.5% setting is preserved.

At 1.25% the package returns more than today's setting does at 1.5% ($1,273 against $1,155) with
two points less drawdown - which is the conservative way to take the same improvement.

### Take 1.0 attacked from four more directions, and the mechanism identified

| attack | result |
|---|---|
| coin split | both halves better (+0.0888 vs +0.0666, +0.0728 vs +0.0517) |
| every rule separately | all five improve - it is geometry, not one scenario's quirk |
| 22 unpinned coins | neutral (+0.0360 vs +0.0359); no power, as before |
| **mirror** | stays firmly negative (-0.0400), but improves by +0.0022R |

The mirror moving the same way is the only blemish, and it is a tenth of the real side's +0.0218R.
It also pointed at the right question: a larger target means fewer trades reach it, so more run to
the 48h exit - and in a market that mostly rose from 2022 to 2026, holding longer in longs collects
drift, which is a bet on the next four years rather than an edge.

It is not drift:

| | take 0.75 | take 1.00 |
|---|---|---|
| exits at the take | 2,373 (85.2%), +549.7R | 2,056 (80.8%), **+647.7R** |
| exits at the stop | 347 (12.5%), -354.7R | 396 (15.6%), -404.8R |
| exits at 48h | 64 (2.3%), **-26.4R** | 93 (3.7%), **-33.3R** |
| LONG total | +97.2R | **+121.4R (+25%)** |
| SHORT total | +71.4R | **+88.2R (+24%)** |

Three things settle it. The time-exit bucket loses money at both settings and is 2-4% of trades, so
the gain is not "hold longer". Longs and shorts improve by the same amount - 25% and 24% - while
drift would help one and hurt the other. And the cost is visible and paid for: the stop rate rises
from 12.5% to 15.6%, costing 50R, against 98R more collected at the target.

Median holding time goes from 2.8 to 4.5 hours, nowhere near the 48h cap or the owner's three-day
limit.

### Two follow-ups the take change forced, both negative

**The 48h exit was re-read**, because it had only ever been validated against a 0.75 take and trades
now live longer (median 4.5h instead of 2.8h, 3.7% reaching the time exit instead of 2.3%):

| hold | fit 2022-24 | exam 2025-26 | R per month on exam |
|---|---|---|---|
| 24h | 0.36 | 0.32 | +3.49 |
| 36h | **0.36** | 0.33 | +3.88 |
| **48h (deployed)** | 0.35 | **0.33** | +3.86 |
| 60h | 0.36 | 0.31 | +3.74 |
| 72h | 0.36 | 0.30 | +3.69 |

The fitting window marginally prefers 36h and the exam cannot tell them apart. The surface is flat.
48h stands, unchanged.

**Per-rule takes were tested and fail on the fitting window itself.** The original bank gave each
scenario its own geometry before the wide set flattened all five to 0.75, so it was worth asking
whether five values beat one:

| | fit | exam | per trade | R per month |
|---|---|---|---|---|
| each rule its own take | 0.29 | 0.28 | +0.0896R | +4.70 |
| **one take of 1.0 for all** | **0.35** | **0.33** | +0.0685R | +3.86 |
| as deployed (0.75) | 0.34 | 0.28 | +0.0501R | +3.08 |

The rules were tuned one at a time but run as a portfolio, so the collection of individually best
takes is worse than a single value even on the window that chose it. The per-rule mix does earn more
per month, but buys it with drawdown - and drawdown is what the live latch spends. Rejected: the
seventeenth candidate.

### The audit re-run under the new take

The hour windows and thresholds were all re-picked while the take was 0.75. They are entry
parameters and the take is an exit parameter, but one-position-per-coin couples them - a
longer-lived trade occupies its coin and a signal that used to be taken is now skipped - so the
conclusions are not automatically inherited.

**All four hour windows still hold** under take 1.0, ranking 11th, 1st, 2nd and 1st of 21 on the
exam. The short morning window, the weakest before, is now first.

**All six thresholds still hold.** Two are the fitting window's own choice again; three others have
a fit-preferred alternative that the exam refuses (0.32, 0.27 and 0.27 against 0.33).

The sixth is worth recording because it nearly became a candidate. `ret24 >= 4.8` instead of 3.864
was rejected at take 0.75 (exam 0.26 against 0.28) and passes at take 1.0 (exam 0.35 against 0.33) -
either a real interaction or a second bite at the same data. The walk-forward answers:

    better in 4 forward half-years of 9 - a coin flip
    the rolling choice lands on 4.8 every time and delivers +2.98R a month against 3.864's +3.10
    bootstrap: +4.5R, at or below zero in 30.2% of draws; without 2022, 37.5%

Rejected - the eighteenth candidate. Set beside take 1.0's nine of nine and a 0.0% bootstrap, the
difference between a finding and a fluctuation is not subtle once the right instrument is used. The
exam alone called it an improvement; the exam alone is not enough.

## Trailing exits: tested in every form, rejected in every form

If a bigger target is better, is no target better still? The 4h engine in this repo makes four times
the bank's R by letting winners run behind a trailing stop, and take 1.25 already earns more per
trade than 1.0 while losing on ratio - the money is out there and drawdown is what stops the bank
reaching for it.

Built the way trend4h builds it: the trail level is recomputed from CLOSED bars and checked against
the NEXT bar, because a trail that moves on the bar it is tested against reads the future.

| exit | fit 2022-24 | exam 2025-26 | exam R per month |
|---|---|---|---|
| **fixed take 1.0 (proposed)** | 0.35 | **0.33** | **+3.86** |
| pure trail 1.5 ATR | **0.39** | 0.03 | +1.11 |
| trail 1.0 armed at +0.5 | **0.51** | 0.08 | +1.72 |
| trail 0.75 armed at +0.75 | 0.45 | 0.18 | +2.60 |
| trail 0.75 armed at +2.0 | 0.21 | 0.17 | +4.82 |

Twenty variants, none within reach of the fixed take on the years that chose nothing. Some earn more
per month - trail 0.75 armed at +2.0 reaches +4.82R - and buy it entirely with drawdown, which is
the currency the live latch spends and where there is no headroom.

Look at the fitting column. A trail reached **0.51** there against the fixed take's 0.35. On that
evidence alone the right move would have been to rebuild the bank's whole exit. The nineteenth
rejected candidate, and the clearest illustration yet of why the fitting window is not evidence.

One correction along the way: the first version of this test armed the trail at +1.0 ATR while the
take sat at 1.0 ATR, so the take always fired first and the trail never engaged - every variant
returned the plain take's numbers to four decimals. An inert knob returning identical numbers is a
result about the test, not about the strategy, and it was rebuilt before being read.

## The 15-minute watch window is not a constraint

The order is placed at the hour's close and given exactly one 15-minute bar to be touched. That is a
chosen number, and the obvious worry is the one already documented for the signal bot - a longer
window fills the signals price walked away from and came back to, which is chasing a level that has
moved.

It turns out not to matter either way:

| watch | trades (fit) | fit ratio | exam ratio | exam R per month |
|---|---|---|---|---|
| **15 min (deployed)** | 1,417 | 0.35 | **0.33** | +3.86 |
| 30 min | 1,421 | 0.35 | 0.33 | +3.88 |
| 45 min | 1,422 | **0.36** | 0.33 | +3.85 |
| 60 min | 1,422 | 0.36 | 0.30 | +3.81 |
| 120 min | 1,425 | 0.35 | 0.30 | +3.89 |

Eight extra trades from stretching the window eight times longer. The reason is mechanical: the
level *is* the hour's close, so the order sits exactly at the market when it is placed - price either
touches it in the first minutes or leaves decisively. There is nothing to chase and nothing to gain.
The twentieth candidate, and one less thing that can go wrong live.

## A deeper entry level: rejected, and the entry side is now closed

With a wider target it is fair to ask whether the entry wants more room too - the order sits at the
hour's close, and a level k ATR below it buys a better price on the fills it gets while losing the
trades that never come back. Tested at constant risk (the stop stays 3 ATR from the actual fill) and
paired with the watch window, since a deeper level obviously needs longer to be reached:

| offset | watch | trades (fit) | fit ratio | exam ratio | exam R per month |
|---|---|---|---|---|---|
| **0 (deployed)** | 15 min | 1,417 | 0.35 | **0.33** | **+3.86** |
| -0.10 ATR | 15 min | 1,181 | 0.26 | 0.19 | +2.80 |
| -0.25 ATR | 15 min | 810 | 0.27 | 0.09 | +1.74 |
| -0.50 ATR | 60 min | 763 | **0.41** | 0.13 | +1.63 |
| -0.75 ATR | 60 min | 433 | 0.20 | 0.05 | +0.61 |

Monotone degradation on the exam, and the fitting window's favourite - half an ATR deeper with an
hour to fill - returns 0.13 against 0.33. The price improvement never covers the trades lost: half an
ATR deeper leaves 289 fills of 1,417. The twenty-first candidate.

**The entry side is now closed.** Hours, thresholds, level and watch window have all been re-picked
blind, and all four sit where the code already had them. Combined with the exit work, every
parameter the bank has now has a number behind it rather than a history.

## The regime gate: the last structural question, and it holds

The bank acts only while the 4h trend strategy holds a position on the same coin. That is a
structural choice rather than a number and had never been re-examined.

| gate | fit trades | fit ratio | exam trades | exam per trade | exam ratio | exam R per month |
|---|---|---|---|---|---|---|
| **the coin's own trend (deployed)** | 1,417 | **0.35** | 1,128 | **+0.0685R** | **0.33** | +3.86 |
| no gate at all | 5,244 | 0.22 | 3,483 | +0.0228R | 0.11 | +3.77 |
| BTC's regime instead | 1,489 | 0.19 | 1,056 | +0.0568R | 0.18 | +3.00 |

Removing the gate triples the trades and earns **the same money per month** - +3.77R against +3.86R
- at twice the drawdown. That is the clearest statement of what the gate does: it is not cutting
profit, it is cutting noise, and the noise is what a drawdown is made of. Gating on BTC's regime
rather than the coin's own is markedly worse.

Trend age at signal time was also measured, since a pullback into a three-day-old trend is not
obviously the same trade as one into a three-hour-old trend. Median age 36h, quartiles 8h and 93h:

    0-12h  +0.0718R     24-48h +0.0730R     96-192h +0.0109R
    12-24h +0.0453R     48-96h +0.0863R     192h+   +0.1568R   (exam)

Non-monotone, and the weakest bucket sits directly below the strongest. There is no filter here.

**With this the bank is audited end to end** - entry, exit, regime, sizing and universe - and every
parameter it has now rests on a measurement rather than on the history of how it got there.

### The take, checked on the exchange that triggers it

Take 1.0 was chosen and validated entirely on the analysis feed, but the take is a level on the
X-Perp, whose candles run narrower. A target set further out is, in the venue's own terms, harder
to reach than the backtest thinks. Run against the cached real X-Perp bars:

| take | filled | win rate | per trade | total | (analysis feed total) |
|---|---|---|---|---|---|
| 0.50 | 99 | 93.9% | +0.0894R | +8.9R | +12.9R |
| **0.75 (deployed)** | 88 | 90.9% | +0.1304R | +11.5R | +14.1R |
| **1.00 (proposed)** | 82 | **89.0%** | **+0.1814R** | **+14.9R** | +14.4R |
| 1.25 | 72 | 83.3% | +0.1848R | +13.3R | +17.4R |
| 1.50 | 67 | 80.6% | +0.2144R | +14.4R | +19.5R |

Two things.

**The change is confirmed on real prices**: +0.1814R against +0.1304R, +14.9R against +11.5R. The
win rate there is 89.0% rather than the 80% the analysis feed predicted, so on the venue the change
costs less win rate than the backtest charged it.

**And the venue caps how far it is worth going.** On the analysis feed the total keeps climbing past
1.0 - 14.4, 17.4, 19.5 - while on the X-Perp it falls back: 14.9, 13.3, 14.4. The narrower candles
reach a distant target less often, and the analysis feed cannot see it. So 1.0 is not merely the
first acceptable value, it is close to the optimum where the money actually is, and there is a
measured reason not to go further.

Six coins over three months, so this confirms a direction rather than measuring a size - but it is
the direction the mechanism predicted before the data was looked at.

### The take on the universe the live gates actually leave

Take 1.0 was validated on all sixteen pinned coins. The deployed bot will not trade sixteen: the
spread gate blocks BILL, XLM and AAVE in every book snapshot and NEAR in 57% of them, and the
slippage gate periodically excludes NEAR and DOT for a week. If the improvement lived in the coins
the bot will not trade, it would not be an improvement.

| universe | take 0.75 (exam) | take 1.00 (exam) | gain |
|---|---|---|---|
| all sixteen | +0.0501R, ratio 0.28 | +0.0685R, ratio 0.33 | +0.0184R |
| BILL dropped (proposed) | +0.0544R, 0.30 | +0.0728R, **0.41** | +0.0184R |
| minus what the gate always blocks | +0.0635R, 0.34 | +0.0842R, **0.44** | +0.0207R |
| also minus NEAR | +0.0676R, 0.36 | +0.0907R, **0.48** | +0.0231R |
| also minus NEAR and DOT | +0.0718R, 0.37 | +0.0917R, **0.49** | +0.0199R |

It holds everywhere and grows as the universe narrows. On the set the bot will actually trade the
ratio reaches 0.44-0.49 against 0.34-0.37 for the deployed take - materially better than the
headline figures, which are computed on coins the gates will refuse.

Dropping BILL is also worth more under the new take than it was under the old one: 0.41 against
0.33 with all sixteen, where at take 0.75 the same removal moved 0.28 to 0.30.

The eleventh check on the take, and the one that matters most operationally: it is about the coins
the money will actually be in.

## The edge against a null, on five years

The X-Perp control gave +0.1814R against -0.0424R for a random entry, but on 82 trades. The same
comparison on the full analysis-feed history, same geometry, same per-coin costs, same 48h exit -
only the decision of when and which way replaced by a coin toss:

| year | bank | random | edge |
|---|---|---|---|
| 2022 | 82%, +0.1096R (n=288) | 74%, -0.0196R (n=4,504) | **+0.1292R** |
| 2023 | 82%, +0.0963R (n=590) | 73%, -0.0349R (n=8,235) | **+0.1312R** |
| 2024 | 82%, +0.0815R (n=539) | 73%, -0.0297R (n=9,007) | **+0.1112R** |
| 2025 | 80%, +0.0698R (n=586) | 74%, -0.0197R (n=12,080) | **+0.0895R** |
| 2026 | 81%, +0.0671R (n=542) | 73%, -0.0443R (n=13,397) | **+0.1114R** |
| **all** | **81%, +0.0824R** (n=2,545) | **74%, -0.0312R** (n=47,223) | **+0.1136R, 11.2 sigma** |

Steady in every year, from +0.089 to +0.131, with no decay across the five.

**And this is where the win-rate argument finally resolves.** A random entry wins 74% of its trades,
because a 1 ATR take against a 3 ATR stop is a bet that usually wins - and it still loses money once
costs are paid. The bank wins 81%. Those seven points above free are the whole edge. The 88%
available by widening the stop to 5 ATR is bought with a third of the money, and the 74% underneath
is worth nothing at all.

The number worth asking for was never the win rate. It is the points above random, and they are
seven, stable across five years, at eleven sigma.

### Every rule against a null of its own side

The combined null is -0.0312R, but a random long and a random short are not the same bet, and
comparing a short rule against a mixed null distorts it. Computed separately over five years:

    random LONG    72.5% win rate,  -0.0464R
    random SHORT   74.8% win rate,  -0.0159R

Both lose; shorts lose less, because the take/stop geometry dominates any directional drift. That
refinement matters: measured against the combined -0.0312R the two short rules would have looked
about 0.015R better than they are.

| rule | side | trades | win rate | per trade | own null | **edge** | sigma |
|---|---|---|---|---|---|---|---|
| btc_pump_evening | LONG | 377 | 88% | +0.1698R | -0.0464R | **+0.2162R** | **10.2** |
| btc_pump_night | LONG | 393 | 87% | +0.1494R | -0.0464R | **+0.1958R** | **8.8** |
| coin_run_evening | LONG | 955 | 81% | +0.0721R | -0.0464R | **+0.1185R** | **7.1** |
| short_btc_up_morning | SHORT | 598 | 84% | +0.0920R | -0.0159R | **+0.1079R** | **5.3** |
| short_pop_in_downtrend | SHORT | 667 | 77% | +0.0578R | -0.0159R | **+0.0737R** | **3.8** |

**All five beat their own null with significance, and none is dead weight.** The weakest is
`short_pop_in_downtrend` at 3.8 sigma - still sound, but it is the bank's thinnest component on
every measure at once: no hour window, the lowest win rate at 77%, and the smallest edge.

This is the first time each rule has had a significance of its own rather than being carried by the
portfolio average.
