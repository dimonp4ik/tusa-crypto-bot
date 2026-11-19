# Dense session learner and minute-source validation

Research only: the model did not qualify, no threshold is selected, and no model
weights or entry rules were deployed. The original two-bot/common-account+30%
monthly objective remains unverified. A separately tested production correction
for empty-bar touch evidence is in main, documented in LIVE_EMPTY_TRADE_BARS_NOV19.md.

## Why another model

The prior hand-written opening/session combinations had very few entries. This
experiment proposes both directions on every quality-qualified closed session
bar and lets one fixed small regressor evaluate explicit session features. This
changes the entry hypothesis rather than repeatedly intersecting rare signals.

The seventeen matched native X-Perp contracts supply22,328 hypothetical proposals
(both directions can be present; these are not executable trades). There are
22,014 known coarse trade-price paths and314 unknown paths. Unknown proposals
remain in prediction, direction ranking and publication admission. An admitted
unknown invalidates the cash case; it is never assigned a flat return.

Closed features include regular-session ATR, candle geometry, native trade
activity, current-session volume-weighted typical price, opening/prior-session
levels, regular-session returns, QQQ/SPY mark context, and leave-one-out peer
context. SOXS inverse exposure is handled. There is no asset-ID or outcome
feature. Futures cannot alter a completed feature vector, tested on nonempty
actual proposals. These native candle averages are not consolidated stock VWAP
or executable bid/ask quotes.

Stops use the last three completed mark-bar extremes plus0.15ATR, with0.3% floor
and1% cap. First/final targets are0.5%/1.5% price moves. Production tick rounding
and minimum1.5 final reward/risk apply before labeling. Entries/exits use adverse
5bp price scenarios. Expiry is the underlying cash session close, with the frozen
2026 holiday/DST calendar. Entry quality is unchanged from the prior session
study: real current trade,75% active session bars, first two opening bars positive
and last-trade reference within10bp of mark.

## Chronology and result

One HistGradientBoostingRegressor has seven leaves, minimum100 samples/leaf,
150 iterations, learning rate0.05, L2=5, no early stopping and fixed seed20261119.
July–August training has5,916 known rows. Full32-bar maturity purges boundary
trades, including fast losses.196 unknown training labels are not fitted; this
may cause censoring bias and is disclosed. All assets/directions at the same
timestamp share one total training weight.

The target is nominal50% TP1/runner signed price return, not win/loss. Private
client5% or minimum50% bank sizes are independently rounded to actual contracts
in the cash simulation. The model does not choose client margin or leverage.

Six predeclared thresholds are evaluated only on September. Qualification needs
at least15 PAPER closes with positive aggregate price and R in each half, plus
positive stock cash contribution and twenty stock closes over September across
both TP1 choices and both bot ordering priorities. The same common account
contains a fixed crypto reference, caps4/8/12 and90% margin, two-writer headroom
and15-minute margin release. It starts once with$130 and3% entry margin/10x.
No floating-equity guard, private fills, funding/fees or liquidation is modeled.

No threshold qualifies. At0.0005, there are89 admitted PAPER signals and seven
cash-ready stock entries. In the minimum50%, crypto-first scenario the September
stock contribution is-0.6323%, crypto+2.0553%, shared cash+1.4230%. The positive
shared total must not be described as stock profitability. At0.001 stocks also
lose; stricter thresholds lack enough entries. October/November are not scored
after failed selection. All history had previously been inspected in other
experiments; chronological separation is not a claim of an untouched holdout.

## Minute refinement, declared before replay

All990 September direction-selected proposals with frozen-model score>=0 are
covered, not only winners, losers or ambiguous15-minute bars. Their266 complete
native sessions were downloaded as confirmed1-minute candles and reconciled
against frozen15-minute contract volume and actual-positive-minute OHLC.
All266 windows pass reconciliation. Raw data stay local; plan and manifest are
committed. Two latency scenarios are declared: amendments active at the next
parent15-minute boundary or next1-minute bar. Neither measures live latency.

Both profiles have874 known paths and116 unknown last-minute time exits. A real
print in an earlier15-minute candle cannot verify a market exit at the scheduled
close. The final minute without trades remains unknown. At the lower working
thresholds7–12 unknowns are publication-admitted, so those cash cases are
unscorable. Remaining high thresholds still fail the same existing prerequisites;
no threshold grid or model is fitted again and no live policy is selected.

The source refinement leaves478 raw proposal SL outcomes as SL. Only two coarse
SLs become TP1_STOP (parent-bar amendment) or LOCK (one-minute amendment), and
one LOCK becomes TARGET. Thus guessing intrabar order cannot explain most losses.
Zero resting-stop/target same-minute ambiguities remain in the scored paths,
but unknown private fills, amendment timing and market liquidity remain.
This is source/timing sensitivity, not proof of execution or monthly returns.

## Runtime correction identified during the audit

Synthetic regression tests reproduce old live-monitor TP/SL transitions on
explicit zero-volume carried-price candles. Main now ignores these for touches
and new trailing actions in both repositories; earlier genuine TP2s still count.
The unsent-outcome tracker keeps missing/stale trade evidence unresolved.
Trading deadlines and exchange protection/reconciliation remain in place.
Crypto commit89dedb8 and stock commitf6c3e34 are separate from this research branch.
They do not activate this learner or change client settings.

## Reproduction and limits

Run `stock_session_dense_nov19_audit.py`, then `stock_dense_minutes_nov19.py`,
then `stock_dense_minute_nov19_audit.py` with `python -X utf8 -u`. Immutable
declarations/source SHA256s reject changed code or inputs. The minute plan pins
the fitted model and direction-selected cohort. Existing completed windows are
reused; incomplete/conflicting sources are not frozen as verified inputs.

The local pool, fitted model pickle and raw candle caches are not committed;
their hashes and environment versions are retained. Original fitted-model bytes
are necessary for exact replay of the pinned minute plan; a different software
environment/refit must be a new declared experiment, not a silent replacement.
Sixteen focused research tests cover causal features, full-horizon training purge,
timestamp weights, unknown direction/admission handling, parent-child source
reconciliation, ordered targets, deferred amendments, gaps, empty minutes and
long/short symmetry. Production checks are recorded separately in the task.
