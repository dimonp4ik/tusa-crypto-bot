# Failed-break acceptance and trend pullback: research only

The deployed TP1 minimum-lot fix is documented separately in TP1_MINIMUM_CLOSE.md.
This study does not alter production signals, client settings, or live orders.

## Inputs and hypotheses

The immutable November 19 public snapshot adds ten crypto contracts and six US
equities to the existing twelve crypto and eleven matched trade/mark equity
contracts. All 22 added history streams completed with validated candle schemas,
alignment, continuity, and SHA256 fingerprints. Exact current contracts only:
DOT begins in August; the current TTWO contract was listed after the research
end and was excluded by the declared listing/warmup rule. Historical unavailable
contracts are not substituted. Snapshot turnover selection has hindsight and
survivorship bias for earlier months; this is not a historical investable universe.

Two new scenarios require closed candles: a breakout that fails, returns inside,
and is followed by a separate acceptance candle; and a pullback to the last closed
hourly EMA20 followed by a closed reclaim. Both use own and market 4h context,
actual trade activity, structural stops, and 0.5–1% first targets. The old range
sweep and trend retest are retained as controls. All brackets use the production
tick-rounding function, reject collapsed targets, and satisfy the declared stop,
target, and reward/risk floors after rounding.

The first declaration stopped before the first candidate output because legacy
cash-sizing metadata omitted tickSz. The frozen original source and failure
record are retained. Revision v2 fills ticks from the frozen snapshot by exact
contract ID and rejects absent, duplicate, or conflicting specifications. It
does not change strategy parameters. V2 was declared before outcome evaluation.

## Result

There are 548 crypto candidate paths and 33 equity candidate paths, with the
equities independently resolved against last-trade and mark prices. Every path
is scorable in this sample. The crypto count comprises 320 control entries and
228 new entries; equities have 26 control and seven new entries. These counts
include overlapping hypothetical strategies and are not executable trade counts.

None of the 100 declared universe/route/trigger combinations met even the
selection requirement of positive cash contribution and twenty closed trades
per bot per month in August and September, across both TP1 policies and order
priorities. Thus no October/November promotion or live activation is justified.

Example diagnostic only, expanded universe, new scenarios combined, 50%/minimum
lot, crypto-first priority: crypto contribution is -0.177% in August (38 closes)
and -0.152% in September (49 closes). Equities remain below the fresh 20-paper
trade gate. Changing the TP1 size does not manufacture an entry edge.

The account simulation starts with one continuous $130 account, 3% client margin,
10x leverage, actual lot rounding, per-bot/shared position caps, and delayed
shared margin release. It is realized cash, not a reconstruction of full equity,
private fills, historical spreads, funding, fees, or current account returns.
Mark-trigger results differ from current last-trigger orders; the NY clock omits
holidays and early closes. No result proves +30% each bot each month.

## Reproduction and checks

Public history: `python -X utf8 -u download_native_expansion_nov19.py` resumes the
immutable page cache and refuses changed source/selection fingerprints. Study:
`python -X utf8 -u liquidity_trap_joint_nov19_audit.py` checks the v2 declaration
and input SHA256s, then writes the v2 report and candidate caches. Raw public
pages and pickle caches are kept locally rather than committed. Source manifests,
public snapshot, declarations, failure record, and report are committed.

The seven focused research test files pass 26 tests: exact-contract ticks,
trade/mark volume distinction, invalid/conflicting bars, post-rounding bracket
floors, separate failure/acceptance, non-vacuous future-mutation causality,
macro opposition, maturity purge, and retaining unknown paths as unknown.
