# Runner protection after minimum-lot TP1 banking

Research only. The already deployed client-selected 50%/minimum-lot TP1 mode is
unchanged. This work does not change client percentages, margin, stops, entries,
or the stock bot's PAPER restriction. The joint +30% per bot per month objective
is not verified.

## What was checked

The same 2,373 frozen native crypto proposals are replayed with four declared
runner rules: BE after TP1; half the original risk retained after TP1 until 1R;
the initial stop retained until 1R; and a two-completed-active-candle structural
stop until 1R. Original targets remain 0.5R and 2R. New stops never loosen an
existing stop. No new stop uses an earlier wick as proof of a fill: a crossed
amendment at the observing close is marked at that close. Zero-volume candles
cannot bank a target or trigger a stop; stale terminal prices remain unknown.

These are unrounded original known-close brackets and public-price proxies,
not proof of exchange execution. Exit-price stress is five basis points in the
checked scenarios. Real latency, historical bid/ask, private fills, floating
equity and current risk guards are not reconstructed.

## Correction to older comparisons

The old unchecked control exactly reproduces the earlier cash comparison:
requested5% +25.910041%, cash drawdown6.769514%; minimum50% +19.204779%,
drawdown5.541379%, over continuous July–partial November. This numerical parity
does not validate those returns as account PnL.

There are 248 proposals with at least one subsequent zero-volume candle. The
checked baseline has two stale terminal paths, one admitted by publication rails:
BILLUSDT, entry September24 15:45 UTC. The BE-at1R rule additionally admits an
unknown AVAX path from September3. Therefore the complete checked account curve
without an entry activity filter is **unscorable**. Neither unknown was assigned
a flat return or silently dropped. V1 refused all unknown proposals; V2 verifies
whether each unknown is causally admitted. Rejected unknown proposals cannot
alter subsequent admission, covered by a payload-mutation test.

The frozen V1 source/report are retained. Exit rules did not change in V2; only
the scope of the unknown-path audit changed. There is no live policy selection.

## Path diagnostics, not a portfolio return

The separate paired transition report labels its common-known intersections and
lists all unknown pairs. Deferring BE until1R recovers133 final targets but causes
296 additional negative runner exits relative to the early-BE baseline. There
are196 improved and367 worsened fixed-size hypothetical trades. These overlap
in time; their count and fixed-size PnL deltas are not account returns or an edge
estimate. Unknown paths are not removed from any claimed account curve.

## Observable activity hypothesis

A separate declaration then tested the already used entry activity rule of at
least12 positive-volume bars out of the last16, with the last closed bar positive.
It reads only candles before entry and reconciles cached activity features with
the raw source. It was motivated by observed stale outcomes and is exploratory,
not an unseen-period validation. It rejects51 of2,373 proposals: BILL34, ZEC11,
HYPE3, AVAX2, AAVE1. It removes the baseline admitted unknown; BE-at1R still has
an admitted unknown and is not cash-scored.

For the same activity-qualified proposals, one continuous $130 account,3% client
margin,10x and actual minimum50% banking, the sensitivity results are:

| Runner rule | July–partial November cash change | Realized cash DD |
| --- | ---: | ---: |
| BE atTP1 | +5.063% | 7.999% |
| Half initial risk until1R | +4.851% | 9.987% |
| Two active candle structure until1R | +4.180% | 9.150% |
| BE only at1R | Unknown | Unknown |

All scored variants have losing months. The activity rule, exit amendments,
zero-volume handling, and price stress differ from the old unchecked control;
the change from19.2% to5.1% must not be attributed to any one correction alone.
No candidate meets the objective or justifies a live exit change.

The account starts once in July and does not reset monthly. It keeps publication
cooldowns and direction/hour/loss-streak rails, caps crypto positions at4 and the
common account at12/90% reserved margin, models two writers and a15-minute margin
release delay, and sizes each entry using the current cash and contract grid.
Stocks are absent because the current stock strategy remains PAPER. This study
cannot establish the objective for both bots or predict the client's balance.

## Reproduction

`crypto_minimum_bank_runner_nov19_audit.py` reproduces V2, including the old-control
cash parity and refusal of admitted unknown paths. `runner_protection_pairs_nov19.py`
writes diagnostic transition counts. `crypto_active_entry_runner_nov19_audit.py`
replays the separately declared activity-filter cohort. All scripts verify frozen
source/data SHA256s and refuse changed declarations. Raw candle/pickle inputs
remain local; summary reports and declarations are committed.

Twenty focused tests cover long/short parity, causal future mutations with known
targets, empty-bar targets, gap fills, crossed amendments, active-only structural
levels, stale expiry refusal, rejected/admitted unknowns, and entry-time activity.
