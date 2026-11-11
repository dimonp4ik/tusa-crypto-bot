# Frozen rules on appended public candles, November 11, 2026

This check does **not** establish +30% monthly profit or readiness for larger live allocations. No trading rule was promoted by this audit.

The new tail was appended to the complete previous history, preserving indicator warm-up. Every overlapping OHLCV value was equal. Both datasets end at the 2026-11-11 04:45 UTC candle start. Crypto added 4,308 symbol-bars across 12 symbols; stocks added 18,400 across 46 available symbols. The stock manifest preserves the unavailable TEST002 placeholder required by its frozen universe.

## New-entry cohorts

All numbers below are gross hypothetical **isolated bot** results with client margin 3% and leverage 10. They are not combined-account or exchange-fill results. Drawdown measures equity after closed trades, excluding unrealized swings.

| Frozen route | Entry cutoff, exclusive (UTC) | Trades | Positive exits | Return | Closed-trade drawdown | Maximum loss streak |
|---|---|---:|---:|---:|---:|---:|
| Crypto baseline | Nov 7, 11:00 | 9 | 33.33% | -1.4673% | 1.5187% | 4 |
| Stocks current route | Nov 7, 00:45 | 32 | 34.375% | +0.0808% | 2.4632% | 6 |

Crypto's raw exporter requires a complete 48-bar outcome window, so the latest 12 hours are not a complete entry cohort. The report records its last exported entry separately. Neither sample is extrapolated to monthly or annual profit.

## Previously proposed crypto filters

- The frozen next-open TP1 extra-close experiment changed the new return to -1.3518%, but left closed-trade drawdown at 1.5187%. It did not solve the losing entries.
- The previously frozen quality threshold 0.3356824161327173, evaluated as a post-signal veto, left 6 trades, worsened return to -1.5370% and drawdown to 1.5455%. It removed the only full runner winner along with two stops. Fewer consecutive losses alone did not improve the account result.
- The 14-day short health veto with at least 15 completed shadow outcomes removed no new signal. It cannot be credited with protecting this episode.

The last two are signal-veto diagnostics without rerouting replacements from freed portfolio slots. They do not approve deployment. Their thresholds were not retuned on this tail.

Rebuilding the old mature prefix produced identical identities and outcomes: crypto verification is in crypto_frozen_nov11_forward.json; stocks matched 886 prior mature trades. Source hashes, individual trades and exact boundaries are recorded in the respective JSON reports. Replay scripts and candle caches remain in the local research workspace; this commit does not claim a self-contained replay package.

Next research should address entry losses and portfolio concentration, rather than treating a shorter loss streak or more TP1 exits as proof of lower drawdown. New rules derived from this episode need a later untouched forward interval before promotion.
