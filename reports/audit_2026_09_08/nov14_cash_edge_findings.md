# Cash-edge and executable TP1 study

The +30% per bot per month objective remains unverified. This update changes research tools and reports only, not the live strategy or a client's settings.

## Fresh data and executable partials

The November-14 public snapshot adds 4,212 symbol-bars across 12 X-Perp symbols to the complete November-11 history. Overlapping candles match exactly. The frozen replay reproduces 891 previously mature trades before evaluating the new tail.

The lot-aware simulation starts each cohort with $125.64, margin 3%, leverage 10 and at most four crypto slots. It uses the production entry and partial quantity rounding, verifies that the contract ID matches the candle manifest, and realizes TP1 and runner cash separately. Equal-time cash flows cannot fund an entry at that same timestamp. Partial exits do not free a position slot.

On the 17 new entries after the November-11 cutoff:

| Requested TP1 share | Unplaceable partials | TP1 hits without a bank | Gross cohort return | Cash-flow drawdown |
|---|---:|---:|---:|---:|
| 5% | 10 | 5 | -1.063128% | 2.143139% |
| 25% | 1 | 1 | -1.076085% | 2.135588% |
| 50% | 0 | 0 | -1.274879% | 2.158686% |
| 75% | 0 | 0 | -1.449648% | 2.172757% |

Larger partials fix the minimum-lot problem but do not establish a profitable entry system. These are hypothetical cohorts, not reconstructed account PnL. They exclude inherited positions, stock exposure, unrealized PnL in entry sizing, fees and actual order-book fills. Current specifications are assumed constant across the short cohort. Recent entry bars without a full 48-bar outcome window are not included.

## New expected-return prototype

A regression prototype predicts the actual price return of a 50%-at-TP1, 2R-runner-with-breakeven policy rather than the old distant-target classification label. It fits 28,938 examples with exits strictly before January 1, 2022. Two fixed tree sizes and eight selection quantiles are compared on 2022 and the available 2023 interval. A passing family with the highest predicted edge wins per symbol/bar. Nothing is retrained on 2024+, the November tail, or current client results.

The chosen development variant uses 15 leaves and the 95th score percentile. Its cut is approximately 0.2661% predicted price return. It produces only 0.52 trades per calendar day in the 2023 confirmation interval and no observed month reaching +30%. This historical pool still has incomplete years; those results are preliminary and are not complete-year validation.

On the current new X-Perp tail with lot rounding, it admits four trades, returns -0.61268% and has 1.029155% cash-flow drawdown. Lower drawdown on fewer trades does not satisfy the profit and frequency objectives. No model artifact is promoted or connected to real orders.

## Scale of the target

For an idealized sequence with 3% margin and leverage 10, an identical effective signed price return r on N trades satisfies `(1 + 0.3*r)^N = 1.3` to reach +30%. At 90 trades it requires r approximately 0.9731% per trade; at 150 trades, approximately 0.5835%. This is an equal-return geometric benchmark, not an arithmetic-mean requirement or a forecast. Overlaps, loss distribution and actual fills require the shared-account ledger.

The reports and drivers require the existing local research inputs/helper modules; the repository commit is not a self-contained market-data package. The live stock publication fix is a separate deployed commit, a6e0956; its successful deployment does not prove profitability or recovery of its 20-trade forward gate.
