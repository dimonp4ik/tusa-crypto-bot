# Shared-account factor-cap study

> Archived on November 15: the crypto leg below uses superseded zero-valued expired-runner labels. These cash returns are not current profit estimates. See `nov15_profit_target_revalidation.md` and the `expiry_mark_v2` reports. No old numbers were silently replaced.

No live policy is changed. The +30% contribution from each bot every month remains unverified.

## Verified mechanism

Direxion identifies SOXL and SOXS as daily +300% and -300% exposures to the same semiconductor benchmark: https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs . Therefore LONG SOXL / SHORT SOXS are directionally aligned daily-benchmark proxies. The mapping is not an assertion of constant intraday beta, exact correlation, or guaranteed hedging. Other equities and crypto-linked stocks are not assigned to this factor by inference.

The research cap permits at most one active ETF position per effective benchmark direction. It is applied before stock portfolio admission; freed scan/hour slots are rerouted and the resulting replacement trades are included. Opposite effective directions retain separate budgets. Exact-time exits cannot free capacity for an entry at that timestamp.

## Current route and data

The full current stock route is rebuilt through the November-14 snapshot. Its mature historical preportfolio prefix matches the November-11 reference. ATR and outcome helpers now allocate arrays once per symbol; regression checks establish identical gap/stop-first behavior and closed-bar ATR values. Both stock replays include the current -2R daily pause, two-loss pause, per-scan/hour/direction rails and all current reversal backing/premarket guards.

The standalone cap removes 47 original admissions and adds 31 replacements. It improves the partial-November return slightly, but worsens September and October return and drawdown. This fails a broad promotion criterion; the cap is not connected to real trading.

## Shared cash ledger

Each monthly scenario starts flat with $125.64, 3% client margin for both bots and leverage 10. Crypto TP1 is the client's 5%, rounded by production lot/minimum functions. Stock single-target exits settle the full position. Stock admission also requires the last 20 shadow outcomes closed strictly before entry to have positive total R; blocked real trades still remain observable shadow outcomes.

The account enforces local slots 4/8, global limit 12, conservative two-writer headroom, 90% maximum margin, and an assumed 900-second reservation-release delay. At every entry, cash sizing includes only earlier realized partial/runner cash. Both crypto-first and stock-first ordering of simultaneous signals are tested.

| Monthly entry cohort | Baseline combined return | Factor-cap combined return | Baseline cash drawdown | Factor-cap cash drawdown |
|---|---:|---:|---:|---:|
| July | +32.751622% | +32.751622% | 3.582018% | 3.582018% |
| August | +25.275759% | +25.275759% | 4.148085% | 4.148085% |
| September | +26.804278% | +22.863554% | 4.961535% | 5.826224% |
| October | +39.097867% | +32.567303% | 2.695827% | 4.549759% |
| November partial | +2.133756% | +2.435876% | 2.936831% | 3.213451% |

These are purged historical entry cohorts, not actual account-month PnL, complete marked equity returns, or forecasts. Month-end boundary exits and inherited positions are excluded. July has partial instrument history and cold-start stock-readiness history. November is incomplete. Current contract specifications are assumed across the history. Historical spread, fees, actual publication/fill latency and unrealized PnL in entry sizing are not reconstructed. Full entry margin is conservatively reserved until slot release. Earlier periods were already inspected during research.

No scenario observes more than 11 reserved positions or 4/8 local slots; peak reserved margin is below 33%. This is model evidence under its stated sequential/timing assumptions, not proof of an atomic coordinator between two Railway databases. The two tested within-bar orderings produce identical returns in these cohorts. Per-bot cash contributions reconcile to the combined return, and do not establish +30% from each bot.

The single-bot ledger's eight existing reference scenarios reproduce unchanged when the new shared-account options are disabled. New tests cover local quotas, delayed slot release, concurrent-writer margin headroom, strict readiness timing, and exclusion of older outcomes outside the readiness window. Existing local research helpers and archived PKL inputs are required to rerun the drivers.
