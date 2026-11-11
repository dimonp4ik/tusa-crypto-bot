# Direction-aware veto and calendar coverage audit

No live trading rules were changed. The monthly objective is not met.

## Calendar coverage

The historical rows labelled 2023, 2024 and 2025 previously covered roughly January/February through July, not full years. This audit reconstructs full annual entry cohorts by joining three overlapping 18,000-bar archives. Every overlapping OHLCV value must match, and every input timeline must be continuous. Each year has 385 bars of pre-year BTC context and at least 48 bars beyond year end for outcome resolution.

The original five-symbol core is BTC, ETH, LINK, XLM and XRP. Additional available production-universe symbols are evaluated separately; missing archive symbols are listed explicitly. Model parameters, entry cutoff, client margin 3% and leverage 10 are unchanged.

| Annual entry cohort | Trades | Positive exits | Gross return after cohort exits | Closed-trade equity drawdown |
|---|---:|---:|---:|---:|
| 2023 | 779 | 64.96% | +4.8094% | 10.3491% |
| 2024 | 1,078 | 65.40% | +7.9043% | 10.5006% |
| 2025 | 1,350 | 65.78% | +5.8276% | 11.2327% |

These are isolated crypto stress tests on archived non-X-Perp candles, not shared-account or current live-bot results. Each annual cohort starts flat on January 1; it does not inherit positions or cooldowns from the preceding year. Return includes exits carried into the following January. Drawdown excludes unrealized swings; costs are excluded. They must not be described as an annual account forecast or evidence of +30% monthly profit.

Before interpreting the extra months, entry identities and geometry were compared against the shared intervals of the old raw-family archives: 780 matches in 2023, 1,210 in 2024 and 1,585 in 2025. Entry, stop, risk fraction, efficiency, z-score and frozen-model probability match to absolute tolerance 1e-12. This establishes entry-input parity, not portfolio-result parity: starting earlier changes portfolio ownership and cooldowns.

## Direction-aware shadow health

The new research filter conditions completed paper results on direction plus market phase and/or family. It evaluates 27 fixed combinations of 30/60/90-day history, 10/20/30 observations and three group definitions. Equal-time exits are unavailable; rejected signals can later restore group health. Portfolio admission is replayed after the veto, including replacement candidates.

The historical candidate pool was rebuilt from raw family archives. The older temporary historical cache contains already-routed rows and is not an adequate source for reconstructing freed-slot replacements. The new audit avoids that cache.

The development winner uses direction plus BTC phase, 60 days and at least 10 completed observations. It improves the first 2022 selection slice, but fails later checks. In the January-July 2025 slice, gross return changes from +20.4651% to -3.9855% and closed-trade drawdown from 4.5442% to 11.4654%. The post-November-7 tail also worsens from -1.4673% to -1.5706%.

On that new tail the rule removes two shorts that stop out, but also removes the ZEC breakout runner winner (+2.62017% signed price move). Portfolio replacement adds only a small banked AAVE long. The group veto is too coarse to distinguish the profitable breakout from the surrounding failures. This is evidence against deploying this rule; it is not evidence that more vetoes automatically reduce account losses.

## Reproduction scope

JSON reports contain source hashes, exact interval boundaries, monthly cash-flow returns, exclusions and individual removed/replacement trades. The research drivers require the existing local research helper modules and archived PKL/CSV inputs; they are not a self-contained backtest package on a fresh clone. No new exchange credentials or other paid data source was needed.
