# Asset-local expected-return prototype

Research only. No model, exit fraction or client setting is promoted to live trading. The objective is unverified.

Eight new features describe the traded asset itself: normalized trend, 96-bar return and efficiency, relative volume, close location in the signal candle, distance from prior upper/lower levels and candle body. All features are shifted to use only bars closed before the entry timestamp. A missing time step invalidates the indicator history; insufficient history is not imputed into fabricated features.

Regression fits only examples exited before 2022. Selection still uses 2022 and the available 2023 interval, then freezes before later evaluation. The new fit has 28,904 rows; 34 old training rows lack valid local features. Two tree sizes and eight score quantiles are compared. This is a new prototype, not a controlled attribution of every difference solely to the added features.

The selected development variant has seven leaves and a 90th-percentile cutoff. The 2022 selection return is +32.9279% for that whole selection period, **not per month**. No month reaches +30%. The 2023 confirmation frequency is 0.878 trades per calendar day, below the requested primary-bot frequency. Historical 2024/2025/2026 checks still use the previously available partial-year windows and cannot prove complete-year performance.

On the X-Perp cohort after the November-11 snapshot, with $125.64 starting capital, 3% margin, leverage 10 and production contract rounding, the new model admits eight trades and returns -1.537215%, with 1.824653% realized-cash drawdown. It fails the profit requirement. On the broader November cohort it returns +0.056517%; that is not reconstructed user-account PnL or a monthly forecast.

Tests cover entry-bar/future OHLCV independence, missing history, missing time steps and correct bulk attachment of features to duplicate family signals. With the optional feature switch disabled, the previous model's selected variant, training count, later-period metrics and lot-rounded X-Perp ledger reproduce exactly.

The JSON contains model feature names, input hashes, missing-row counts, every tested cutoff and individual hypothetical trades. Local archives and the existing research environment are required to rerun it.
