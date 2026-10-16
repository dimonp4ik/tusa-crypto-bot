# Live stop and cluster audit — 2026-10-16

This audit reads the September 8 bot exports as signal accounting. It excludes
fees and funding, applies the requested 3% of current equity as margin with 10x
leverage, and compounds exits in entry order. Overlapping trades make the account
path an approximation; the signal outcomes and causal entry-cap selection are
exactly reproducible from the exports.

## Crypto

- 70 resolved signals, 57.14% successful statuses and 55.71% strictly positive
  exits. The estimated account result is -8.04%, with 10.75% maximum drawdown
  and 5.0 trades per active UTC day.
- Signals with another entry within 15 minutes lost 69.23% of the time and made
  -10.8934R (13 signals). Isolated signals lost 36.84% and made -0.5854R
  (57 signals). Fisher's exact p-value is 0.0597, so the effect is strong but the
  sample is still small.
- The worst hour was 2026-09-04 09:00 UTC: six LONG entries, five stops and
  -9.2613R. Three entries opposed the labelled trend and two were in range. The
  only winner was the one aligned with a bull trend.
- A causal maximum of two accepted entries in any rolling hour reduces the
  estimate to -5.79% and the maximum stop streak from five to three, but the bot
  remains losing. The cap is a loss brake, not a source of edge.
- Trend alignment alone does not repair crypto: the 17 aligned signals still
  estimate -2.71%. FVG entries account for -11.9467R; the five OB entries made
  +0.4679R, so false OBs are not the observed crypto failure.

## Stocks

- 144 resolved signals, 57.64% successful statuses and 56.25% strictly positive
  exits. The estimated account result is +10.41%, with 6.07% maximum drawdown
  and 7.2 trades per active UTC day.
- A causal maximum of two entries in any rolling hour keeps 104 signals and 5.2
  trades per active day. It estimates +10.29% with 3.86% drawdown and cuts the
  maximum stop streak from eight to five.
- The cap is not stable enough to install blindly: in the early chronological
  half it lowers return from +8.18% to +6.43%; in the late half it improves return
  from +2.06% to +3.63% and drawdown from 4.30% to 2.66%.
- The same-window trend label is the strongest diagnostic: 52 aligned signals
  estimate +12.04%, while 25 opposed signals estimate -1.94%. Range or missing
  labels estimate +0.50% but include a 14-stop run when isolated as a bucket.
  This is a legacy-export observation, not a validated rule for the current
  shadow router.
- Stock FVG entries made +26.6719R. The 18 OB entries made -2.4065R, but that
  sample is too small to override the separate cross-era valid-OB rejection.

## Tested explanations

- Thin liquidity is not the crypto cause in these exports. Median displayed
  depth was higher on losses (77,317 USD) than other outcomes (38,637 USD,
  p=0.0046). Spread had no association (p=0.785), and directional book imbalance
  was weak (p=0.174).
- The exported support/resistance proxies do not separate crypto losses:
  overhead distance p=0.835 and underfoot distance p=0.233.
- Event proximity is associated in this short window (median 5.45 hours on losses
  versus 14.85 hours otherwise, p=0.032), but the broader causal calendar audit
  did not transfer across eras. It is not promoted from this sample.
- The old language-model decision layer did not create crypto edge: its report
  shows 57% TP1 for approved setups versus 60% for rejected setups. The current
  shadow runtime correctly skips that decision layer.

## Decision

No new entry filter is promoted from this live window. The crypto rolling
two-entry hourly portfolio cap remains justified as a risk rail and is already
implemented and tested. The legacy stock export supports researching a
route-specific alignment gate and hourly cap, but neither can be installed in
the current router until it passes chronological and cross-era validation on
that exact route. No live deployment or order behavior is changed by this audit.
