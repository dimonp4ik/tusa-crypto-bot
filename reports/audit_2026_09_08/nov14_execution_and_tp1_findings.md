# Live incident follow-up, November 14

Crypto's public status endpoint reports build bc0066e1, live execution and 34 completed TP1 shadow observations: -4.22% baseline, -3.63% volume variant, -3.79% volume-plus-continuation variant. These are strategy diagnostics, not the client's exchange PnL.

Five successive public books for AVAX-USD_UM_XPERP-310530 showed spreads of approximately 1.3662, 5.4637, 1.3662, 1.3662 and 1.3666 basis points. The execution guard's actual limit is 5 basis points (0.05%). A single transient quote can therefore cause a refusal; these samples do not establish the reason for historical refusals.

The execution fix retries only spread-limit failures, up to three fresh books with 0.2 seconds between attempts. Each book still has to pass freshness, ordering, displayed depth and slippage checks. Signal bracket and deadline checks remain in the caller. A persistent rejection now reports measured spread and limit. No limit is relaxed, and no limit entry order is introduced.

The signal's theoretical TP1 fraction and the client's exchange TP1 setting are different fields. Real partial orders use autotrade_users.tp1_close_pct, whose default is zero. A nonzero percentage can additionally floor below lotSz/minSz. Neither a theoretical 5% signal bank nor a TP1 notification proves an actual partial fill.

The fraction sensitivity report compares 0/5/25/50/75% on fixed entry cohorts and unchanged runner exit times. It excludes lot rounding and partial-exit cash timing; it cannot approve a live policy. At 50%, historical closed-trade drawdown decreases versus 5%, but the nine-entry post-November-7 X-Perp cohort worsens from -1.4673% to -1.5857%. No client's TP1 setting is changed by this update.

Stocks has a separate live-readiness gate: the latest 20 completed production-route paper outcomes must have positive total R. If false, signals continue as paper while exchange entries are withheld. The deployed stock commit is successful, but its current readiness state and the client's actual skip reasons have not yet been retrieved. Do not label this gate as the confirmed cause of the client's week without entries until that evidence is available.
