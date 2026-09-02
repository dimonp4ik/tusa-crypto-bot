import os
from dotenv import load_dotenv

load_dotenv()

# --- Required secrets (set in Render environment variables) ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# --- Admin panel: Telegram user IDs that can access /admin in DM ---
ADMIN_IDS = {671071896}  # super-admin only; others added via bot → DB

# --- Scan settings ---
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))
TOP_COINS_COUNT = int(os.getenv("TOP_COINS_COUNT", "25"))
TIMEFRAME = "15m"          # 15m candle → swing signals, hold 2-8h
# Lookback windows MUST match backtest.py's WINDOW_15M/WINDOW_1H/WINDOW_4H
# (300/90/50) — see the KLINES_1H_LIMIT note below for what happened when they
# did not. Same number of API requests either way (OKX caps at 300/request).
KLINES_LIMIT = 300         # 300 × 15m = ~75 hours of data for SMC (= WINDOW_15M)

# --- Symbol quality filter ---
# ALLOWED_SYMBOLS="" (default) → auto top-volume mode, top 45 by 24h USDT volume.
# Bybit uses BTCUSDT format. BTC-USDT / BTC_USDT / BTC/USDT env values are
# accepted too and normalized at startup.
MIN_24H_QUOTE_VOLUME_USDT = float(os.getenv("MIN_24H_QUOTE_VOLUME_USDT", "5000000"))
MAX_SPREAD_PCT            = float(os.getenv("MAX_SPREAD_PCT", "0.20"))

def _parse_symbol_list(value, default=None):
    if not value:
        return list(default or [])
    return [s.strip().upper() for s in value.split(",") if s.strip()]

def _normalize_market_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace("-", "").replace("_", "").replace("/", "")

ALLOWED_SYMBOLS = [_normalize_market_symbol(s) for s in _parse_symbol_list(os.getenv("ALLOWED_SYMBOLS", ""))]
BLOCKED_SYMBOLS = [_normalize_market_symbol(s) for s in _parse_symbol_list(os.getenv("BLOCKED_SYMBOLS", ""))]
# Always block commodity derivatives — metals/indices follow macro drivers, not crypto SMC
_ALWAYS_BLOCKED = {"XAUUSDT", "XAGUSDT", "XAUTUSDT", "XBTUSDT"}
BLOCKED_SYMBOLS = list(set(BLOCKED_SYMBOLS) | _ALWAYS_BLOCKED)

# Stablecoins and fiat pairs — no trading signals
BLOCK_STABLE_BASES = {
    "USDC", "TUSD", "FDUSD", "DAI", "USDD", "USDP", "BUSD", "USTC",
    "EUR", "TRY", "BRL", "GBP", "JPY", "RUB", "UAH", "PYUSD", "USDE",
}
# Leveraged/synthetic tokens — unpredictable, not SMC-tradeable
LEVERAGED_TOKEN_SUFFIXES = ("3L", "3S", "2L", "2S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")

# --- Technical filter thresholds ---
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
VOLUME_SPIKE_MULTIPLIER = 1.8
MIN_SIGNALS_TO_PASS = 2

# --- Signal deduplication ---
# Readable from the environment since 2026-08-26: it was a plain constant, so
# every sweep of it silently re-ran the default and reported "no effect".
SIGNAL_COOLDOWN_HOURS = float(os.getenv("SIGNAL_COOLDOWN_HOURS", "3"))  # 15m swing signals hold 2-8h

# --- Signal expiry (no TP1/SL within this window → EXPIRED) ---
SIGNAL_EXPIRY_HOURS = int(os.getenv("SIGNAL_EXPIRY_HOURS", "48"))

# --- Market data (OKX since 2026-07-02; the _KUCOIN names are legacy labels
# for the timeframe strings, kept because they are threaded through the whole
# codebase — the data source itself is OKX). KUCOIN_BASE_URL was removed
# 2026-08-03: dead since the migration, nothing read it.
QUOTE_ASSET = "USDT"
TIMEFRAME_KUCOIN = "15min"
KLINES_INTERVAL_SEC = 15 * 60

# --- 1h candles for trend direction ---
TIMEFRAME_1H_KUCOIN = "1hour"
# 50 -> 90 (= backtest WINDOW_1H) on 2026-07-31. get_1h_trend() computes its
# "strong" flag (EMA9>EMA21>EMA50) only when len(closes) >= 51 — at 50 candles
# that branch NEVER RAN LIVE, so trend_1h_strong was permanently False in
# production while the backtest (90 candles) computed it for real. Consequences
# measured on two windows: 87.6% / 89.3% of ALL backtest trades carried the
# resulting Strong1h+1 mtf_score bonus that live could never award, and 238 /
# 233 of them sat exactly on the MTF_MIN_SCORE=14 gate — i.e. ~13% of backtest
# trades scored 13 live and were rejected outright. Live was effectively
# running a one-point stricter filter than every figure ever measured for it,
# and also never emitted the "StrongTrend1h" confirmation toward
# SMC_MIN_CONFIRMATIONS.
KLINES_1H_LIMIT = 90
KLINES_1H_INTERVAL_SEC = 3600

# --- 4h candles for higher timeframe bias ---
TIMEFRAME_4H_KUCOIN = "4hour"
# 30 -> 50 (= backtest WINDOW_4H). NB: 50 < 51, so trend_4h_strong stays False
# on BOTH sides — consistent, and confirmed dead in the data (Strong4h+1 appears
# on 0.0% of backtest trades in both windows). Deliberately matched rather than
# raised: going past 51 would enable a bonus live that the backtest still would
# not award, recreating the exact mismatch this change fixes. Enabling it is a
# strategy change and needs its own backtest, not a parity fix.
# 🔑 2026-08-28: at 50 the 4h "strong trend" flag can NEVER be set, because
# get_1h_trend() needs >= 51 closes to compute EMA50 — and it is used for the 4h
# series too. Verified in the book: "Strong4h" appears on 0 of 1167 trades while
# "Strong1h" appears on 85.7%. So the HTF_STRONG_SCORE bonus for 4h is dead
# code, and the Strong=1 vs Strong=0 comparison recorded beside HTF_STRONG_SCORE
# only ever tested the 1h side. This is the SAME defect that was found and fixed
# on the 1h side (KLINES_1H_LIMIT 50 -> 90); the 4h side was missed then.
# backtest.py's WINDOW_4H is 50 as well, so the two paths agree — no live/backtest
# gap, both are simply blind to it.
#
# Reviving it was measured (--window-4h 60 and 80) and is NOT shipped:
#   win4h  2023 tr/profit  ratios       2026 tr/profit  ratios
#    50    694 +143.59R   21.0/51.4     1192 +396.98R  80.3/216.4  <- kept
#    60    706 +142.89R   22.3/50.6     1199 +402.56R  76.2/224.9
#    80    706 +141.37R   21.4/48.3     1196 +397.94R  76.0/224.5
# +7 to +12 trades, profit flat (-0.5% / +1.4%), and the two risk measures split
# MIRRORED between windows: 2023 gains on worst-windows and loses on ulcer, 2026
# does the reverse. Same rule as HTF_STRONG_SCORE — when the risk measures split
# and profit does not move, the change is not supported.
# Note also a confound if anyone retries: raising the window changes BOTH whether
# `strong` can fire AND the EMA values themselves (more warmup), so the effect
# cannot be attributed. Isolating it needs the flag gated separately from the
# fetch size.
KLINES_4H_LIMIT = 50
KLINES_4H_INTERVAL_SEC = 4 * 3600

# --- 1D candles for macro trend ---
TIMEFRAME_1D_KUCOIN = "1d"
# 5 -> 8 to match the backtest's daily slice (aligned_slice_by_time(c1d,...,8)).
# _daily_trend only needs 3 closes so both sides already behaved the same, but
# matching removes one more place where the two can silently drift apart.
KLINES_1D_LIMIT = 8
KLINES_1D_INTERVAL_SEC = 86400

# --- Trading hours filter (UTC) — 24/7 since 2026-07-31 ------------------------
# Was Mon-Fri 07:00-21:00 UTC, i.e. only 70 of 168 hours (42% of the week), while
# backtest.py scans every bar. So 63% of every backtest figure came from hours
# the live bot never traded — and those hours turned out to be slightly BETTER,
# not worse. Measured with portfolio_sim.py on ONE shared account with all live
# guardrails applied, two independent ~6-month windows:
#   window 1  hours-only: 479tr  WR 81.6%  +221.6R  maxDD  -6.69R  (R/DD 33)
#   window 1       24/7: 1211tr  WR 81.8%  +595.1R  maxDD -11.17R  (R/DD 53)
#   window 2  hours-only: 452tr  WR 79.2%  +194.1R  maxDD -12.41R  (R/DD 16)
#   window 2       24/7: 1128tr  WR 80.2%  +479.1R  maxDD  -8.46R  (R/DD 57)
# Win rate is unchanged; profit is 2.5-2.7x, and on window 2 drawdown actually
# IMPROVED — more trades spread losses over more time, diluting the clusters.
# Better risk-adjusted on both windows, same direction, so not a single-window
# artefact.
# Checked before flipping: the X-Perp basis does NOT decouple off-hours
# (-0.111% vs -0.115%, same spread), so prices stay trustworthy. X-Perp VOLUME
# does roughly halve (BTC 19->9, ETH 193->116, SOL 1308->645 median), which the
# backtest cannot model — its slippage is a constant. Doubling off-hours
# slippage would eat only ~10% of the gain, so the conclusion holds.
# Env-overridable to revert without a redeploy.
TRADING_HOURS_START = int(os.getenv("TRADING_HOURS_START", "0"))
TRADING_HOURS_END   = int(os.getenv("TRADING_HOURS_END", "24"))
TRADE_WEEKENDS      = os.getenv("TRADE_WEEKENDS", "1") != "0"

# --- SMC settings ---
# Readable from the environment since 2026-08-26. It was a plain constant,
# so every sweep of it silently re-ran the default and reported "no effect"
# — and this is the parameter that DEFINES the structure everything else
# sits on: how many candles either side make a confirmed swing, and hence
# where BOS, FVG and order blocks are found at all.
# 5 -> 3 on 2026-08-26, and the story matters more than the number.
# This was a plain constant, so it had never been measured — like
# SIGNAL_COOLDOWN_HOURS and ATR_PERIOD, all three caught the same way: two
# runs with different values matching to the last decimal.
# It defines the STRUCTURE everything else sits on — how many candles either
# side confirm a swing, and hence where BOS, FVG and order blocks exist at
# all. Three days of tuning ran on top of an unmeasured foundation.
#
#   окно  глубина  сд   винрейт  прибыль   п/окна  п/ulcer
#   2023     3     543   66.9%   +54.2R      6.3     13.4
#   2023     5     483   66.0%   +33.7R      4.3      4.7
#   2024     3     731   72.9%  +124.7R     19.0     44.3
#   2024     5     664   72.1%  +107.1R     16.6     39.6
#   2025     3     841   76.6%  +220.2R     20.8     74.6
#   2025     5     748   75.9%  +205.8R     19.9     60.8
#   2026     3     981   76.5%  +336.6R     39.1    126.9
#   2026     5     901   76.8%  +342.1R    136.8    248.6
#
# 3 wins THREE historical windows on both measures, adds 60-93 trades in
# every one, and lifts the hostile 2023 window most (+61% profit, ulcer
# ratio 4.7 -> 13.4). It "loses" only on 2026 — where profit and win rate
# are flat and only the risk ratios drop, and those ratios are exactly what
# is anomalous there: the 2026 book scores worst-windows 2.50 against 3.5-8.6
# for every other configuration measured in three days. That single low
# number was silently rejecting changes — see memory drawdown-is-one-week.
SMC_SWING_LOOKBACK    = int(os.getenv("SMC_SWING_LOOKBACK", "3"))
# Both were HARDCODED until 2026-08-28, i.e. never swept. The same was true of
# SMC_SWING_LOOKBACK and EFF_RATIO_LOOKBACK, and both turned out to be worth
# 60-128 extra trades per window once they could actually be varied. These two
# control how many setups EXIST at all: the gap threshold decides what counts
# as an FVG, the lookback decides how far back order blocks are searched.
# 0.0005 -> 0.0003 on 2026-08-28, the first sweep this value has ever had.
#   thr       2023 profit/worst/ulcer   2024                    current
#   0.0008  624tr +109.82R 13.3/32.0    -                     1113tr +349.34R 61.5/178.2
#   0.0005  668tr +123.39R 15.2/37.2  872tr +177.84R 36.0/72.6 1167tr +377.46R 66.1/195.4
#   0.0003  694tr +136.43R 17.6/44.1  901tr +187.48R 40.5/82.9 1193tr +379.65R 72.4/198.4
#   0.0002  703tr +130.54R 16.8/40.5    -                     1199tr +384.66R 70.1/195.8
#   0.0001  714tr +130.63R 16.8/41.2    -                     1196tr +382.96R 69.8/192.9
#
# 0.0003 improves MORE TRADES, MORE PROFIT and BOTH risk measures in all three
# windows at once — the only change in two days of sweeps to do that. And the
# gain is not leverage: ABSOLUTE worst-windows falls too, 8.12 -> 7.75R,
# 4.94 -> 4.63R and 5.71 -> 5.25R. Strictly better on every axis.
#
# Below 0.0003 it turns: 0.0002 and 0.0001 add a few more trades and give back
# profit and risk, so the extra gaps down there are noise rather than
# structure.
#
# Reading: the threshold was set by eye when the detector was written and never
# touched, because it was hardcoded. Strictness looked safe and was not — the
# bot was failing to SEE real gaps, not filtering bad ones. Third find of that
# exact shape after SMC_SWING_LOOKBACK and EFF_RATIO_LOOKBACK, both also
# hardcoded, both also worth trades AND risk once they could be varied.
SMC_FVG_MIN_PCT       = float(os.getenv("SMC_FVG_MIN_PCT", "0.0003"))
# Swept 2026-08-28, its first ever. Unlike the FVG threshold next to it, this
# one was already right:
#   bars    2023 profit worst/ulcer   2024                  current
#    20   692tr +129.85R 16.8/41.1      -              1199tr +377.95R 72.1/197.8
#    30   694tr +136.43R 17.6/44.1  901tr +187.48R 40.5/82.9  1193tr +379.65R 72.4/198.4
#    45   711tr +141.58R 19.3/45.2  906tr +185.81R 49.7/79.6  1210tr +381.21R 72.6/181.3
#    60   722tr +131.62R 16.9/35.9      -              1229tr +382.48R 56.5/165.9
#
# 45 looks tempting — worst-windows improves in all three windows and by 22.7%
# in 2024 — but ulcer falls in two of them, 8.6% in the current one. The two
# measures disagree, which is the signature of noise rather than an edge, and
# 60 is plainly worse. Kept at 30.
#
# Worth noting against the FVG result above: two hardcoded recognition
# parameters, swept the same night. One was far too strict and worth trades AND
# risk; this one was already at its optimum. "Never swept" means unknown, not
# wrong.
SMC_OB_LOOKBACK       = int(os.getenv("SMC_OB_LOOKBACK", "30"))
# Minimum impulse after the order-block candle: price must travel this far over
# the next three candles for the candle to count as an order block at all.
# Hardcoded as 0.005 until 2026-08-28 and therefore never swept — the same
# situation as SMC_FVG_MIN_PCT next door, which turned out to be far too strict
# and was worth trades AND risk in every window. This one governs how many
# order blocks the bot SEES.
# Swept 2026-08-28, its first ever, and 0.005 is already the optimum:
#   impulse   2023 profit worst/ulcer      current
#   0.003   703tr +120.72R 17.0/41.1   1197tr +376.48R 63.6/175.3
#   0.005   694tr +136.43R 17.6/44.1   1193tr +379.65R 72.4/198.4  <- kept
#   0.008   691tr +127.63R 18.2/43.7   1190tr +363.06R 73.9/189.4
# Loosening and tightening both cost profit in both windows.
#
# Running tally for the hardcoded-number hunt, stated honestly: six swept,
# three changed (swing 5->3, eff-lookback 20->10, FVG 0.0005->0.0003), three
# already correct (cooldown, OB lookback, this one). Half is still the best
# hit rate of any approach tried, but "hardcoded" means UNTESTED, not wrong.
SMC_OB_MIN_IMPULSE    = float(os.getenv("SMC_OB_MIN_IMPULSE", "0.005"))
# Candle-shape thresholds, both hardcoded until now and both in the
# recognition path. WICK: what fraction of the candle must be wick for it to
# count as a rejection. BODY: what fraction must be body for a structure break
# to score as "strong" — that one feeds the MTF score and the adaptive packs,
# so it affects whether a setup passes the quality gate at all.
# BODY swept 2026-08-28, its first ever — 0.4 is already right:
#   frac    2023                        current
#   0.3   696tr +130.19R 18.0/44.7   1195tr +373.63R 72.7/198.2
#   0.4   694tr +136.43R 17.6/44.1   1193tr +379.65R 72.4/198.4  <- kept
#   0.5   689tr +134.47R 19.4/46.8   1191tr +375.65R 73.1/199.4
# Profit falls in BOTH directions. 0.5 buys ~6-10% on the 2023 risk ratios but
# costs trades and profit in both windows, and moves the current window only
# 0.5-1% — under the bar a rule has to clear to earn its maintenance surface.
# WICK deliberately NOT swept: it reaches the score through the same +1
# confirmation slot as the body (1 of 12) AND is diluted further by the `or`
# against bull_pressure, so its effect is bounded below the body's, which is
# already inside the noise. Made configurable so the next person need not
# re-derive that; the default is the value it has always had.
SMC_WICK_REJECT_FRAC  = float(os.getenv("SMC_WICK_REJECT_FRAC", "0.4"))
# How deep into an FVG price may already have travelled and still count as a
# valid entry zone. Hardcoded as a bare 0.80 in signal_filter.py until
# 2026-08-28, therefore never swept. Only bites when there is no OB — the
# selector prefers OB and falls back to FVG.
# Note which end it cuts: for a LONG, fill=0 is price at the TOP of the zone (a
# shallow retest) and fill=1 is at the BOTTOM (the deepest retest, right at
# invalidation). So this gate rejects the DEEPEST pullbacks, not the shallowest.
# Swept 2026-08-28, its first ever, and kept — the gate is nearly inert:
#              2023 tr/profit  ratios        2026 tr/profit  ratios
#   0.60       691 +143.24R   20.9/51.2      1184 +389.28R  84.2/215.2
#   0.80       694 +143.59R   21.0/51.4      1192 +396.98R  80.3/216.4  <- kept
#   0.95       698 +145.47R   21.1/52.9      1192 +396.98R  80.3/216.4
# It moves 3-8 trades out of 700-1200, under 1% of the book, because it only
# bites when a setup has NO order block (the selector prefers OB and falls back
# to FVG). At 0.95 the current window is identical to base to the last decimal
# — not the "knob never reached" failure, the env anchor was verified first and
# 2023 does move; there simply is no setup there with fill between 0.80 and
# 0.95 and no OB. Loosening is mildly positive in 2023 and free in 2026, but
# 1.3% in one window and nothing in another does not clear the bar.
SMC_FVG_MAX_FILL      = float(os.getenv("SMC_FVG_MAX_FILL", "0.80"))
# Bars used to find the swing high/low the structural stop is placed beyond.
# Hardcoded as a bare [-21:-1] in indicators.py until 2026-08-28 and therefore
# never swept, despite being the single number that sets stop distance — and so
# R, and so every figure in this project. Same family as SMC_SWING_LOOKBACK.
# Swept 2026-08-29, its first ever, and 20 stands:
#   bars   2023 net  worst/ulcer     2024 net  worst/ulcer    current  worst/ulcer
#    20   +142.50   14.8 / 34.8     +201.26   29.1 / 68.2   +465.37  33.8 / 151.2
#    30   +142.32   14.0 / 33.9          -                  +432.20  37.6 / 157.0
#    45   +147.66   15.0 / 41.4     +169.88   27.0 / 68.6   +381.03  32.4 / 128.6
# Trade count falls monotonically as the window lengthens (909 -> 889 -> 866 and
# 1570 -> 1503 -> 1451) because a wider swing reference means a wider stop, and
# a wider stop fails the risk gates more often. Profit follows it down in the
# current window. 45 wins the hostile window alone (+3.6% profit, +19% on the
# ulcer ratio) and loses the other two by 15.6% and 18.1%; 30 buys ~15% lower
# absolute drawdown in the current window for 7.1% of its profit, and is flat to
# slightly worse in the hostile one. Neither is an improvement — both are the
# same profit-for-drawdown trade the account owner already has better handles for.
#
# Method note: this sweep was nearly wasted. The 45-bar runs were started while
# mistaking the SMC_OB_LOOKBACK table above for this parameter's, which would
# have compared 45 against a value this parameter has never held. The config
# assertion in the shipping step caught it before anything was written.
SL_REF_LOOKBACK       = int(os.getenv("SL_REF_LOOKBACK", "20"))
# How far outside a zone price may sit and still count as "at" it. Four bare
# numbers in indicators.py until 2026-08-28. They are deliberately MIRRORED —
# wide on the side price approaches from, tight on the far side, so "coming to
# the zone" counts and "blown through it" does not — but the FVG and OB pairs
# differ (0.1%/1.0% vs 0.2%/0.5%) with nothing on record saying why.
# NEAR = far side (price must not have passed through), APPROACH = the side
# price comes from. These affect DETECTION only; entry still waits for price to
# return into the zone, so the stale-entry guard is untouched.
SMC_FVG_NEAR_TOL      = float(os.getenv("SMC_FVG_NEAR_TOL", "0.001"))
SMC_FVG_APPROACH_TOL  = float(os.getenv("SMC_FVG_APPROACH_TOL", "0.01"))
# Swept 2026-08-28, first ever. 0.01 KEPT.
#   tol     2023 tr/profit  ratios        2024 tr/profit  ratios       2026 tr/profit  ratios
#   0.005   640 +126.82R   18.1/43.3                                   1075 +355.28R  48.2/168.3
#   0.010   694 +143.59R   21.0/51.4      901 +190.15R  42.3/88.9      1192 +396.98R  80.3/216.4  <- kept
#   0.015   702 +138.22R   20.2/45.0                                   1249 +419.69R  81.2/225.2
#   0.020   702 +139.57R   20.4/45.5      918 +186.22R  40.6/82.4      1246 +416.50R  84.1/224.6
# Tightening to 0.005 (matching the OB pair) is worse everywhere, so the
# FVG/OB difference is not an accident even though nothing recorded why.
# Widening is the biggest trade-count gain found all session (+54 in the current
# window, +4.9% profit, both risk ratios up) and it is STILL rejected: 2023 and
# 2024 both lose on profit AND on both risk measures. Two windows against one,
# and the one in favour is the one most likely to be regime-specific.
SMC_OB_NEAR_TOL       = float(os.getenv("SMC_OB_NEAR_TOL", "0.002"))
SMC_OB_APPROACH_TOL   = float(os.getenv("SMC_OB_APPROACH_TOL", "0.005"))
# Swept 2026-08-28, its first ever. 20 KEPT — but read the risk column carefully:
#   bars   2023 tr/profit  ratios       2026 tr/profit  worst  ratios
#    10    678 +136.22R   23.2/47.9     1193 +396.65R  10.13  39.2/179.2
#    15    678 +134.82R   21.1/46.9     1199 +401.17R   9.71  41.3/193.4
#    20    694 +143.59R   21.0/51.4     1192 +396.98R   4.95  80.3/216.4  <- kept
#    30    686 +128.34R   20.8/46.3     1178 +379.05R   4.44  85.3/230.7
# 20 wins on profit in BOTH windows, which is the whole case for keeping it.
#
# ⚠️ Do NOT read the 2x drawdown gap between 15 and 20 in the current window as
# evidence. Checked it three ways and it does not survive:
#  - In MONEY per unit of risk the two are identical: +0.3376 vs +0.3370. The
#    +1.1% profit edge 15 shows in R is the unit, not the result — a tighter
#    stop means less money in the trade, and R does not know that.
#  - The loss tails are the same shape: worst trade -4.15R vs -3.92R, 22 vs 21
#    trades below -2R.
#  - The 2023 window shows no gap at all (6.40 vs 6.84).
# Same per-trade returns, same tails, no gap in the other window: the current
# window's gap is where seven trades landed in the sequence, not a property of
# the parameter. Drawdown is made by the SERIES here, so a metric built on the
# five deepest 25-trade stretches moves on reshuffles.
#
# Useful side measurement: at 20 bars, 22.1% of the book carries a stop wider
# than 3.33%, which is where RISK_NORMALIZED_SIZING hits its floor and live
# risks MORE money than the flat-R accounting assumes. Aggregate effect ~1.2%.
SMC_STRONG_BODY_FRAC  = float(os.getenv("SMC_STRONG_BODY_FRAC", "0.4"))
SMC_MIN_CONFIRMATIONS = int(os.getenv("SMC_MIN_CONFIRMATIONS", "2"))
# 2026-08-28: 1.5 -> 1.4. This gate had no recorded justification and had never
# been swept, while binding HARD — the minimum volume_ratio in the whole book was
# exactly 1.500, i.e. every trade sat on it.
#   min   2023 tr/WR/profit  ratios      2024 tr/WR/profit  ratios      2026 tr/WR/profit  ratios
#   1.3   799 70.1% +165.41R 23.8/64.8   1016 73.8% +225.90R 60.2/112.6  1332 75.9% +426.47R 72.2/196.8
#   1.4   737 69.6% +148.46R 23.2/53.7    955 73.8% +206.87R 74.5/102.3  1266 76.0% +416.45R 87.2/230.9  <- shipped
#   1.5   694 69.5% +143.59R 21.0/51.4    901 73.7% +190.15R 42.3/88.9   1192 75.6% +396.98R 80.3/216.4  <- was
#   1.8   564 70.6% +123.32R 18.9/40.3                                    947 74.4% +300.23R 29.3/108.2
#
# 1.4 beats the old value on EVERY measure in ALL THREE windows: +43/+54/+74
# trades, +3.4/+8.8/+4.9% profit, win rate up in each, and both risk ratios up in
# each. 1.3 adds more trades and more profit still but gives up ~10% of both risk
# ratios in the current window, so the optimum sits between and is not at an edge.
#
# Caveat on one number: the 2024 worst-windows ratio moves +76%, and that is the
# noisiest measure here — a false 2x gap in it was caught the same night on
# SL_REF_LOOKBACK. The case does not rest on it: ulcer moves +15.1% in that
# window, and profit, trades and win rate all move too.
#
# Reading: profit PER TRADE is unchanged by the loosening (2023 +0.207 before and
# after; 2026 +0.333 -> +0.320), so the gate was not separating good setups from
# bad — it was just cutting on volume. Same family as the swing/eff/FVG finds:
# the bot was not selecting badly, it could not SEE the structure.
#
# ⚠️ MEASURED, after two wrong write-ups of this same paragraph. The threshold
# also feeds the score's +2 volume tier, max(threshold * 1.35, 2.0), which IS
# 2.025 at gate 1.5 and 2.000 at 1.4 and 1.3 — so the shipped step moves the
# tier as well as the gate. But moving the tier changes NOTHING, measured by
# pinning it with SMC_VOL_STRONG_TIER and running the arms apart:
#   gate 1.5, tier 2.025 (base)   2023  694tr +143.59R 21.0/51.4 | 2026 1192tr +396.98R 80.3/216.4
#   gate 1.5, tier 2.000          2023  694tr +143.59R 21.0/51.4 | 2026 1192tr +396.98R 80.3/216.4
#   gate 1.4, tier 2.025 (frozen) 2023  737tr +148.46R 23.2/53.7
#   gate 1.4, tier 2.000 (shipped)2023  737tr +148.46R 23.2/53.7
# The tier arm is identical to base to the last decimal in both windows, and the
# frozen-tier gate arm is identical to the shipped one. The 2.000-2.025 band is
# 1.2% wide and simply contains no setup. This is a real null, not the "knob
# never arrived" failure — SMC_VOL_STRONG_TIER was anchor-checked first.
#
# So: the gate is the entire effect, and the first write-up ("the only change is
# which setups exist") was right in substance while its reasoning was wrong.
# What IS clean and worth keeping: 1.4 -> 1.3 leaves the tier untouched either
# way, and that pure gate step comes out mixed — +62/+61/+66 trades and
# +11.4/+9.2/+2.4% profit, but worst-windows ratios +2.6/-19.2/-17.2%. Loosening
# adds trades of diminishing quality; the first increment earns its place, the
# second does not.
#
# ⚠️ STILL OPEN: the stocks desk lands on 1.3 and shows 1.4 as a DIP, worse than
# BOTH neighbours, in nested windows and in two disjoint slices. Diminishing
# returns cannot make a dip, and the tier confound is now ruled out as the cause
# here — so whatever produces it there is still unidentified. Measure per desk.
# ⚖️ SIGNIFICANCE 2026-08-28: NOT significant on its own, and that is expected
# rather than damning. Bootstrap on the current window, 1.5 vs 1.4:
#   delta net +19.48R, delta R/trade -0.004, p_gt_zero 0.636,
#   90% CI -72.55 to +110.42 — straddles zero widely.
# Arithmetic agrees the test simply lacks power here: per-trade R scatter is ~1.0
# over ~1200 trades, so the standard error on total profit is ~35R and a +19.5R
# effect is 0.56 of one. Detecting it on one window would need roughly +70R.
# It also cannot be paired — the change alters WHICH trades exist, so only 1128
# rows matched and the remainder carries the variance. Compare TRAIL_ATR_MULT,
# which IS paired and comes back p=1.0.
# What this setting rests on instead: the same sign in three independent windows
# (2023/2024/2026 do not overlap at 18000 candles), an understood mechanism
# (profit per trade unchanged, so the gate was cutting on volume rather than
# selecting), and walk-forward showing no out-of-sample decay. That is a
# reasonable basis for a config value; it is NOT proof, and should not be
# described as one.
# 1.4 -> 1.3 on 2026-08-29. The rejection recorded above ("worst-windows ratios
# +2.6/-19.2/-17.2%") was measured while the kill-switch replay peeked at future
# outcomes, so those ratios were three to four times too kind and unevenly so.
# Re-swept honestly. The parameter's only effect in this range is the setup
# filter in signal_filter.py (volume_ratio < X -> reject); the volume TIER that
# also reads it is max(X*1.35, 2.0), which pins at 2.0 throughout. So one export
# at the loosest value, filtered down, reproduces every threshold exactly —
# verified against the 1.4 export, whose minimum volume_ratio is 1.400 with
# nothing below it.
#   thr    2023 net  worst/ulcer     2024 net  worst/ulcer    current  worst/ulcer
#   1.60    +93.46    8.4 / 18.7     +167.67   25.2 / 54.7   +339.09  26.3 /  89.1
#   1.50   +107.02   10.1 / 23.3     +171.76   25.8 / 51.1   +374.45  23.7 /  91.3
#   1.40   +112.20   12.1 / 24.8     +180.76   27.0 / 59.0   +419.62  29.1 / 126.6
#   1.35   +118.53   13.4 / 27.2     +195.18   29.5 / 69.2   +429.69  29.8 / 127.1
#   1.30   +133.16   13.9 / 32.2     +204.48   26.1 / 69.3   +433.47  31.5 / 126.2
# The gradient runs the other way from the old conclusion: every measure improves
# as this loosens, and it improves MOST in the hostile window (+18.7% profit with
# absolute ulcer FALLING, 4.53 -> 4.13). 1.35 is better than 1.4 on all six
# numbers in all three windows; 1.30 earns more (+8.2% profit and +6.9% trades in
# total) and is better on five of six, the exception being 2024 worst-windows at
# -3.3%. Taking 1.30 — and note the stocks desk arrived at 1.3 independently, so
# the long-standing disagreement between the two desks on this parameter was an
# artefact of the peeking gate here.
#
# Below 1.30 is unexplored on the historical windows. The current window keeps
# improving down to 1.20 (+9.2% profit, both ratios better than 1.4), but a gate
# must not be loosened past what the hostile window has been asked about.
SMC_BOS_MIN_VOLUME    = float(os.getenv("SMC_BOS_MIN_VOLUME", "1.3"))

# Research handle, default OFF (0 = compute as before). The +2 volume score tier
# is normally max(SMC_BOS_MIN_VOLUME * 1.35, 2.0), so it MOVES whenever the gate
# moves and every gate sweep is impure by construction. Set this to a fixed
# number to pin the tier while the gate moves. Used 2026-08-28 to prove the tier
# arm is inert (see the block above); left in place so the next gate sweep can
# be run clean without re-deriving the trick.
SMC_VOL_STRONG_TIER   = float(os.getenv("SMC_VOL_STRONG_TIER", "0"))
SMC_RSI_LONG_MAX      = float(os.getenv("SMC_RSI_LONG_MAX", "72"))   # skip overextended longs
SMC_RSI_SHORT_MIN     = float(os.getenv("SMC_RSI_SHORT_MIN", "28"))  # skip overextended shorts
MAX_SETUPS_TO_CLAUDE  = int(os.getenv("MAX_SETUPS_TO_CLAUDE", "7"))  # only strongest go to Claude

# --- Entry zone (FVG / Order Block) ---
# When enabled, setups without an active FVG or OB zone near price are skipped.
REQUIRE_ENTRY_ZONE       = os.getenv("REQUIRE_ENTRY_ZONE", "1") != "0"
ENTRY_ZONE_SL_BUFFER_ATR = float(os.getenv("ENTRY_ZONE_SL_BUFFER_ATR", "0.25"))

# --- Regime / retest filters (cut chop + false breakouts) ---
# REQUIRE_HTF_TREND : reject when both 1h AND 4h are neutral (no real trend = chop).
# REQUIRE_RETEST    : price must currently sit at/near the entry zone (true retest),
#                     not a far-away limit order that the backtest fills optimistically.
REQUIRE_HTF_TREND   = os.getenv("REQUIRE_HTF_TREND", "1") != "0"
REQUIRE_RETEST      = os.getenv("REQUIRE_RETEST", "1") != "0"
RETEST_MAX_DIST_PCT = float(os.getenv("RETEST_MAX_DIST_PCT", "0.015"))  # within 1.5% of zone edge

# --- Multi-timeframe score gate (max ~15) ---
# 2026-06-11 A/B (20 sym, 2880+5760×15m, trail): scores 12-13 = WR ~20%, -6.3R.
# Raising 10→14 cut those: WR 48.9→50.7%, R/tr +17%, DD -25% on both windows.
# 14 -> 13 on 2026-08-24. The 2026-06-11 note above measured the 12-13 band at
# WR ~20% / -6.3R — under the OLD backtest, which filled at the zone midpoint
# without checking the market ever traded there. Re-measured on the honest
# execution model, the band that gate was discarding is BETTER than the book it
# was protecting: score exactly 13 runs 154 trades at 76.6% WR and +0.224R,
# against a base of 73.8% and +0.204R, and it holds on both halves
# (+0.253/+0.195).
#
# Effect of the change alone, live gates applied:
#   14: 767 сд  73.8%  +170.73R  DD -15.52R  ratio 11.0
#   13: 843 сд  74.3%  +190.22R  DD -14.38R  ratio 13.2
# More trades, higher win rate, more profit, less drawdown — no trade-off.
# 12 was also tested and adds little beyond 13, so the gate stops here.
#
# RE-CHECKED 2026-09-03 on honest costs (fees were 5x too high until today) and
# 13 SURVIVES here, even though the same re-check moved the stocks desk to 11.
#            trades      net R      maxDD   worst-win ratio   profit/ulcer
#   2026-08-26  1570/1597  597.61/604.48  -14.86/-14.86   48.6/49.1   252.8/251.2
#   2024-07-31  1175/1187  287.90/295.07   -8.67/ -8.34   54.7/55.2   130.7/134.7
#   2023-07-31   908/ 938  223.18/220.10   -9.24/-11.81   30.6/21.9    83.6/ 72.7
# (left = 13, right = 11.) The two calm windows favour 11 slightly, and the
# HOSTILE one rejects it outright: the 30 extra trades earn nothing while max
# drawdown worsens 28% and both risk ratios fall by a quarter and a seventh.
#
# The mechanism explains why the desks disagree rather than one of them being
# wrong. In stocks the quota binds harder than setup quality, so lowering the
# bar only reshuffles WHICH setups fill the same slots — trade count barely
# moves and only the refusal count climbs. Here the quota has room, so a lower
# bar admits genuinely weaker setups, and a hostile tape is where that shows.
# Rule this confirms: a boost is only earned if the subset does not weaken in
# the hostile window.
MTF_MIN_SCORE = int(os.getenv("MTF_MIN_SCORE", "13"))

# INERT since 2026-08-07 — variant D was removed on request, and it was the only
# consumer. signal_filter.py no longer calls _soft_fail("score"), so nothing
# scoring below MTF_MIN_SCORE survives and no score-shadow setups reach Claude.
# Kept only so reviving D is a one-line change there (see filter_variants.py,
# "Slot D is FREE"). Was: setups in [SHADOW_MIN_SCORE, MTF_MIN_SCORE) were never
# real signals but were sent to Claude + logged so arm D got real verdicts.
SHADOW_MIN_SCORE = int(os.getenv("SHADOW_MIN_SCORE", "12"))

# --- Signal-quality filters (backtested on a PINNED 20-coin / ~21-day set) ---
# №1 Volatility regime — DEFAULT ON after re-test 2026-06-05 on full context
#    momentum stack: +1.60R net, non-negative on all monthly slices, better MC p05.
#    Upper ceiling still OFF (hurt R in backtest — cuts TP2 runners).
VOL_REGIME_FILTER = os.getenv("VOL_REGIME_FILTER", "1") != "0"
VOL_MIN_ATR_PCT   = float(os.getenv("VOL_MIN_ATR_PCT", "0.0015"))  # <0.15% range = too dead
VOL_MIN_RATIO     = float(os.getenv("VOL_MIN_RATIO", "0.55"))      # cur/median below = collapsed
VOL_MAX_RATIO     = float(os.getenv("VOL_MAX_RATIO", "99"))        # ceiling OFF (hurt R in backtest)
# Median window for the vol-regime ratio. Was declared but never passed to
# volatility_regime(), which silently used its own default — same number, so
# behaviour is unchanged, but the knob did nothing. Wired 2026-08-03.
VOL_REGIME_LOOKBACK = int(os.getenv("VOL_REGIME_LOOKBACK", "50"))

# №3 Strong BOS and №4 Structural-only confirmation were BOTH backtested and
# DROPPED (default off): each lowered win rate (37.5% → 35.0%) and Expected R
# (+0.12R → +0.03R). Strong-BOS pushed entries late (momentum spent → SL);
# structural-only cut valid reversals. Flags kept for experimentation.
REQUIRE_STRONG_BOS = os.getenv("REQUIRE_STRONG_BOS", "0") != "0"
STRONG_BOS_VOL_MULT = float(os.getenv("STRONG_BOS_VOL_MULT", "1.3"))  # x SMC_BOS_MIN_VOLUME
REQUIRE_STRONG_CONFIRM  = os.getenv("REQUIRE_STRONG_CONFIRM", "0") != "0"
MACD_CHOCH_NOISE_FILTER = os.getenv("MACD_CHOCH_NOISE_FILTER", "0") != "0"
# 2026-06-05 A/B, 8640×15m: +9.39R net, +0.5pp WR, better portfolio guard and Monte Carlo.
# Skips only overlap-session setups when 1h trend is bearish (latecomers get squeezed at NYC open).
# DISABLED 2026-08-24. This guard only ever mattered once STABILITY_FILTERS was
# switched off and the OVERLAP session came back — while that session was cut
# entirely, the guard was watching an empty room, and an earlier sweep of it
# reported "no change" for exactly that reason. Re-measured after the session
# returned, it is actively costing money:
#   on:  921 сд  74.8%  +230.87R  DD -9.25R  ratio 25.0
#   off: 947 сд  75.1%  +243.79R  DD -9.25R  ratio 26.4
# Its own note claims +9.39R / +0.5pp WR, measured on the old fill model.
OVERLAP_BEARISH_1H_GUARD = os.getenv("OVERLAP_BEARISH_1H_GUARD", "0") != "0"

# 1D macro trend filter — skip LONG when daily candle trend is BEARISH.
# Prevents buying into a day-scale downtrend (as happened with sideways/red daily days).
DAILY_TREND_FILTER = os.getenv("DAILY_TREND_FILTER", "1") != "0"

# Double-neutral LONG block — skip LONG when BOTH 4h AND 1D are NEUTRAL.
# Two-TF neutrals = sideways/chop at macro level; longs get chopped out by range boundaries.
DOUBLE_NEUTRAL_LONG_FILTER = os.getenv("DOUBLE_NEUTRAL_LONG_FILTER", "1") != "0"

# Daily SHORT guard — mirror of DAILY_TREND_FILTER for shorts.
# Skip SHORT when daily trend is BULLISH — don't short into a day-scale uptrend.
DAILY_TREND_SHORT_FILTER = os.getenv("DAILY_TREND_SHORT_FILTER", "1") != "0"

# №A Efficiency-Ratio chop filter — DEFAULT ON (backtest-proven winner).
#    Kaufman ER over EFF_RATIO_LOOKBACK bars: ER~1 = clean trend, ER~0 = chop.
#    Skip setup if ER < EFF_RATIO_MIN. Targets the proven loss source: false BOS
#    in ranges (LINK 2W/26SL, SOL 6W/19SL). Distinct from ATR-vol (size) — ER
#    measures DIRECTION. Backtest (pinned 20 symbols, ~21d 15m), threshold sweep:
#       base 430tr 36.7% +0.08R/+33R | 0.10 341tr +0.11R/+38R | 0.12 323tr +0.12R/+39R
#       0.15 293tr 37.2% +0.14R/+41R (PEAK) | 0.20 245tr +0.13R/+31R | 0.30 151tr +0.13R/+20R
#    0.15 = clean unimodal peak: beats baseline on win%, R/trade AND total R while
#    cutting 32% junk trades. First filter to beat baseline on every axis.
#
# 🔑 RE-VALIDATED 2026-08-28. That sweep ran on 430 trades over 21 days at 36.7%
# win rate — a system with half today's. Re-measured across three windows,
# both as on/off and as a threshold:
#
#   thr     2023 profit/worst    2024 profit/worst    current profit/worst
#   0.15   +123.39R  15.2       +177.84R  36.0       +377.46R  66.1   <- kept
#   0.12   +123.86R  10.9       +184.59R  37.3       +394.88R  74.6
#   0.10   +128.10R  11.2       +183.79R  37.2       +394.51R  65.9
#   off    +146.89R  12.5       +183.92R  22.0       +422.51R  52.4
#
# Loosening pays in two windows — the current one by +12.9% on worst-windows —
# and the hostile one goes from 8.12R of absolute worst-windows to 11.41R, a
# 40% risk increase for no extra profit. Both 0.12 and 0.10 land on exactly
# 11.41R, so a specific cluster enters between 0.12 and 0.15 and it is
# expensive in a bad market.
#
# So 0.15 is not an arbitrary number from an obsolete sweep: it is protecting
# the hostile regime specifically. Dead justification, sound setting — the same
# as the TP1 history. A dead justification is a reason to re-measure, not a
# reason to change.
#
# The other two legacy filters were checked at the same time and also hold:
# turning off BEAR_TREND_HOT_VOL_GUARD is near-neutral (+2-5% trades, profit
# within 2%) but degrades both hostile-window measures; DIRECTIONAL_RSI_MIDLINE
# adds 6-9% of trades and loses 31% at equal risk. All three earn their keep.
EFF_RATIO_FILTER   = os.getenv("EFF_RATIO_FILTER", "1") != "0"
# 20 -> 10 on 2026-08-26. How many bars the Kaufman efficiency ratio looks
# back over — i.e. over what span "is this a clean move or chop" is judged.
#
#   окно  знач  сд   винрейт  прибыль   п/окна  п/ulcer
#   2023   10   622   68.3%    +76.0R     8.2     19.5
#   2023   20   543   66.9%    +54.2R     6.3     13.4
#   2024   10   820   74.0%   +164.0R    26.2     61.3
#   2024   20   731   72.9%   +124.7R    19.0     44.3
#   2026   10  1110   75.5%   +344.5R    34.4    119.6
#   2026   20   982   76.6%   +338.7R    39.3    127.8
#
# Wins both historical windows on both measures with a higher win rate, and
# adds 79-128 trades in every window — profit +40% on the hostile 2023 one.
# The current window objects only on the risk ratios while profit still
# rises; same shape as SMC_SWING_LOOKBACK, and the same verdict.
# Reading: a 20-bar window (5 hours on 15m) is too slow to describe the move
# a zone retest is about to join — by then the chop it measures is history.
EFF_RATIO_LOOKBACK = int(os.getenv("EFF_RATIO_LOOKBACK", "10"))
# 0.15 -> 0.12 on 2026-08-29. The 2026-08-28 re-validation above kept 0.15 on
# the grounds that loosening took the hostile window "from 8.12R of absolute
# worst-windows to 11.41R, a 40% risk increase for no extra profit". That was
# measured while the kill-switch replay peeked at future outcomes. Re-swept
# honestly on the post-1.3-volume book — this gate is a single reject line on
# ind["eff_ratio"], so one export with it open reproduces every threshold
# exactly (tools_filter.py); the 0.15 row reproduces the shipped book to the
# decimal in all three windows, which anchors it.
#   thr     2023 net  worst/ulcer     2024 net  worst/ulcer   current  worst/ulcer
#   off    +146.84   13.1 / 33.3     +216.11   31.5 / 72.9   +482.76  35.1 / 149.3
#   0.10   +144.20   14.1 / 36.1     +191.91   26.7 / 63.7   +461.61  33.6 / 148.1
#   0.12   +142.50   14.8 / 34.8     +201.26   29.1 / 68.2   +465.37  33.8 / 151.2
#   0.15   +133.16   13.9 / 32.2     +204.48   26.1 / 69.3   +433.47  31.5 / 126.2
#   0.18   +137.68   13.9 / 33.7     +207.97   24.2 / 71.2   +406.89  39.4 / 123.2
# The feared risk increase is 9.60 -> 9.61R of absolute worst-windows in the
# hostile window, not 8.12 -> 11.41. 0.12 improves the worst-windows RATIO in
# ALL THREE windows and the ulcer ratio in two, for +82 trades and +4.9% profit.
# The 2024 profit dip of -3.2R is inside that window's own noise: 0.10 reads
# 191.91 between 0.08's 196.07 and 0.12's 201.26, so ~10R of scatter lives there.
#
# Turning it OFF earns more still (+9.7% profit, +268 trades, better ulcer ratio
# in all three) but costs the hostile window 5.8% of its worst-windows ratio and
# removes the chop protection entirely. Recorded, not taken: a filter in place
# since the beginning should not be deleted on the strength of one sweep.
EFF_RATIO_MIN      = float(os.getenv("EFF_RATIO_MIN", "0.12"))
# Upper bound on trend cleanliness. Measured 2026-08-24 on the honest model:
# the top eff_ratio quartile (>=0.365) runs +0.147R vs +0.244R base, consistent
# across both halves. Reading: a very clean trend at entry means the move is
# already extended, so the retest we are buying is late. 0 = off.
EFF_RATIO_MAX      = float(os.getenv("EFF_RATIO_MAX", "0"))
# MTF scoring weights for higher-timeframe trend. The original scheme rewarded
# alignment (+2, plus +1 when the EMA stack confirms) over a neutral HTF (+1),
# but the outcome data runs the other way: trend_4h=neutral scores 84.8% WR /
# +0.365R and trend_1h=neutral 79.8% / +0.357R, both stable across halves.
HTF_ALIGNED_SCORE  = int(os.getenv("HTF_ALIGNED_SCORE", "2"))
HTF_NEUTRAL_SCORE  = int(os.getenv("HTF_NEUTRAL_SCORE", "1"))
# REVERTED to 1 on 2026-08-25. Setting it to 0 shipped earlier the same day
# on a +4.5% equal-risk gain measured against MAX drawdown — which turned
# out to be fifteen trades in one April week. Re-checked against measures
# that need several bad patches to move, the two disagree symmetrically:
#   Strong=1  966 сд  +339.1R  худ.окна 4.49 (75.6)  ulcer 1.91 (177.6)
#   Strong=0  922 сд  +341.9R  худ.окна 4.97 (68.8)  ulcer 1.75 (195.1)
# Profit is a wash (0.8%) and Strong=0 costs 44 trades for it. When the
# risk measures split and the profit does not move, the change is not
# supported — keep the version with more trades.
HTF_STRONG_SCORE   = int(os.getenv("HTF_STRONG_SCORE", "1"))
# Premium/Discount structure gate — "discount" only counts as a buy signal inside
# a bullish/neutral dealing range, "premium" only inside a bearish/neutral one. In
# a clean lower-high+lower-low down-structure, price below the range midpoint is
# mid-decline, not cheap — without this a LONG into descending swings wrongly got
# a "Discount" confirmation (the 16.06 XRP loss). Set PD_TREND_GATE=0 to disable.
PD_TREND_GATE      = os.getenv("PD_TREND_GATE", "1") != "0"
# №B Strict HTF alignment — DROPPED (default off). Backtested: 232tr +0.04R/+8R,
#    half of baseline. Cutting counter-trend also cut winners. Flag kept for experiments.
REQUIRE_STRICT_HTF = os.getenv("REQUIRE_STRICT_HTF", "0") != "0"

# --- Asymmetric bear-squeeze guard (DEFAULT ON) --------------------------------
# In crypto, full-HTF bearish shorts (BOS + 1h + 4h all bearish) with overheated
# volume attract crowded late entries → market-makers squeeze them upward.
# Skip SHORT when: bos=bearish AND trend_1h=bearish AND trend_4h=bearish AND
# vol_ratio_regime >= threshold (2.5 = 2.5× normal volume = overheated).
# Also skip "LONDON" session for full-bearish shorts (expansion attracts latecomers,
# then NYSE open reverses them).
# A/B backtest, 20 symbols, 8640×15m (~3 months), trail exit:
#   base:  2646tr  38.1% WR  +0.118R/tr  DD -68.17R
#   guard: 2344tr  39.6% WR  +0.150R/tr  DD -47.36R  (+27% R/tr, -31% DD)
# Re-measured 2026-08-29 together with the RSI midline filter below. Both are
# justified on a bot that won 39.6-40.1% against this book's ~70%, and the
# STOCKS desk had to switch this same guard OFF the same night (it cost profit
# in all five windows there — the squeeze mechanism is crypto microstructure and
# does not exist in equities). Here it EARNS its place, which is the point.
#
# Both are pure setup rejects on columns the export already carries, so one run
# with both OFF reconstructs every combination exactly. Anchored each time: the
# "both on" row reproduced the live book to the decimal in all three windows.
#
#                     2023 net  worst/ulcer    2024 net  worst/ulcer   current  worst/ulcer
#   both off          +135.56   11.1 / 20.2    +220.46   21.6 / 58.3   +540.13  53.1 / 157.0
#   bear only         +140.47   14.3 / 31.9    +218.86   23.4 / 65.3   +534.76  52.0 / 162.6
#   rsi only          +143.68   13.6 / 31.5    +199.76   24.6 / 59.6   +474.33  32.5 / 154.2
#   both on (kept)    +142.50   14.8 / 34.8    +201.26   29.1 / 68.2   +465.37  33.8 / 151.2
#
# Dropping the bear guard costs ratios in every window — keep it. Dropping the
# RSI midline is the arguable one: +10.5% profit and +374 trades in total, but
# the risk ratios fall in two windows of three INCLUDING the hostile one, and
# only the current window prefers it. Two of three, hostile among them, says
# keep. Recorded because that is a real profit figure being left on the table
# for a risk reason, and the owner may want it.
BEAR_TREND_HOT_VOL_GUARD     = os.getenv("BEAR_TREND_HOT_VOL_GUARD", "1") != "0"
BEAR_TREND_HOT_VOL_MIN_RATIO = float(os.getenv("BEAR_TREND_HOT_VOL_MIN_RATIO", "2.5"))
BEAR_TREND_SKIP_SESSIONS     = set(_parse_symbol_list(os.getenv("BEAR_TREND_SKIP_SESSIONS", "LONDON")))

# --- Directional RSI midline confirmation (DEFAULT ON) ------------------------
# A BOS without RSI reclaiming the 50 midline (LONG) or dropping below 40
# (SHORT) = structural break without momentum confirmation → higher false-break
# rate. Distinct from the overextension caps (SMC_RSI_LONG_MAX / SHORT_MIN).
# A/B backtest, on top of bear-trend guard, same 20 symbols × 8640×15m:
#   guard:    2344tr  39.6% WR  +0.150R/tr  DD -47.36R
#   +RSI mid: 2117tr  40.1% WR  +0.175R/tr  DD -37.38R  (+17% R/tr, -21% DD)
DIRECTIONAL_RSI_MIDLINE_FILTER = os.getenv("DIRECTIONAL_RSI_MIDLINE_FILTER", "1") != "0"
RSI_LONG_MIN_MIDLINE           = float(os.getenv("RSI_LONG_MIN_MIDLINE", "42"))  # lowered 50→42: catches zone entry earlier, same WR/R (+3 trades, +4R on 8640-bar test)
RSI_SHORT_MAX_MIDLINE          = float(os.getenv("RSI_SHORT_MAX_MIDLINE", "40"))

# --- Per-symbol / per-source / per-direction edge filters (DEFAULT ON) ----------
# Populated from loss_taxonomy analysis. Skip instruments/direction combos that
# repeatedly showed poor edge after enough backtest data.
SYMBOL_EDGE_FILTER  = os.getenv("SYMBOL_EDGE_FILTER", "1") != "0"
LOW_EDGE_SYMBOLS    = _parse_symbol_list(os.getenv("LOW_EDGE_SYMBOLS", "XMR-USDT,XMRUSDT"))

# 2026-06-05 A/B, 8640×15m: skipping NEAR FVG entries improved raw net (+9.99R),
# WR/R-trade, and Monte Carlo while keeping trade count high.
SOURCE_EDGE_FILTER     = os.getenv("SOURCE_EDGE_FILTER", "1") != "0"
LOW_EDGE_FVG_SYMBOLS   = _parse_symbol_list(os.getenv("LOW_EDGE_FVG_SYMBOLS", "NEAR-USDT,NEARUSDT"))
LOW_EDGE_OB_SYMBOLS    = _parse_symbol_list(os.getenv("LOW_EDGE_OB_SYMBOLS", ""))

DIRECTION_EDGE_FILTER  = os.getenv("DIRECTION_EDGE_FILTER", "1") != "0"
LOW_EDGE_LONG_SYMBOLS  = _parse_symbol_list(os.getenv("LOW_EDGE_LONG_SYMBOLS", ""))
LOW_EDGE_SHORT_SYMBOLS = _parse_symbol_list(os.getenv("LOW_EDGE_SHORT_SYMBOLS", "AAVE-USDT,AAVEUSDT"))

# --- Context momentum pack (DEFAULT ON, validated together 2026-06-05) ----------
# Weak relative strength and session-momentum mismatches → higher SL rate.
# All four proven together on 8640×15m across all monthly slices.
RELATIVE_STRENGTH_LOOKBACK_HOURS      = int(os.getenv("RELATIVE_STRENGTH_LOOKBACK_HOURS", "1"))
LONG_RELATIVE_WEAKNESS_FILTER         = os.getenv("LONG_RELATIVE_WEAKNESS_FILTER", "1") != "0"
LONG_RELATIVE_WEAKNESS_MAX_PCT        = float(os.getenv("LONG_RELATIVE_WEAKNESS_MAX_PCT", "-1.60"))

BULL_NEUTRAL_LONG_NARROW_ZONE_FILTER  = os.getenv("BULL_NEUTRAL_LONG_NARROW_ZONE_FILTER", "1") != "0"
BULL_NEUTRAL_LONG_MAX_ZONE_WIDTH_PCT  = float(os.getenv("BULL_NEUTRAL_LONG_MAX_ZONE_WIDTH_PCT", "0.00173509"))

LONG_NY_COIN_MOMENTUM_FILTER          = os.getenv("LONG_NY_COIN_MOMENTUM_FILTER", "1") != "0"
LONG_NY_MIN_COIN_CHANGE_1H            = float(os.getenv("LONG_NY_MIN_COIN_CHANGE_1H", "0.0"))

SHORT_FVG_COIN_MOMENTUM_FILTER        = os.getenv("SHORT_FVG_COIN_MOMENTUM_FILTER", "1") != "0"
SHORT_FVG_MAX_COIN_CHANGE_1H          = float(os.getenv("SHORT_FVG_MAX_COIN_CHANGE_1H", "0.0"))

# Small edge on top of context momentum pack: improves aggregate/MC/DD.
# DISABLED 2026-08-24. Measured individually on the honest execution model
# rather than as part of the pack it shipped with:
#   on:  921 сд  74.8%  +230.87R  DD -9.25R  ratio 25.0
#   off: 946 сд  74.9%  +241.10R  DD -9.25R  ratio 26.1
FVG_LONDON_BTC_UP_FILTER  = os.getenv("FVG_LONDON_BTC_UP_FILTER", "0") != "0"
FVG_LONDON_BTC_UP_MIN_PCT = float(os.getenv("FVG_LONDON_BTC_UP_MIN_PCT", "0.29"))

# --- Risk sizing overlays (COMPUTED BUT NOT APPLIED) ---------------------------
# ⚠️ 2026-09-02: these do NOT size anything, despite the name and despite the
# 'DEFAULT ON' they used to advertise. They write setup['risk_mult'], and
# NOTHING multiplies a position or an R by that field:
#   * src/autotrader.py never mentions risk_mult (its size comes from
#     SYMBOL/SESSION/HTF/EXTENSION/VOL_ATR, a separate stack);
#   * backtest.py only copies it into the export column;
#   * neither `signals` nor `setup_log` even has a risk_mult column, so it
#     cannot reach the live sizing path at all.
# The same was true of the kNN overlay removed on 2026-08-31: its commit said
# the live book carried +-20% from it, and that was wrong for this reason.
#
# Left in place rather than deleted: inert code costs nothing to keep.
#
# ✅ 2026-09-02 the open question is CLOSED and the answer is do not wire them.
# The field does vary -- 1.0 / 1.15 / 1.25, with 72-79% of the book boosted --
# but it does not predict. Unit R by bucket, size divided out:
#          1.0        1.15       1.25        base
#   2023  +0.095     +0.095     +0.213      +0.119
#   2024  +0.150     +0.195     +0.159      +0.177
#   2026  +0.234     +0.272     +0.206      +0.248
# The LARGEST boost lands on the WORST bucket in two windows of three, and the
# response is not monotone in any of them. Applying this would grow the size of
# the trades that do worst -- exactly the defect found in the stocks kNN rule
# on 2026-08-31. A boost covering three quarters of the book is leverage, not
# selection, and this one is not even aimed the right way.
QUALITY_RISK_OVERLAY    = os.getenv("QUALITY_RISK_OVERLAY", "1") != "0"
QUALITY_RISK_MULT       = float(os.getenv("QUALITY_RISK_MULT", "1.15"))
QUALITY_RISK_MAX_MULT   = float(os.getenv("QUALITY_RISK_MAX_MULT", "1.15"))
QUALITY_RISK_VOL_MIN    = float(os.getenv("QUALITY_RISK_VOL_MIN", "0.8"))
QUALITY_RISK_VOL_MAX    = float(os.getenv("QUALITY_RISK_VOL_MAX", "1.2"))
QUALITY_RISK_RSI_MIN    = float(os.getenv("QUALITY_RISK_RSI_MIN", "50"))
QUALITY_RISK_RSI_MAX    = float(os.getenv("QUALITY_RISK_RSI_MAX", "60"))
HIGH_EDGE_RISK_SYMBOLS  = _parse_symbol_list(
    os.getenv(
        "HIGH_EDGE_RISK_SYMBOLS",
        "SUI-USDT,SUIUSDT,SOL-USDT,SOLUSDT,TON-USDT,TONUSDT,HYPE-USDT,HYPEUSDT,AAVE-USDT,AAVEUSDT",
    )
)
REL_STRENGTH_RISK_UP          = os.getenv("REL_STRENGTH_RISK_UP", "1") != "0"
REL_STRENGTH_RISK_UP_MIN_PCT  = float(os.getenv("REL_STRENGTH_RISK_UP_MIN_PCT", "0.5"))
REL_STRENGTH_RISK_UP_MAX_PCT  = float(os.getenv("REL_STRENGTH_RISK_UP_MAX_PCT", "2.0"))
REL_STRENGTH_RISK_UP_MULT     = float(os.getenv("REL_STRENGTH_RISK_UP_MULT", "1.15"))
REL_STRENGTH_RISK_UP_MAX_MULT = float(os.getenv("REL_STRENGTH_RISK_UP_MAX_MULT", "1.25"))
TREND_PAIR_RISK_UP            = os.getenv("TREND_PAIR_RISK_UP", "1") != "0"
# Flipped bullish->bearish 2026-07-18: 365d backtest showed 1h=bull&4h=bull is
# the WEAKEST major bucket (WR 61.4%, SL% 20.4%, netR/tr +0.376 — an already-
# extended move, not a fresh one), while 1h=bear&4h=bear is genuinely strong
# (WR 66.0%, SL% 14.6%, netR/tr +0.511). Sizing up into the bullish pair was
# boosting risk on the bot's own worst-performing setup type.
TREND_PAIR_RISK_UP_1H         = os.getenv("TREND_PAIR_RISK_UP_1H", "bearish").lower()
TREND_PAIR_RISK_UP_4H         = os.getenv("TREND_PAIR_RISK_UP_4H", "bearish").lower()
TREND_PAIR_RISK_UP_MULT       = float(os.getenv("TREND_PAIR_RISK_UP_MULT", "1.15"))
TREND_PAIR_RISK_UP_MAX_MULT   = float(os.getenv("TREND_PAIR_RISK_UP_MAX_MULT", "1.25"))

# --- Adaptive market-regime filter packs (from friend's v2 — DEFAULT OFF) ------
# Graduated quality gate: requires progressively higher MTF score + structure as
# the regime worsens (clean trend → mixed → choppy), and returns a per-regime
# risk_mult for position sizing.
#
# A/B BACKTEST RESULT (10 symbols, 2880×15m, ~2 months):
#   CURRENT : 604 tr, 41.7% WR, +0.153 net R/trade, +92R total
#   ADAPTIVE: 378 tr, 41.5% WR, +0.129 net R/trade, +49R total
# Verdict: DEFENSIVE filter — helps in choppy month (May: +0.043→+0.081 R/trade)
# but cuts winners in strong-trend month (June: +0.625→+0.416).
#
# 🔑 RE-VALIDATED 2026-08-28 on the current config. The A/B above ran at 41.7%
# win rate, i.e. on a system that no longer exists — different fill model,
# different exits, no sizing stack. Third dead justification found in two days
# (after _setup_rank and the TP1 history), so the number mattered.
#
#            base                          packs ON
#   2023   668tr +123.39R 15.2/37.2 | 561tr +116.37R 19.8/46.1   +30% / +24%
#   2024   872tr +177.84R 36.0/72.6 | 761tr +142.16R 25.0/55.6   -31% / -23%
#   cur   1167tr +377.46R 66.1/195.4| 994tr +303.95R 47.0/162.3  -29% / -17%
#
# The old verdict was right in substance: it cuts 12-16% of trades, gives back
# 6-20% of profit, and trades a large risk improvement in the HOSTILE window
# for a large degradation in the other two. Stays OFF as a default.
#
# What changed is that it now has a measured USE. This is a ready-made
# emergency brake: one variable turns the bot conservative, and in the hostile
# window it produced the single largest risk improvement measured anywhere in
# this project (+30% on worst-windows, +24% on ulcer). It cannot be switched
# automatically — detecting the regime live has been tested and does not work —
# but if the account owner decides the market has turned, the price of flipping
# it is now known in advance rather than guessed.
ADAPTIVE_FILTER_PACKS       = os.getenv("ADAPTIVE_FILTER_PACKS", "0") != "0"
ADAPTIVE_MIXED_SCORE_BUMP   = int(os.getenv("ADAPTIVE_MIXED_SCORE_BUMP", "1"))
ADAPTIVE_CHOP_SCORE_BUMP    = int(os.getenv("ADAPTIVE_CHOP_SCORE_BUMP", "2"))
ADAPTIVE_HOT_SCORE_BUMP     = int(os.getenv("ADAPTIVE_HOT_SCORE_BUMP", "1"))
ADAPTIVE_MIXED_EFF_MIN      = float(os.getenv("ADAPTIVE_MIXED_EFF_MIN", "0.20"))
ADAPTIVE_CHOP_EFF_MIN       = float(os.getenv("ADAPTIVE_CHOP_EFF_MIN", "0.28"))
ADAPTIVE_HOT_EFF_MIN        = float(os.getenv("ADAPTIVE_HOT_EFF_MIN", "0.22"))
ADAPTIVE_CHOP_MIN_VOLUME    = float(os.getenv("ADAPTIVE_CHOP_MIN_VOLUME", "2.0"))
ADAPTIVE_HOT_MIN_VOLUME     = float(os.getenv("ADAPTIVE_HOT_MIN_VOLUME", "2.0"))
ADAPTIVE_HOT_VOL_RATIO      = float(os.getenv("ADAPTIVE_HOT_VOL_RATIO", "3.0"))
ADAPTIVE_EXTREME_VOL_RATIO  = float(os.getenv("ADAPTIVE_EXTREME_VOL_RATIO", "5.0"))
ADAPTIVE_EXTREME_ATR_PCT    = float(os.getenv("ADAPTIVE_EXTREME_ATR_PCT", "0.035"))
ADAPTIVE_MIXED_RISK_MULT    = float(os.getenv("ADAPTIVE_MIXED_RISK_MULT", "0.75"))
ADAPTIVE_CHOP_RISK_MULT     = float(os.getenv("ADAPTIVE_CHOP_RISK_MULT", "0.50"))
ADAPTIVE_HOT_RISK_MULT      = float(os.getenv("ADAPTIVE_HOT_RISK_MULT", "0.50"))
ADAPTIVE_BEAR_SQUEEZE_GUARD = os.getenv("ADAPTIVE_BEAR_SQUEEZE_GUARD", "1") != "0"
ADAPTIVE_BEAR_SKIP_NEW_YORK = os.getenv("ADAPTIVE_BEAR_SKIP_NEW_YORK", "1") != "0"
ADAPTIVE_BEAR_VOL_MIN_RATIO = float(os.getenv("ADAPTIVE_BEAR_VOL_MIN_RATIO", "0.8"))
ADAPTIVE_BEAR_VOL_MAX_RATIO = float(os.getenv("ADAPTIVE_BEAR_VOL_MAX_RATIO", "1.8"))

# --- Stability overlay: deterministic kill-switch for poorly-validated regimes -
# 2026-06-11 A/B: OVERLAP session (London+NY overlap) = WR 32%, -6.3R over 19tr.
# Skipping it: +5R total, DD -20%. Both sessions fight at overlap = chop hour.
# DISABLED 2026-08-24. The justification above rests on NINETEEN trades, and on
# the old backtest that filled at prices the market never offered. It bans the
# entire OVERLAP session — 122 trades in the current window, which run 73.8% and
# +0.257R against a base of 73.8% and +0.204R, positive in both halves.
#
# Effect of switching it off, live gates applied:
#   on:  767 сд  73.8%  +170.73R  DD -15.52R  ratio 11.0
#   off: 838 сд  74.0%  +191.73R  DD -12.15R  ratio 15.8
# The OVERLAP half-split is uneven (+0.061/+0.446), so this is not shipped
# because that session is proven good — it is shipped because the ban was never
# proven at all. A gate that closes a whole trading session needs more than 19
# samples behind it, and the burden of proof belongs to the filter.
STABILITY_FILTERS_ENABLED   = os.getenv("STABILITY_FILTERS_ENABLED", "0") != "0"
STABILITY_SKIP_PACKS        = {s.lower() for s in _parse_symbol_list(os.getenv("STABILITY_SKIP_PACKS", ""))}
STABILITY_SKIP_SESSIONS     = set(_parse_symbol_list(os.getenv("STABILITY_SKIP_SESSIONS", "OVERLAP")))
STABILITY_MIN_EFF_RATIO     = float(os.getenv("STABILITY_MIN_EFF_RATIO", "0.0"))
STABILITY_MIN_VOLUME_RATIO  = float(os.getenv("STABILITY_MIN_VOLUME_RATIO", "0.0"))
STABILITY_MIN_QUALITY_SCORE = float(os.getenv("STABILITY_MIN_QUALITY_SCORE", "0.0"))

# --- Claude tiered analysis (cascade: cheap LIGHT gate + rare deep HEAVY) ---
# LIGHT  : Haiku validates every passed setup in ONE cached batch call (JSON via tool).
# HEAVY  : Sonnet re-checks only top setups (score >= HEAVY_MIN_SCORE) with coin memory.
# Caching: static rules block cached 1h → cheap re-reads on the 5-min scan loop.
CLAUDE_LIGHT_MODEL        = os.getenv("CLAUDE_LIGHT_MODEL", "claude-sonnet-4-5")
CLAUDE_HEAVY_MODEL        = os.getenv("CLAUDE_HEAVY_MODEL", "claude-sonnet-4-5")
CLAUDE_HEAVY_MIN_SCORE    = int(os.getenv("CLAUDE_HEAVY_MIN_SCORE", "9"))    # lowered 10→9: all survivors get Sonnet check
CLAUDE_HEAVY_MAX_PER_SCAN = int(os.getenv("CLAUDE_HEAVY_MAX_PER_SCAN", "5")) # max HEAVY checks per scan
CLAUDE_MEMORY_LIMIT       = int(os.getenv("CLAUDE_MEMORY_LIMIT", "25"))      # recent outcomes per coin (HEAVY)
CLAUDE_MAX_RISK_SCORE     = int(os.getenv("CLAUDE_MAX_RISK_SCORE", "7"))     # counter-arg auto-reject if risk >= this (7 = "real concern" per scale)

# Claude as a GATE, or as an observer. 1 = his verdict withholds setups. 0 =
# SHADOW: he is still called, scored and logged, but the rules filter alone
# decides what trades. Switchable at runtime from the admin panel; the DB state
# wins over this default.
#
# backtest.py never calls Claude, so with the gate OFF the live bot trades the
# same population the model measures — the only way to score his verdict against
# real fills rather than simulated ones.
#
# ⚠️ Approval here runs 52%, so shadow mode nearly DOUBLES this book. On the
# stocks desk approval is 80% and the step is a quarter. Treat these very
# differently.
CLAUDE_GATE_ENABLED = os.getenv("CLAUDE_GATE_ENABLED", "1") != "0"
CLAUDE_CACHE_TTL          = os.getenv("CLAUDE_CACHE_TTL", "1h")              # prompt cache TTL ("5m" or "1h")
CLAUDE_DAILY_BUDGET_USD   = float(os.getenv("CLAUDE_DAILY_BUDGET_USD", "1.00"))  # hard daily cap (real Sonnet usage ~$0.3-0.5/day)
CLAUDE_BUDGET_RESERVE_USD = float(os.getenv("CLAUDE_BUDGET_RESERVE_USD", "0.05")) # stop when remaining < reserve

# Epoch for every live-history read: Claude's Hist/SCORECARD/coin-memory, the
# over-strictness corrector, the rolling report windows, the variant arms and
# the auto-block statistics. Rows before it describe a bot that no longer
# exists and must not be averaged in with the current one.
#
# 1786575600 = 2026-08-12 23:00 UTC — ZONE_WATCH shipped. The bot stopped
# publishing at whatever price was showing and started waiting for price to
# return to the setup's own zone. That changes the ENTRY, which is the single
# thing that moves both win rate and profit together, so every rate measured
# before it belongs to a different system. Two markers died on exactly this
# transition and are the proof it matters: the old sniper definition went from
# 90% to 71% win rate, and the extreme-RSI claim in Claude's prompt from
# 83-86% to no edge at all. Nothing else changed about either — only the fill.
#
# Previous epoch was 1785456000 (2026-07-31, the parity fixes: Strong1h finally
# computing, matched history windows, 24/7 scanning, the stale-entry guard).
# Raise this WHENEVER a change makes past live outcomes unrepresentative; that
# is cheaper than discovering months later that a number was a blend of two
# different bots. Historical rows are never deleted — the admin report's
# "all time" and typed-date windows still reach them, which is the only way to
# compare eras at all.
# Previous epoch was 1786575600 (2026-08-12). Raised 2026-08-25: until that
# day the shadow tracker resolved every UNSENT setup by opening the trade at
# `entry` whether or not price ever traded there. Measured on the live A/B
# export, inside ONE arm: sent 71.0 pct WR / +0.145R against unsent 83.6 pct
# / +0.931R -- same filter, different resolver. Every rejected-bucket number
# Claude read before this timestamp is inflated by roughly that much, and
# the prompt turns a strong rejected bucket into "you are over-rejecting".
LIVE_HIST_EPOCH_TS = float(os.getenv("LIVE_HIST_EPOCH_TS", "1787670000"))

# --- Structure-based stops/takes (swing mode, 15m, ~20x leverage) ---
# SL sits at swing invalidation (recent swing low/high) + ATR buffer, then
# clamped to safe leverage bounds. TPs are R-multiples for swing-sized moves.
#   risk%  ~1.2–3.0% of price  → on 20x = 24–60% margin at risk per stop
#   TP1 = 1.5R (1.8–4.5% move → 36–90% on 20x), close 50%, move SL to BE
#   TP2 = 3.0R (3.6–9%   move → 72–180% on 20x), let winner run
# Readable from the environment since 2026-08-26 — same reason as
# SMC_SWING_LOOKBACK: it was a constant and could not be measured.
# RE-SWEPT 2026-09-02 on current mechanics (the earlier note predates
# LEVELS_FROM_STRUCTURE). 2026-08-26 window, trades / net R / worst-windows
# (ratio) / profit-per-ulcer:
#   10   1577  +471.16  13.75 (34.3)  147.5
#   14   1570  +472.41  13.75 (34.3)  153.8   ← best profit and ulcer
#   20   1563  +463.13  14.14 (32.8)  146.0
#   28   1561  +463.32  14.04 (33.0)  144.8
# Longer periods are worse on every measure; 10 is a near-tie on profit but
# loses on ulcer. 14 stays.
ATR_PERIOD    = int(os.getenv("ATR_PERIOD", "14"))
# 0.5 -> 1.0 on 2026-08-24. This is how far beyond the swing the structural stop
# is pushed, and at 0.5 ATR it sat close enough to the level that noise took it
# out without the structure actually breaking.
#
# Swept on the honest execution model, live gates applied. The response is
# monotone across the whole range — win rate up, profit up, drawdown down —
# which is what a real effect looks like, and it holds in BOTH halves:
#   буфер   сд   WR      netR      DD      p/DD    1пол WR / 2пол WR
#   0.5    921  74.5%  +215.6R  -11.91R   18.1     —
#   0.7    924  74.8%  +222.5R   -9.65R   23.1    76.6% / 75.6%
#   0.8    924  74.9%  +228.5R   -9.78R   23.4    76.8% / 75.8%
#   1.0    923  74.9%  +232.0R   -9.25R   25.1    77.2% / 75.6%
#   1.3    923  75.5%  +231.1R   -9.15R   25.2    77.8% / 76.5%
#
# Trade count barely moves: the buffer changes where the stop goes, not which
# setups fire. So this is purely an exit improvement.
#
# Stopped at 1.0 rather than 1.3 because the risk clamp starts binding: the
# share of trades pinned at RISK_MAX_PCT goes 23.3% -> 27.8% -> 32.5%, and past
# that the buffer is not actually expressed — it is silently truncated, and
# risk-normalised sizing then halves those positions.
#
# RE-SWEPT 2026-09-02. The table above predates LEVELS_FROM_STRUCTURE (shipped
# 2026-08-31), which changed where the stop is placed — it shows 921-924 trades
# where the same window now has ~1570, so it was measuring a different bot.
# Re-run on current mechanics, 2026-08-26 window: trades / net R / worst-windows
# (ratio) / profit-per-ulcer
#   0.5   1605  +456.94  12.75 (35.8)  140.8
#   0.7   1594  +469.56  13.71 (34.2)  141.0
#   1.0   1570  +472.41  13.75 (34.3)  153.8   ← best on profit AND ulcer
#   1.3   1546  +455.29  13.45 (33.9)  147.2
#   1.6   1523  +454.36  12.96 (35.1)  151.4
# The value survives the mechanics change unchanged. Worst-windows is flat
# across the whole range (33.9-35.8), so the choice rests on profit and ulcer,
# and 1.0 tops both.
SL_ATR_BUFFER = float(os.getenv("SL_ATR_BUFFER", "1.0"))
# SWEPT 2026-09-01 (first time -- the ceiling had two recorded sweeps, this
# floor had none). Motive: cost_r scales inversely with stop width, so trades
# pinned here pay 0.166R against 0.057R at the ceiling, and the floor bucket
# sits below book in all three windows. Widening the floor should have bought
# that back. It does not:
#   floor   2023 profit  DD      ratios       2024 profit  ratios      2026 profit  DD      ratios
#   0.012   +142.50R  -11.75  14.8/34.8      +201.26R  29.1/68.2     +465.37R  -17.55  33.8/151.2
#   0.015   +147.24R  -10.84  16.8/42.9      +202.84R  29.3/68.1     +435.26R  -18.05  31.9/134.3
#
# Better on EVERY measure in the hostile window, flat in 2024, worse on every
# measure in the current one -- and the current loss (-30R) is larger than the
# hostile gain (+4.7R). Regime-dependent, not a cost effect, so 0.012 KEPT.
#
# 0.015 is the highest value comparable in R at all: RISK_REFERENCE_PCT is
# 0.015, and above it risk-normalised sizing shrinks the position, so R stops
# meaning the same money. Sweeping higher needs money accounting -- the same
# warning the ceiling carries.
RISK_MIN_PCT  = float(os.getenv("RISK_MIN_PCT", "0.012"))  # min SL distance = 1.2%
RISK_MAX_PCT  = float(os.getenv("RISK_MAX_PCT", "0.035"))  # max SL distance = 3.5%
# 🔁 RE-SWEPT 2026-08-28 in BOTH directions under the new exit (trail 0.02,
# TP2 3.0R). Reason to re-check: stop width sets the R unit, and TP1, the trail
# arming point and TP2 are all R multiples — two of the three had just changed.
# The 08-24 result survives; 0.035 KEPT:
#   ceil    2023 profit  ratios      2026 profit  ratios
#   0.028   +148.32R   19.2/49.3     +409.91R   76.6/211.6
#   0.030   +152.22R   20.7/51.8     +409.98R   73.7/207.2
#   0.035   +148.46R   23.2/53.7     +416.45R   87.2/230.9   <- kept
#   0.040   +144.46R   20.7/48.8     +407.18R   87.7/219.9
#   0.045   +151.95R   22.0/55.9     +421.32R   86.6/248.2
# Tightening costs 11-17% of both risk ratios in both windows for flat profit.
#
# ⚠️ 0.045 LOOKS better (profit up in both windows, ulcer ratio up in both) and
# is rejected as leverage. Normalisation floors at REF/FLOOR = 3.33%, so any
# ceiling above that lets money at risk exceed one unit — 1.05 at 3.5%, 1.20 at
# 4.0%, 1.35 at 4.5%. The signature is there: profit +1.2/+2.3% while
# worst-windows ratio falls in BOTH windows (23.2->22.0, 87.2->86.6). Profit
# bought with exposure, not edge. Read any sweep of this parameter in money,
# never in R alone.
#
# Useful negative on interactions: changing the exit did NOT invalidate this
# setting, unlike TP2 whose "unreachable" justification died with the exit
# change. Not every coupled parameter needs re-deriving after a change — but
# the check is cheap and this one was worth running.
#
# ⚠️ Does NOT transfer to the stocks desk, where the same clamp is the primary
# stop mechanism (44-47% of trades pinned on it) and TIGHTENING is what helps
# early. Opposite direction, same parameter. Measure per desk.
# 2026-08-24: raised from 0.03. The clamp was pinning 23-33% of stops tighter
# than structure wanted, i.e. into noise. mae_r shows a trade sitting 0.9R
# against us still wins 87.2% of the time, so that zone is not information.
# Raising the cap improved EVERY axis at once: +2 trades, +0.5pp WR, +7%
# profit, -10% drawdown. See memory exit-mechanics-sweep.
# 2026-06-11 TP1 sweep (20 sym, 90d×15m, trail 0.5): TP1=1.0R beats 1.5R on WR
# (+13-16pp, 65-76% across 30/60/90d) at equal-or-better total R and half the DD.
# 2026-07-22 follow-up sweep (365d/20sym, corrected 192 window): closer STILL
# better on risk-adjusted return — TP1=0.7R vs 1.0R lifts WR 72.7→79.6%, cuts
# stops -25% and maxDD -32.15R→-21.02R (-35%) for only -7% total net R (each
# win banks smaller; breakeven protection arms sooner so fewer full -1R losses).
# Validated on an independent recent 6mo window (WR 72.4→80.7%, DD -29.7→-18.6R,
# profit only -2.7% there). User chose 0.7 (drawdown/survival > absolute profit,
# esp. at 10x leverage). Wider runner trail does NOT recover the lost profit
# (tested, worse). Set TP1_R_MULT=1.0 to revert to the max-absolute-profit variant.
# 0.7 -> 0.6 on 2026-08-09. Measured on TWO independent windows, same sign and
# similar size on both — profit barely moves, drawdown falls hard:
#   window 1 (to 09.08.26, 1214tr): 80.6%/+522R/-21.3R  ->  83.4%/+500R/-11.8R
#   window 2 (to 31.01.26,  924tr): 81.1%/+396R/-18.2R  ->  84.2%/+387R/-12.8R
#   i.e. win rate +2.8/+3.1pp, profit -4.2%/-2.4%, DRAWDOWN -45%/-30%
# Profit per unit of drawdown goes 24->42 and 22->30. Giving up ~3% of profit
# for a third to half the drawdown is worth it here specifically because
# drawdown is the binding constraint on position size: losses cluster (real DD
# measured 2.9x worse than any of 5000 shuffles of the same trades), so the
# size a book can carry is set by the cluster, not by the average.
# 🔴 2026-08-26 — EVERY measurement above predates the 2026-08-23 execution
# fix and is therefore void. Win rates of 79-84% are the signature of the
# fantasy-fill model: the backtest filled mid-zone without checking price ever
# returned there. The "give up 3% of profit for 30-45% less drawdown" trade
# that chose 0.7, and then 0.6, does not reproduce on honest data — there,
# widening TP1 makes MORE money, not less.
#
# Re-measured across three windows (18000 candles each):
#            base 0.6                          0.75
#   2023   668tr 68.9% +95.96R  11.3/26.7 |  653 65.2% +112.30R 12.2/28.2
#   2024   872tr 73.5% +161.14R 28.4/58.0 |  840 68.3% +168.33R 31.7/59.8
#   cur   1172tr 75.9% +369.51R 32.4/127.7| 1075 70.7% +346.74R 34.3/104.0
#   (ratios are profit/worst-windows and profit/ulcer)
#
# 0.75 is rejected: win rate drops 3.7-5.2pp in every window and the current
# window loses 97 trades, while the profit gain is ambiguous — normalised to
# equal risk it is +7.3/+11.3/+6.0% by worst-windows but +5.3/+3.3/-18.6% by
# ulcer, i.e. the two risk measures disagree on the window that matters least.
# 0.5 was tested too and is worse on everything: 2023 gives 72.0% win rate for
# +74.60R and ratios of 6.7/14.0 against the base 11.3/26.7.
#
# So 0.6 STAYS — tuning on a broken model happened to land somewhere sane.
# What is dead is the REASON, and a live parameter carrying a false reason is
# a trap: the next person to open this file would reason forward from it.
#
# Useful thing the sweep does show, on honest data: TP1 is a clean win-rate
# dial. 0.5 -> 0.6 -> 0.75 moves 2023 win rate 72.0% -> 68.9% -> 65.2% while
# profit goes +74.60R -> +95.96R -> +112.30R. Win rate can be bought to
# almost any level and the price is always money.
# 🔁 RE-CHECKED 2026-08-28 under the NEW exit (trail 0.02, TP2 3.0). TP1 arms
# the trail, so changing the trail could have moved its optimum — it did not:
#   TP1    2023 tr/WR/profit  ratio    2026 tr/WR/profit  ratio
#   0.5R   708 72.3% +113.85R  13.5    1277 78.2% +324.67R  47.4
#   0.6R   694 69.5% +143.59R  21.0    1192 75.6% +396.98R  80.3  <- kept
#   0.75R  678 66.1% +161.44R  24.5    1082 70.7% +379.65R  44.0
# 0.75 buys the hostile window but wrecks the current one (worst-windows 8.63
# against 4.95, i.e. 74% deeper). 0.6 stays.
#
# 📌 FOR THE ACCOUNT OWNER, not an optimisation: this is the win-rate dial, and
# 0.5R delivers BOTH standing goals at once — win rate 75.6% -> 78.2% and 1192
# -> 1277 trades — for 18% of the profit and roughly half the risk-adjusted
# return. Win rate is purchasable to almost any level here and the price is
# always money. Do not move this without the owner choosing the trade.
TP1_R_MULT    = float(os.getenv("TP1_R_MULT", "0.6"))      # TP1 = entry ± risk * 0.6
# SWEPT 2026-09-02 (first time -- TP1 above has two recorded sweeps, this had
# none). TP2 is nearly inert: under the trail exit fewer than 2% of trades ever
# reach it. But those few pay ~+4.2R each, so the cap decides the whole tail.
#
#   TP2   2023 profit  ratios      2024 profit  ratios      2026 profit  ratios
#   2.5      --           --          --          --        +459.26R  33.4/148.0
#   3.0   +142.50R   14.8/34.8    +201.26R   29.1/68.2     +465.37R  33.8/151.2
#   4.0   +145.52R   15.1/35.7    +203.17R   29.4/68.9     +468.41R  34.1/152.4
#   5.0   +147.23R   15.3/36.1    +204.61R   29.6/69.4     +472.41R  34.3/153.8
#
# 4.0 wins on profit AND both risk ratios in ALL THREE windows, with drawdown
# and win rate unchanged to the decimal and the trade count moving by one. That
# is not leverage -- no position is larger and no trade is riskier; the cap was
# simply truncating runners the trail would have carried further.
#
# 5.0 then measured on all three and beats 4.0 everywhere on profit and on both
# risk ratios, so it is taken. The hostile window is where a longer tail could
# have stopped paying and it does not: +147.23R against +145.52R there.
#
# Not pushed further. On the stocks desk, where the same sweep ran wider, the
# current window plateaus past 6.0 -- nothing reaches TP2 any more and the trail
# decides everything -- while another window drops at 6.0 and wobbles back up at
# 8.0 and 12.0. That is individual trades crossing a boundary, not a trend.
TP2_R_MULT    = float(os.getenv("TP2_R_MULT", "5.0"))      # TP2 = entry ± risk * 5.0
# 2026-08-28: back to 3.0. It was cut to 2.0 for being "unreachable", which was
# right under the OLD exit (bank 50% at TP1, fixed TP2) and is wrong under the
# current one (full position past TP1 + a 0.02 ATR trail). TP2 is no longer the
# exit — the trail is, on ~70% of trades — so TP2 is just a CAP, and an
# unreachable cap costs nothing while a near one truncates the best trades.
#            2023 profit  ratios      2026 profit  ratios
#   1.5R      +125.11R   17.7/43.1     +360.94R   66.5/185.0
#   2.0R      +137.03R   20.0/48.5     +378.16R   74.6/200.5   <- was
#   2.5R      +139.66R   20.4/49.9     +390.02R   78.8/210.6
#   3.0R      +143.59R   21.0/51.4     +396.98R   80.3/216.4   <- shipped
#   4.0R      +145.29R   21.2/52.0     +400.90R   81.0/218.5
#   8.0R      +147.82R   21.6/52.9     +404.95R   81.9/220.8
# 2024: 2.0R +189.26R 42.1/88.0 -> 3.0R +190.15R 42.3/88.9.
# All three windows improve on all three measures (+4.8/+0.5/+5.0% profit).
#
# Two things worth noting about the shape. Worst-windows is IDENTICAL from 2.5R
# up (6.84 / 4.95 in every row) — TP2 only ever touches winners, so it cannot
# move drawdown. And unlike the trail sweep, monotone-to-the-edge is NOT a
# warning here: raising TP2 REDUCES reliance on the optimistic wick-fill the
# backtest grants at that level, so the far end of this curve is the
# conservative end, not the fitted one. 3.0 rather than 8.0 because returns
# diminish (2.0->3.0 captures ~80% of the total available) and because TP2 has
# a second job the backtest cannot see: it is a resting exchange order, and if
# the bot is down it is the only thing that banks a winner. The exchange
# backstop sits at STOP_EXCHANGE_BACKSTOP_R=1.5, so bot-down is -1.5R against
# +3.0R rather than +2.0R — a wider net, not a weaker one, but a more distant
# one, and that trade-off is the reason not to push it to 8.
#
# ⚠️ This interacts with TRAIL_ATR_MULT and TP1_CLOSE_FRAC. If the exit ever
# goes back to banking part of the position at TP1, re-measure this.

# Runner exit after TP1: trail the remaining 50% by ATR instead of fixed TP2.
# Backtest (10 sym, 2880x15m): +21% net R, -27% max drawdown, same win rate vs
# fixed TP2. Trailing stop = peak ∓ TRAIL_ATR_MULT×ATR, floored at breakeven.
TRAIL_RUNNER_ENABLED = os.getenv("TRAIL_RUNNER_ENABLED", "1") != "0"
# RE-SWEPT 2026-09-02 (the 0.02 predates LEVELS_FROM_STRUCTURE). Seven values
# on 2026-08-26: the response is monotone, tighter is better on profit AND
# both risk measures, while trade count and win rate do not move at all --
# the trail only acts inside trades that are already winning, so this is a
# pure exit change, the class that is actually provable.
#   0.0 +477.15/156.5 | 0.003 +477.29/156.4 | 0.006 +476.43/155.9
#   0.01 +475.28/155.3 | 0.02 +472.41/153.8 (was) | 0.05 +464.09/149.2
# Confirmed on both other windows, better on every measure in all three:
#   2023 +147.23 → +149.06   2024 +204.61 → +207.17
# (those two re-measured on the shipped uniform config, not on the variant
#  with the strong branch revived — they differ by 0.02-0.03R)
# Not taking the 0.0-0.003 floor: it puts the stop flush against price and
# live X-Perp wicks are harsher than the model represents.
TRAIL_ATR_MULT       = float(os.getenv("TRAIL_ATR_MULT", "0.006"))  # base trail; post_tp1_v2 overrides per-context
# ✅ SIGNIFICANCE-TESTED 2026-08-28 (significance_check.py, 5000 bootstrap runs),
# trail 0.02 + TP2 3.0R together against the old exit on the current window:
#   baseline +390.99R -> candidate +416.45R, delta +25.47R, delta R/trade +0.020
#   p_gt_zero 1.0, 90% CI +17.42 to +33.96 — entirely above zero
# Every one of 5000 runs positive. This is the only change shipped this session
# that is significant on its own rather than merely consistent across windows.
#
# 🔑 WHY it tests so cleanly, and the general lesson: the exit change is PAIRED —
# the same 1266 trades in both arms, only their handling differs — so the
# comparison is within-trade and the between-trade variance cancels. Changes that
# alter WHICH trades are taken (a gate, a threshold) cannot be paired: only 1128
# of them matched, and the unmatched remainder swamps a 5% effect. On a single
# window, expect selection changes to come back "not significant" even when real,
# and judge them on cross-window consistency plus mechanism instead. Prefer a
# paired formulation of an experiment wherever one exists.
# 2026-08-28: 0.05 -> 0.02. The 08-24 sweep stopped at 0.05 and the curve was
# still falling. Swept further on all three pinned windows (uniform, i.e. the
# STRONG/WEAK overrides pinned to the same value so only the trail width moves):
#   mult    2023 profit  worst/ulcer     2026 profit  worst/ulcer
#   0.05     +136.43R     17.6/44.1       +379.65R     72.4/198.4
#   0.02     +137.03R     20.0/48.5       +378.16R     74.6/200.5   <- shipped
#   0.01     +138.00R     20.2/49.0       +380.20R     75.4/202.1
#   0.00     +138.97R     20.5/49.6       +381.69R     76.1/203.4
# 2024 at 0.02: +189.26R, 42.1/88.0 vs base +185.43R, 40.6/85.0.
# Against base all three windows improve on BOTH risk ratios (2023 +13.6/+10.0%,
# 2024 +3.7/+3.5%, 2026 +3.0/+1.1%) while profit is flat (+0.4 / +2.1 / -0.4%).
#
# 🔑 NOT shipped at 0.00 even though it measures best everywhere, monotonically.
# Two reasons, both about the boundary rather than the numbers:
#  1. An optimum sitting exactly on the edge of the tested range means the model
#     cannot say where the real optimum is — it can only say "further". Sizing
#     onto that edge is the largest available bet on the weakest evidence.
#  2. 0.00 puts the stop exactly on a level price has just traded. 0.02 ATR is
#     ~0.03% of price, roughly 1.5-3x the OKX perp spread, so the stop still
#     sits clear of the touch. There is no look-ahead in either (BT_TRAIL_LAG
#     anchors the peak to CLOSED bars only) — this is about fillability.
# 2026-08-24: 0.25 -> 0.05, monotone across the whole range on the honest
# model and confirmed in both halves. The runner earns, but gives it back on
# the pullback; exiting at the first failure to extend keeps +0.029R per
# trail exit over 702 of them. NOT the same as banking at TP1 — TP1_CLOSE_FRAC
# 0.5 was tested and is far worse (+184R vs +236R).

# Exit profile. ⚠️ As of 2026-08-28 this switch is a NO-OP and the description
# below is historical. It gates ONLY the context-aware trail split, and that
# split is inert (see POST_TP1_STRONG_TRAIL_ATR_MULT), so "post_tp1_v2" and
# "fixed" now produce identical results. The half that still does real work is
# TP1_CLOSE_FRAC=0 — keeping the FULL position past TP1 instead of banking 50%
# — and that is its own setting, not this one. Kept as a switch because the
# split machinery is still there and would come back if the widths diverge.
# Historical description: "post_tp1_v2" keeps the FULL position past TP1
# (TP1_CLOSE_FRAC=0) and trails by an ATR multiple chosen from the
# TP1-acceptance candle — strong follow-through trails wide (let it run),
# weak/rejected trails tight (lock).
# Validated 3 windows on our cache (90/180/365d): net R +80/+91/+124% with LOWER
# drawdown, win rate / trades / SL count UNCHANGED — it only changes how winners
# are harvested, never which trades are taken. "fixed" = legacy 50%-at-TP1 + BE.
TP1_CLOSE_FRAC = max(0.0, min(1.0, float(os.getenv("TP1_CLOSE_FRAC", "0.0"))))
EXIT_PROFILE   = os.getenv("EXIT_PROFILE", "post_tp1_v2").strip().lower()
# 🔑 The context split below has been INERT since 2026-08-24 and is now inert by
# choice. It reads `max(base, STRONG)` and `min(base, WEAK)`, so with base at or
# below STRONG and at or below WEAK every branch returns base. Lowering the base
# trail to 0.05 that day silently neutralised the whole "strong trails wide,
# weak trails tight" feature the comment above describes, in both bots.
# Reactivated and measured 2026-08-28, two windows, against base 0.05:
#   strong 0.15 (widen strong)  2023 +133.46R 19.2/46.4   2026 +371.08R 72.2/195.0
#   weak   0.02 (tighten weak)  2023 +135.85R 19.8/47.8   2026 +375.65R 73.8/198.5
#   both                        2023 +135.18R 19.6/47.5   2026 +374.31R 73.5/197.5
#   uniform 0.02 (no split)     2023 +137.03R 20.0/48.5   2026 +378.16R 74.6/200.5
# Widening the strong branch LOSES in both windows, and the plain uniform trail
# beats every split on every measure. The context signal carries nothing the
# trail width does not already carry, so STRONG is pinned to the base value to
# keep the branch inert. WEAK stays 0.15 and is inert too: min(0.02, 0.15)=0.02.
# 2026-09-02: pinned to 0.0 instead of to the base value. The branch is
# max(base, STRONG), so holding it inert by setting it EQUAL to the base
# only works until the base is tuned — lowering the base to 0.006 today
# would have silently revived it. 0.0 keeps max(base, 0.0) == base for any
# base, so the 2026-08-28 decision survives future tuning. Measured at the
# new base: uniform +476.48R / ulcer-ratio 156.0 vs revived +476.43 / 155.9
# — neutral now (it lost at base 0.05), so this costs nothing today and
# removes the trap.
POST_TP1_STRONG_TRAIL_ATR_MULT = float(os.getenv("POST_TP1_STRONG_TRAIL_ATR_MULT", "0.0"))
POST_TP1_WEAK_TRAIL_ATR_MULT   = float(os.getenv("POST_TP1_WEAK_TRAIL_ATR_MULT", "0.15"))
POST_TP1_STRONG_CLOSE_PROGRESS = float(os.getenv("POST_TP1_STRONG_CLOSE_PROGRESS", "0.25"))
POST_TP1_STRONG_WICK_PROGRESS  = float(os.getenv("POST_TP1_STRONG_WICK_PROGRESS", "0.55"))
POST_TP1_WEAK_CLOSE_PROGRESS   = float(os.getenv("POST_TP1_WEAK_CLOSE_PROGRESS", "-0.10"))

# --- k-NN price-shape analog risk overlay (Kronos-inspired, CPU-only) ----------
# After a setup passes, fetch a deep 15m series and match the recent price shape
# against the symbol's own past (nearest-neighbour). Score = fraction of the K
# most-similar past windows whose forward move favoured the trade direction.
# Backtest (2026-06-13, 90d, live-like 800-bar pool): score>=0.55 → WR ~68%,
# score<0.50 → WR ~59%. Used as a size multiplier (no gating) → +6% total R,
# trade frequency unchanged. Edge needs a deep pool, so a ~1000-candle fetch is
# done ONLY for symbols that already produced a setup (rare → cheap).
# Off since 2026-08-31. With KNN_HIGH_MULT/KNN_LOW_MULT neutralised the overlay
# changes no decision, but it still deep-fetched ~1000 candles PAGINATED for
# every qualified setup, before ranking - so most of those fetches were thrown
# away by the cap anyway. That sits on the publish path, and publish latency is
# what drags the fill away from the entry zone, which is the single largest cost
# in this book. backtest.py still computes knn_score independently, so the
# feature stays available for research; only the live fetch is gone.
KNN_RISK_OVERLAY   = os.getenv("KNN_RISK_OVERLAY", "0") != "0"
KNN_DEEP_CANDLES   = int(os.getenv("KNN_DEEP_CANDLES", "1000"))   # 1 Bybit page
KNN_MAX_HISTORY    = int(os.getenv("KNN_MAX_HISTORY", "800"))     # analog pool cap
KNN_SHAPE_LEN      = int(os.getenv("KNN_SHAPE_LEN", "12"))        # query window (3h)
KNN_HORIZON        = int(os.getenv("KNN_HORIZON", "16"))          # forward bars (4h)
KNN_K              = int(os.getenv("KNN_K", "40"))                # neighbours
KNN_MIN_HISTORY    = int(os.getenv("KNN_MIN_HISTORY", "120"))     # min bars to score
KNN_HIGH_SCORE     = float(os.getenv("KNN_HIGH_SCORE", "0.55"))   # size-up threshold
# kNN sizing was LIVE-ONLY: main.py applied these multipliers to risk_mult,
# backtest.py computes knn_score and exports it but never sizes on it. So
# every validated number assumed no kNN sizing while the live book carried
# +-20% from it. Measured on six windows across both bots, the score does
# not separate: crypto top vs bottom quartile +0.129/+0.225/+0.246 against
# +0.127/+0.223/+0.256 (bottom equal or better), and in stocks the boosted
# top quartile is BELOW book in all three windows (+0.590/+0.214/+0.576
# against +0.593/+0.529/+0.632) - the rule sized up the worst quartile.
# Neutralised rather than deleted: the score is still computed and exported,
# so it stays available if someone models it properly first.
KNN_HIGH_MULT      = float(os.getenv("KNN_HIGH_MULT", "1.0"))     # neutralised 2026-08-31
KNN_LOW_SCORE      = float(os.getenv("KNN_LOW_SCORE", "0.50"))    # size-down threshold
KNN_LOW_MULT       = float(os.getenv("KNN_LOW_MULT", "1.0"))      # neutralised 2026-08-31
KNN_RISK_MAX_MULT  = float(os.getenv("KNN_RISK_MAX_MULT", "1.50"))  # cap after overlays
KNN_RISK_MIN_MULT  = float(os.getenv("KNN_RISK_MIN_MULT", "0.50"))  # floor after overlays

# --- Research-validated setup cuts (2026-06-11, 20 sym, 30/60/90d backtests) ---
# RSI_Div confirmations: WR 23%, -0.21R/tr over 22tr — divergence in 15m chop = noise.
SKIP_RSI_DIV_SETUPS = os.getenv("SKIP_RSI_DIV_SETUPS", "1") != "0"
# Hour/weekday cuts — OFF by user choice (Mon-Fri 07-21 UTC full window).
# Backtest note: Monday ~0R/tr (53tr), 18-20 UTC ~+0.09R/tr (38tr) — re-enable
# via env SKIP_WEEKDAYS=0 / SKIP_UTC_HOURS=18,19,20 if WR needs a boost.
SKIP_UTC_HOURS = {h for h in os.getenv("SKIP_UTC_HOURS", "").split(",") if h.strip()}
SKIP_WEEKDAYS  = {d for d in os.getenv("SKIP_WEEKDAYS", "").split(",") if d.strip()}

# --- Zone watch (2026-08-10) ---------------------------------------------------
# Instead of publishing the moment Claude approves — which means entering at
# whatever price is showing, on average 0.55% worse than the setup's own zone —
# hold the setup and poll until price trades back INTO the zone, then publish
# and enter at market that instant.
#
# Why this matters more than anything else measured so far: the backtest fills
# at the zone midpoint, and that single assumption is worth +0.341R of its
# +0.400R/trade headline. Every other knob tried (TP distance, entry-drift
# tolerance, structural-target threshold) only trades win rate against
# drawdown; entry price is the one lever that raises both.
#
# Measured over 18000 candles (~187 days), same filter, same everything else:
#   publish immediately, 0.25% drift gate:  285 trades, 75.4% WR, +47R, DD -17.5R
#   wait for the zone (60 min):             708 trades, 73.4% WR, +116R, DD -17.6R
# Same profit per trade, same drawdown, 2.5x the trades — because the drift
# gate throws away far more than the wait does.
#
# A resting limit order would be slightly better still (+116R vs an estimated
# +85-100R here, since a market fill crosses the spread), but it would need the
# whole protective-OCO/position-tracking path rebuilt around a maybe-filled
# order, and it can sit and fire hours later at a moment nobody chose. Polling
# keeps every existing mechanism and the user gets the alert at the instant of
# entry. The spread cost is small on the pairs that survive SPREAD_MAX_BPS —
# but it is steep in relative terms: +0.05% of overpay costs ~27% of the edge,
# which is exactly why the poll interval is seconds, not minutes.
ZONE_WATCH_ENABLED  = os.getenv("ZONE_WATCH_ENABLED", "1") != "0"
# 60 -> 90 on 2026-08-26. How long a parked setup keeps waiting for price to
# come back to its zone before it is dropped.
# Rejected once already at 90/120 — on the old structure and against the
# 2026 base whose risk figure was anomalous. Both reasons are gone, so it
# was re-measured across three windows:
#
#            2023 (кризис)   2024 (обычное)   2026 (текущее)
#           окна  ulcer     окна  ulcer      окна  ulcer
#   60       8.2   19.5     26.2   61.3      34.3  119.1
#   90      11.3   26.7     28.4   58.0      32.4  127.7
#  120      12.6   25.6     20.5   56.2      38.1  135.0
#
# Neither wins everywhere. Across the three windows together: profit 583R ->
# 627R at 90 and 645R at 120, and both mean risk ratios improve at both.
# 120 buys its extra 3% by taking 22% off the 2024 window's worst-windows
# ratio; 90's worst case anywhere is -5.5%. Bounded damage beat the bigger
# headline. Adds 46-102 trades per window, win rate flat to +0.6pp.
ZONE_WATCH_MINUTES  = float(os.getenv("ZONE_WATCH_MINUTES", "90"))
ZONE_WATCH_POLL_SEC = int(os.getenv("ZONE_WATCH_POLL_SEC", "15"))

# --- Spread gate (2026-08-09) --------------------------------------------------
# Skip a signal when the bid/ask gap on the X-Perp we actually trade is wider
# than this. The spread is paid in full entering and again exiting, before the
# market moves at all, and the stop is only ~2% away — so 0.25% of spread is
# already an eighth of the risk, twice.
#
# Not a market prediction, which is why this is a gate and the price-pattern
# hypotheses tested the same day are not: the number is known exactly at
# decision time and measures whether the instrument can be transacted, not
# where it will go.
#
# Live spreads span ~7000x across the scannable universe (BTC 0.0001%, AEON
# 0.68%), and — the reason a volume-ranked universe cannot fix this — turnover
# does NOT predict spread: BICO is #3 in the world by 24h volume and still
# shows 0.54%, worse than the HOME trade that prompted this. It is #3 on the EU
# venue too, so ranking by local volume fails identically.
#
# 25bp chosen to match STALE_ENTRY_MAX_ADVERSE_PCT (0.25%): we already allow
# price to drift that far from the signal before refusing to publish, so an
# instrument whose spread alone eats that entire budget does not belong.
# Blocked setups are tagged 'wide_spread' and keep being shadow-resolved, so
# the cost of this gate is measurable in the cap-impact panel.
SPREAD_GATE_ENABLED = os.getenv("SPREAD_GATE_ENABLED", "1") != "0"
SPREAD_MAX_BPS      = float(os.getenv("SPREAD_MAX_BPS", "25"))

# --- Counter-structure marker (2026-08-13) -------------------------------------
# Telemetry only: written to setup_log, shown to nobody, read by no gate. It
# exists so the split stays measurable — the user asked for no label on the
# signal, and there is none.
#
# Marks entries that cut AGAINST the 15m swing (a LONG while structure is
# bearish, a SHORT while it is bullish). The only marker validated twice, on
# two different populations:
#   seed, 10,300 trades, filled at the zone:  83.1%/+0.552R vs 80.7%/+0.442R,
#     same sign every year 2022-2026
#   after ZONE_WATCH, 1,353 trades:           77.3%/+0.286R vs 73.8%/+0.168R,
#     same sign in both windows (75%/+0.232 and 79%/+0.339)
# Frequency ~1 setup in 6.
#
# ⚠️ Replaced the previous definition (stop < 2.09 ATR AND no full 1h/4h
# agreement), which measured 90% win rate while the bot chased the price and
# fell to 71.1%/+0.162R — BELOW the 73.8% baseline — once ZONE_WATCH began
# filling at the zone. Its stop-distance condition selected for roughly what
# waiting for the zone now selects for, so the two overlapped and the edge
# disappeared. SNIPER_MAX_STOP_ATR is gone with it. A marker validated on one
# population is not validated on another.
SNIPER_TAG_ENABLED = os.getenv("SNIPER_TAG_ENABLED", "1") != "0"

# --- Stale-entry guard (VALIDATED 2026-07-31, default ON) ----------------------
# The strategy's edge lives in entering AT the FVG/OB retest zone. The backtest
# always fills there by construction, but live the bot re-anchors the entry to
# the current price at publish time — so if price has already left the zone, it
# published a chase entry the backtest never modelled. Measured cost of that
# adverse entry (17000 candles, 20 symbols, same config, only entry shifted):
#   0.00%: WR 82.8%  netR +945R  maxDD  -16R
#   0.25%: WR 76.0%  netR +590R  maxDD  -39R
#   0.50%: WR 71.2%  netR +379R  maxDD  -58R
#   0.85%: WR 61.1%  netR  +25R  maxDD -108R   <- a real live signal (AAVEUSDT)
#   1.50%: WR 49.1%  netR -387R  maxDD -421R
# Brutal sensitivity, because the stop sits only ~2% away: a 0.85% adverse entry
# burns ~42% of the stop distance before the trade even starts. The old guard
# allowed 3% drift — deep inside the loss-making region.
#
# Guard: if the setup has a real FVG/OB zone, the live price must still be
# INSIDE it (plus a tolerance as a fraction of the zone's own width). Outside
# the zone the retest premise is void, so the signal is skipped rather than
# published as a chase. Adverse entry is then bounded by half the zone width
# (~0.3% typical) instead of the old flat 3%.
STALE_ENTRY_GUARD = os.getenv("STALE_ENTRY_GUARD", "1") != "0"
# Extra tolerance beyond the zone edge, as a fraction of the zone width.
STALE_ENTRY_ZONE_TOLERANCE = float(os.getenv("STALE_ENTRY_ZONE_TOLERANCE", "0.25"))
# Hard numeric backstop for setups with no usable zone (entry_source=MARKET),
# expressed as a fraction of the trade's own risk distance — a principled cap
# on "how much of the stop may be burned on entry" rather than a flat percent.
STALE_ENTRY_MAX_RISK_FRAC = float(os.getenv("STALE_ENTRY_MAX_RISK_FRAC", "0.25"))

# Absolute floor on that tolerance, as a fraction of price. The table above is
# measured in PERCENT OF PRICE; the zone-width rule is not, and on a narrow 15m
# FVG (~0.2-0.4% wide) 0.25 * width lands near 0.05-0.10% — an order of
# magnitude tighter than anything measured. Observed 2026-08-01: the guard
# blocked 12 of 12 Claude-approved setups in 32h, i.e. zero signals published.
# Skipping is not free — at 0.25% adverse the same trades still return +590R at
# WR 76.0%, versus 0R for a signal never sent — so the guard must only stop the
# chases the table shows as ruinous (0.85%+), not every tick outside the zone.
STALE_ENTRY_MAX_ADVERSE_PCT = float(os.getenv("STALE_ENTRY_MAX_ADVERSE_PCT", "0.0025"))

# --- BTC correlation filter ---
BTC_BLOCK_THRESHOLD_PCT = 1.0

# --- Close-confirmed stop (VALIDATED 2026-07-26, default ON) --------------------
# A stop that fires on a WICK touching the level exits trades that never really
# broke — the classic stop-hunt. Requiring the 15m candle to CLOSE beyond the
# level instead keeps those trades alive. Backtested honestly: the surviving
# stops exit at the candle CLOSE, which is worse than the level (avg ~-1.13R,
# never beyond -2.0R across ~3600 trades), so the gain is real, not accounting.
#   window 1 (2026-01→07, 1841tr): WR 80.8→82.8%, SL 347→304, netR +904→+945
#   window 2 (2025-08→2026-01, 1764tr): WR 78.0→80.6%, SL 377→326, netR +720→+762
#                                       maxDD -20.19R → -17.21R
# Holds on both independent windows on WR, profit AND (window 2) drawdown.
# Live effect should be LARGER than backtest: the backtest runs on the deep
# global feed, while the user trades the thin X-Perp where false wicks are more
# common (this is the same phenomenon the SL-wick diagnostic was built to log).
# 🔴 A note added here earlier today claimed that an autotrading account gets a
# touch-triggered stop from the exchange OCO, and that this flag therefore
# described the wrong bot. That was WRONG and is retracted. src/autotrader.py
# moves the exchange stop OUT to STOP_EXCHANGE_BACKSTOP_R from the real fill
# exactly so it cannot fire on the wicks this rule exists to ignore; the
# exchange leg is a disaster backstop for when the bot is down. The working stop
# is the monitor's, and it is close-confirmed. =1 is correct here.
#
# What survives from that detour: BT_EXCHANGE_STOP in backtest.py models
# touch-triggered stops as a research handle (default OFF), and it measures what
# moving the backstop onto the signal level would cost — on the stocks book that
# is about -7.6% of profit and 2.4-7.0pp of win rate, on the hostile crypto
# window -32%. That is the price of the close-confirmed rule, and it is large.
STOP_CLOSE_CONFIRM = os.getenv("STOP_CLOSE_CONFIRM", "1") != "0"

# Require the DEEP feed to confirm a stop before closing, not just the X-Perp
# the position trades on. The monitor already computes this -- it is the
# sl_xperp_only diagnostic -- and then closes anyway, so the information is
# there and unused.
#
# Measured on the live week of 2026-09-02: 3 of 15 crypto stops (20%) and 6 of
# 22 stocks stops (27%) were breached on the X-Perp alone. Those are not thin
# symbols -- GOOGL, META, TSLA, ETH -- and in stocks they cost MORE than real
# stops (-1.697R against -1.321R), because a spike closes far past the level
# and a close-confirmed exit follows it down. Skipping them would move the
# live stop rate from 38.5% to 30.8% (crypto) and 44.9% to 32.7% (stocks),
# which is most of the remaining gap against the model.
#
# The catch, stated plainly: nobody knows what those trades would have done
# next. Held instead of closed, they run until the deep feed agrees or the
# exchange backstop fires at STOP_EXCHANGE_BACKSTOP_R -- so a -1.1R stop can
# become -1.5R (crypto) or -2.5R (stocks). The counterfactual is not in the
# data and cannot be backtested: the model only ever sees the deep feed, so it
# already behaves as if this were on. That is also the argument FOR it -- with
# it on, live and model use the same evidence for the same decision.
#
# Default OFF. This is the account owner's call, same class as
# LEVELS_FROM_STRUCTURE.
STOP_REQUIRE_GLOBAL_CONFIRM = os.getenv("STOP_REQUIRE_GLOBAL_CONFIRM", "0") != "0"

# Where TP1/TP2/SL are anchored when the fill drifts away from the signal price.
#
# 0 (current): levels are recomputed FROM the live fill. The setup fires on a
#   break, the order lands ~2 minutes later with price extended a median 0.22%
#   (crypto) / 0.37% (stocks), and the stop is then placed relative to THAT —
#   so for a long it sits higher in absolute terms than the structure justifies,
#   and an ordinary pullback takes it out. This is why a fresh trade so often
#   shows red immediately, and why a long and a short can both stop out in the
#   same chop: each side's stop has been dragged toward the noise.
#
# 1: levels stay anchored to the pre-drift price — the structure the setup was
#   actually read from. Only the fill moves.
#
# Measured on five windows at the observed 37.4 bps drift:
#            win rate      profit      stops
#   from fill  58-66%     +196.44R    34-41%
#   structure  72-76%     +144.45R    24-28%
# Win rate +13pp and a third fewer stops, but 26% less profit — the anchored
# stop sits closer, so each trade both risks and earns less. Risk-adjusted it is
# mixed across windows: 05-07 strongly favours the current behaviour (ulcer
# ratio 7.8 against 1.0), 08-26 strongly favours anchoring (6.1 against 2.5).
#
# ON since 2026-08-31 by the account owner's decision: he repeatedly reported
# trades opening red and stopping out in both directions, which is exactly
# what a fill-anchored stop produces. Fewer stops chosen over headline
# profit. Same profit-for-drawdown axis as SIZE_MULT_MAX and TP1_R_MULT.
LEVELS_FROM_STRUCTURE = os.getenv("LEVELS_FROM_STRUCTURE", "1") != "0"
# The exchange-side stop stays in place as a DISASTER backstop at this multiple
# of R. It protects the position when the bot itself is down (deploy, restart,
# network) — without it a close-confirmed stop leaves the position naked and 10x
# leverage liquidates ~9% away.
#
# 2.0 -> 1.5 on 2026-08-22, after it fired TWICE within three minutes on live
# money (DOGE and PUMP, both longs opened 08:07 Riga, both backstopped by ~08:10).
# The old comment here claimed 2.0R "would not have fired once in either
# backtest window, so it costs nothing in normal operation". That was wrong on
# both halves: mae_r (added 2026-08-16) shows two trades reaching 2.0R in the
# 1773-trade window, and at 10x a 2R stop on a 3%-risk trade is -60% of that
# trade's margin — the opposite of costing nothing.
#
# Measured on wick depth before TP1, which is exactly what a trigger order sees:
#   уровень   сделок глубже   из них ВЫИГРЫШНЫХ
#   1.20R          138             6 (4%)
#   1.30R           73             2 (3%)
#   1.50R           25             0 (0%)
#   2.00R            2             0 (0%)
# No winning trade in 1773 ever wicks 1.5R against itself, so a backstop there
# kills nothing that close-confirm was protecting — it only truncates 25 tails.
# Net R is marginally BETTER for it (+919.3R -> +921.8R); tighter still keeps
# gaining (+924.1R at 1.3R) but starts costing real winners, so 1.5 is the last
# level that is unambiguously free.
#
# Why this matters more live than in backtest: the engine's own stop waits for a
# 15m candle to CLOSE beyond 1R, and in a fast dump price does not wait. The
# backstop, not the engine, is what actually exits — so its level IS the real
# stop in exactly the cases that hurt most.
STOP_EXCHANGE_BACKSTOP_R = float(os.getenv("STOP_EXCHANGE_BACKSTOP_R", "1.5"))

# --- Concurrent same-direction exposure cap (2026-07-26) ------------------------
# Nothing capped total open positions before this: only per-symbol dedup and
# MAX_SIGNALS_PER_SCAN=3, while signals live up to SIGNAL_EXPIRY_HOURS=48, so
# same-side positions accumulate — portfolio_sim.py measured a peak of TWENTY
# open in one direction. Alts run ~0.7-0.9 correlated to BTC with beta above 1,
# so those are not 20 independent trades: one BTC move resolves them together.
#
# Set to 8 (not 4) after measuring it — see portfolio_sim.py, which replays
# backtest trades on one shared account with the live guardrails applied:
#   cap off : w1 +597.4R / DD -11.17R   w2 +480.1R / DD -8.46R
#   cap 4   : w1 +521.6R / DD -10.41R   w2 +410.2R / DD -5.85R
#   cap 8   : w1 +595.1R / DD -11.17R   w2 +479.1R / DD -8.46R
# A tight cap costs 12-15% of profit for a drawdown benefit that is wildly
# inconsistent between windows (-6.8% vs -31%), and caps of 5-6 came out WORSE
# than no cap at all on w1 — i.e. the drawdown effect is mostly noise. Worse,
# the trades a tight cap removes are BETTER than average (0.632R vs the 0.491R
# mean on w1; same direction on w2): a crowded book forms in a strong trend,
# and trend trades win more, so the cap bites exactly where the edge is.
# 8 costs 0.2-0.4% and still bounds the disaster case. It is deliberately
# dormant in normal markets — its job is the crash that is NOT in this data,
# which is precisely why the cap level cannot be tuned on it.
# 8 -> 6 on 2026-08-29, after the backtest's kill-switch replay stopped
# peeking at future outcomes (see backtest.apply_live_gates). That peek was
# deleting the tail of every bad patch, which hid how much of this book's
# drawdown is correlated same-side exposure. Measured honestly on raw pre-gate
# exports of all three windows, at kill=5:
#   cap   2023 net/worst-ratio   2024 net/worst-ratio   current net/worst-ratio
#   8     +116.99R  11.7         +193.16R  22.0         +422.81R  21.4
#   6     +112.20R  12.1         +180.76R  27.0         +419.62R  29.1
#   5     +115.86R  13.5         +168.07R  30.0         +401.77R  36.5
# 6 keeps the trade count identical to the old 8/kill=2 book in every window
# (817/1070/1455 against 820/1081/1454) while raising profit in all three and
# cutting worst-windows hard in two. 8 earns more but pays in drawdown in all
# three; 5 is safer still but costs ~6% of the trades. Tightening this cap is
# the safe direction anyway — it is the correlation rail.
MAX_SAME_DIRECTION_POSITIONS = int(os.getenv("MAX_SAME_DIRECTION_POSITIONS", "6"))

# --- Graded crowding trim: REJECTED, premise was wrong -----------------------
# Idea: the cap above is a cliff — positions 1-8 ride full size, the ninth is
# refused — while the reason for it (correlated alts resolve together) applies
# gradually. So scale size down as same-side exposure builds. Unlike the
# outcome-based version of this thought, the input is genuinely available at
# entry: the open-position count is known, whereas earlier trades' OUTCOMES are
# not (that one produced z=+5.52 purely from look-ahead).
#
# Measured, step per open position, floor 0.5:
#             base                   step 0.08              step 0.15
#   2023  +122.76R 14.1/32.5 | +107.98R 13.7/33.1 |  +96.90R 13.4/33.1
#   cur   +399.10R 48.4/167.8| +330.13R 49.2/162.4| +296.64R 53.1/161.9
# Absolute risk falls hard (worst 8.24R -> 6.71R, ulcer 2.38 -> 2.03) and
# profit falls exactly as much, so the ratios sit still and the two measures
# disagree in opposite directions between windows. That is de-leveraging, not
# edge: it does not tell good trades from bad, it just holds less when holding
# more.
#
# 🔑 The premise is simply false — crowded trades are BETTER, in all three
# windows:
#   same-side open |   2023            2024           current
#         0        | 67.8% +0.145   71.2% +0.135   72.2% +0.365
#         3        | 70.0% +0.096   74.8% +0.240   80.4% +0.318
#         6+       |     -          84.0% +0.372   83.9% +0.443
# A crowded book forms in a strong trend and trend trades win more, so trimming
# for crowding trims the best trades.
#
# This was already written down twenty lines above: the cap's own comment says
# the trades a tight cap removes run 0.632R against a 0.491R mean. Two backtest
# runs went into rediscovering it. Read the neighbouring parameter's reasoning
# before building a new one.
#
# What survives: drawdown clustering is real, but it comes from correlated
# OUTCOMES, not from worse trades. The right tool for that is the hard cap that
# already exists — bounding the disaster case — not graded sizing, which
# penalises quality that is in fact higher.

# --- News filter (per-coin keywords) ---
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
NEWS_BLOCK_KEYWORDS = ["hack", "exploit", "scam", "lawsuit", "sec ", "ban", "delist", "rug"]

# --- Global macro news agent (Groq free tier) ---
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
NEWS_LOOKBACK_HOURS = 2

# --- Economic calendar warning (ForexFactory weekly XML, free) ---
# Warn on a signal when a HIGH-impact macro event (CPI/FOMC/NFP) lands within
# this many hours — high whipsaw risk around scheduled releases.
EVENT_WARN_HOURS = float(os.getenv("EVENT_WARN_HOURS", "3"))

# --- Auto-block symbols with bad recent stats ---
AUTO_BLOCK_ENABLED           = os.getenv("AUTO_BLOCK_ENABLED", "1") != "0"
AUTO_BLOCK_LOOKBACK_TRADES   = int(os.getenv("AUTO_BLOCK_LOOKBACK_TRADES", "20"))
AUTO_BLOCK_MIN_TRADES        = int(os.getenv("AUTO_BLOCK_MIN_TRADES", "8"))
AUTO_BLOCK_MAX_PROFIT_FACTOR = float(os.getenv("AUTO_BLOCK_MAX_PROFIT_FACTOR", "0.80"))
AUTO_BLOCK_MAX_WIN_RATE      = float(os.getenv("AUTO_BLOCK_MAX_WIN_RATE", "35"))
AUTO_BLOCK_DAYS              = int(os.getenv("AUTO_BLOCK_DAYS", "7"))

# --- Database ---
DB_PATH = os.getenv("DB_PATH", "signals.db")  # Railway: set DB_PATH=/data/signals.db

# --- Backtest ---
BACKTEST_CANDLES        = int(os.getenv("BACKTEST_CANDLES", "1152"))  # 1152 × 15m ≈ 12 days
# 192 × 15m = 48h, matching live SIGNAL_EXPIRY_HOURS. Was "48" (=12h, 4x too
# short) since the project's early days — drifted silently from live's expiry
# as SIGNAL_EXPIRY_HOURS was tuned over time and this wasn't updated alongside
# it. Fixed 2026-07-18: the wrong window was inflating EXPIRED-outcome trades
# and understating both win rate (63.0%→72.7% corrected) and true drawdown
# (-22.24R→-32.15R corrected) in every backtest run before this fix.
BACKTEST_TP_WINDOW      = int(os.getenv("BACKTEST_TP_WINDOW", "192"))
# BACKTEST_TOP_COINS removed 2026-08-03 — it was dead and actively misleading.
# `--top` defaults to 0, so the backtest does NOT pick the top-N by volume: it
# runs the hand-pinned BACKTEST_SYMBOLS list in backtest.py (18 coins), or
# whatever env BACKTEST_SYMBOLS / --symbols / --top override it with. The
# pinned list is deliberate — a volume-ranked set would change under us and
# make runs non-reproducible — but it does mean results carry survivorship
# bias (coins that mattered in 2022 and died are absent by construction) and
# do not match the live universe, which scans TOP_COINS_COUNT dynamically.
# Costs are the single largest drag on this strategy — measured over 1758 trades
# they take 333R against gross winnings of 1437R (23%), MORE than the 284R the
# stops take (20%). So the number has to be right.
#
# FEE was 0.001 (0.1% per side) until 2026-08-16. That is Binance's classic SPOT
# taker rate, inherited when this bot still ran on that feed, and it was never
# revisited when execution moved to OKX EU X-Perps. OKX Lv1 perpetual futures is
# 0.020% maker / 0.050% taker; zone-watch fires a MARKET order, so taker applies.
# The old value charged double the real fee. Correcting it: costs 333R -> 222R,
# net +819.9R -> ~+931R on the same trades.
#
# EVERY figure recorded before this date was measured at the doubled fee and is
# therefore CONSERVATIVE, not wrong in direction — comparisons between variants
# were run at the same setting and still hold.
#
# SLIPPAGE measured 2026-08-20 and DELIBERATELY LEFT AT 0.05%/side.
#
# Four order-book snapshots over ~3 minutes across the pinned universe, then
# weighted by how many trades each coin actually produced (1603 of 1773 trades
# covered; SEI and LAB returned no book):
#   trade-weighted half-spread   2.65 bps = 0.027% per side
#   model                        5.00 bps = 0.050% per side
#   raw median 1.49 · 90th 6.20 · 95th 20.00 · max 25.01 bps
# The distribution is heavily skewed — BILLUSDT alone sits at 23.74 bps across
# 52 trades and drags the weighted figure up from the 1.49 bps median.
#
# So the model is roughly 1.9x conservative, and it stays that way on purpose:
#   * snapshots were taken in calm tape; entries fire on zone touches, which are
#     by definition moments when price is moving and books widen;
#   * a book snapshot captures half-spread only — it says nothing about the
#     latency between decision and fill, which is real slippage we do not model
#     anywhere else;
#   * 10% of trades are on coins that returned no book at all.
# Unlike the fee, which was verifiably wrong against OKX's published schedule,
# this is an estimate weighed against an estimate. Erring high on costs is the
# safe direction. Recorded here so the number is measured rather than assumed.
# 2026-09-03: both rates were 0.0005 and both were wrong, by a lot.
# FEE: the account owner's actual per-side fee is 0.0001, five times lower than
# what was assumed here.
# SLIPPAGE: 0.0005 per side was never measured. The honest cost of crossing is
# half the spread, and the spread was measured live on the venue — crypto perps
# above $5M turnover have a median spread of 0.0172% (half = 0.0086%), and the
# stock X-Perps 0.011% (half = 0.0055%). Rounded up to 0.0001 per side for both.
# Round trip therefore goes from 0.20% to 0.04% of notional.
#
# This matters far beyond tidiness. Cost in R is round-trip / risk-in-price, so
# at a 2.14% median stop the old rates charged 0.0934R per trade against a live
# gross edge of +0.091R — i.e. the model believed the whole edge was eaten by
# costs. On the real rates the charge is 0.0187R and the live book is clearly
# profitable: crypto +19.90R gross over 218 trades becomes +15.8R net, stocks
# +27.44R over 168 becomes +21.9R.
# Every table recorded before this date was measured with costs 5x too high.
# Overstated costs bias the model toward fewer trades and wider stops, so
# cost-sensitive decisions need re-checking, not just re-baselining.
BACKTEST_FEE_RATE       = float(os.getenv("BACKTEST_FEE_RATE", "0.0001"))
BACKTEST_SLIPPAGE_RATE  = float(os.getenv("BACKTEST_SLIPPAGE_RATE", "0.0001"))
# BACKTEST_USE_BTC_FILTER removed 2026-08-03 — dead, and it implied the BTC
# context was optional in backtest. It is not: since the 2026-07-31 fix
# backtest.py always computes a REAL btc_change_pct per scan bar, exactly as
# live does. There is no toggle and there should not be one — a backtest that
# can silently run without the live BTC filters is how the 100%-BTC-bonus bug
# went unnoticed for the project's whole life.

# --- Autotrading (real OKX EU orders for allow-listed users) ---
AUTOTRADE_ENABLED           = os.getenv("AUTOTRADE_ENABLED", "1") != "0"
AUTOTRADE_LEVERAGE          = int(os.getenv("AUTOTRADE_LEVERAGE", "10"))
AUTOTRADE_BALANCE_THRESHOLD = float(os.getenv("AUTOTRADE_BALANCE_THRESHOLD", "100"))
# Per-symbol position-size multiplier. Format: "BTCUSDT:0.5,ETHUSDT:1.0".
#
# BTC earns money but shakes: measured 2026-08-16 over 1758 backtest trades it
# stops out at ~2x the rate of every other coin, and does so in all four
# quarters of the window (24.2 / 25.0 / 20.0 / 30.0% against 9.4 / 14.4 / 13.7 /
# 16.6% for the rest). It is 7.6% of trades and 3% of profit, yet the portfolio's
# worst drawdown shrinks from -11.83R to -8.67R when it is removed.
#
# Halved rather than dropped, deliberately: BTC is +25.8R over the window, so it
# is not a losing coin. And the elevated stop rate is confirmed four times over
# while the drawdown contribution rests on ONE cluster of 11 consecutive trades —
# strong evidence for trimming size, weak evidence for cutting the coin.
# Measured effect of halving: +819.9R -> +807.0R (-1.6%), DD -11.83R -> -10.09R
# (-15%), profit-per-drawdown 69.3 -> 79.9.
def _parse_symbol_size_mult(raw: str) -> dict:
    out = {}
    for part in (raw or "").split(","):
        if ":" not in part:
            continue
        sym, _, val = part.partition(":")
        try:
            out[sym.strip().upper()] = max(0.0, float(val))
        except ValueError:
            continue
    return out

SYMBOL_SIZE_MULT = _parse_symbol_size_mult(
    os.getenv("SYMBOL_SIZE_MULT", "BTCUSDT:0.5")
)

# Size multiplier for counter-structure setups (the `sniper` flag: LONG while the
# 15m swing is bear, SHORT while it is bull). NOT a filter — every setup still
# trades, this only sizes the validated ones up.
#
# Counter-structure is the only marker validated three separate times on three
# different populations: the 10,300-trade seed (83.1% vs 80.7%), the zone-watch
# set (77.3% vs 73.8%), and a 65-feature screen on 2026-08-16 where it was one of
# only two survivors of a both-halves test.
#
# Measured 2026-08-16 on the trade set that survives the live gates — the book
# production can actually carry, not the raw backtest headline. 228 of 1248
# trades are counter-structure, 89.0% WR against 84.8%, +0.572R against +0.450R:
#   x1.00  +589.3R  DD -6.46R  ratio  91.2
#   x1.25  +621.9R  DD -6.46R  ratio  96.2
#   x1.50  +654.6R  DD -6.46R  ratio 101.3   <- +11% profit, worst drawdown unchanged
#   x1.75  +687.2R  DD -6.61R  ratio 104.0
#   x2.00  +719.8R  DD -7.46R  ratio  96.5
#
# 1.5 is the last value that leaves the WORST drawdown untouched. Be honest about
# what it does cost: in the first half alone drawdown deepens -4.68R -> -5.76R.
# It is not free, it is free at the peak. 1.75 buys a little more at the price of
# exceeding the baseline drawdown, and past 2.0 the ratio falls apart.
#
# Sizing does not change WHICH setups fire or how they resolve, so this does not
# confound the still-pending live validation of the zone-watch entry.
# 🔴 2026-08-26 — the "+22% at equal risk" claim above is a fantasy-fill
# number from before the 2026-08-23 execution fix. Re-measured on honest data
# it is NEGATIVE at equal risk in two windows out of three. Full sweep, as
# profit / (profit per worst-windows) / (profit per ulcer):
#
#   mult        2023                  2024                  current
#   1.0    85.92R 10.8 26.0     146.48R 30.7 60.7     336.17R 37.7 145.4
#   1.25   90.83R 11.0 26.4     153.89R 29.9 59.1     353.90R 34.9 137.1
#   1.5    95.96R 11.3 26.7     161.14R 28.4 58.0     369.51R 32.4 127.7
#
# Monotone in every window, and 2023 runs OPPOSITE to the other two: there a
# bigger multiplier improves risk, in 2024 and current it degrades it. 1.25 is
# plain interpolation, not a sweet spot — the response is linear.
#
# The feature itself is real: counter-structure trades run +0.664R against
# +0.286R on the current window, t=+2.50. The two facts are consistent. Those
# trades are better ON AVERAGE but more dispersed, and they are correlated
# with each other, so sizing them up concentrates risk faster than it adds
# expectancy. Same shape as the averaging experiment: a knob that raises mean
# AND variance together is leverage, not edge.
#
# NOT changed here. This is a bare profit-for-risk trade with no free lunch:
# dropping to 1.0 costs 9-10% of raw profit in every window, which fails the
# stated goal of more profit, and the version that pays — remove the
# concentrated bet and carry the freed risk evenly across the whole book,
# worth +16.5%/+7.6%/-4.8% at UNCHANGED drawdown — means raising base risk
# per trade, i.e. changing live position sizes. That is the account owner's
# call, not a tuning decision. Left at 1.5 pending it.
# 🔴 1.5 -> 1.0 on 2026-08-27. Re-measured on the current config (the earlier
# numbers predate five size rules added overnight):
#   mult    2023 profit/worst    2024 profit/worst    current profit/worst
#   1.5    +133.10R  15.9       +190.26R  32.0       +403.27R  51.3
#   1.25   +128.92R  15.5       +185.04R  34.6       +392.70R  60.1
#   1.0    +123.39R  15.2       +177.84R  36.0       +377.46R  66.1
#
# Removing it costs 6.4-7.3% of profit and buys a large risk reduction where
# it matters most in practice: the current window's profit-per-worst-windows
# goes 51.3 -> 66.1 and its max drawdown -11.06R -> -8.26R, a quarter less.
# 2024 improves too. The hostile window gives back 4.4%, which is small against
# that.
#
# The "+22% at equal risk" claim this multiplier was originally shipped on came
# from the pre-2026-08-23 fantasy-fill model and is void.
#
# NOT paired with a base-risk increase, and that is a correction of what was
# proposed yesterday. The idea was to remove the concentrated bet and scale the
# whole book up to restore the old risk level, estimated at +16.5%. Computed
# exactly, no single scale factor exists: removal cuts absolute worst-windows
# by 27% in the current window, 17% in 2024 and only 3% in 2023. Scaling by the
# current window's factor would raise hostile-window risk by a third; scaling
# by the hostile window's factor recovers almost nothing. The estimate was made
# through an average and did not survive the arithmetic.
COUNTER_STRUCTURE_SIZE_MULT = float(os.getenv("COUNTER_STRUCTURE_SIZE_MULT", "1.0"))

# Score relief for counter-structure setups, 2026-08-26. The marker is the most
# validated thing in this system — same sign in all five years of the 10,300
# trade seed, and again on the 1,353-trade zone-watch population — and on the
# current book it is the best group there is: 78.5% WR / +0.469R against a
# +0.313R base, rising to 83.6% / +0.668R when the entry is also close to the
# break.
# We cannot manufacture more of them; the market decides when a retest cuts
# against the swing. What we CAN do is stop discarding them for want of a point
# or two of MTF score, which is the only lever that adds trades in the BEST
# category rather than the average one.
COUNTER_STRUCTURE_SCORE_BONUS = int(os.getenv("COUNTER_STRUCTURE_SCORE_BONUS", "0"))
# Session and HTF-context size multipliers, measured 2026-08-24 on the honest
# model over the combo book (924 trades). Both groups beat base expectancy in
# BOTH window halves: LONDON 145 сд +0.385R, trend_4h=neutral 59 сд +0.365R at
# 84.8% WR, against base +0.244R. Loading them 1.5x is worth +22% at equal risk
# (+343.9 vs +280.9) with drawdown DOWN (-7.60 vs -8.02).
# Deliberately NOT the maximum: neutral x3.0 scores higher still (+401) but that
# is 59 trades carrying triple weight and the gain rides on this window's
# drawdown path avoiding them; LONDON x2.5+ puts the whole gain in the second
# half (first half falls BELOW base). Monotone improvement under concentration
# is the same tell that exposed the trail bug — see memory exit-mechanics-sweep.
# 2026-08-25: extended beyond LONDON. Sessions were re-examined by their
# contribution to the WORST STRETCHES rather than to the mean, after max
# drawdown turned out to be one week — see memory drawdown-is-one-week.
# Two sessions carry clustered losses, i.e. their stops arrive back-to-back
# rather than spread out, which is what actually builds a drawdown:
#   NEW_YORK   52 стопов, 23 подряд против 14.7 ожидаемых  (+56%)
#   OFF_HOURS 105 стопов, 43 подряд против 31.7            (+36%)
#   LONDON / OVERLAP / DEAD_ZONE — at or below chance
# They are also the two overrepresented in the five worst 25-trade stretches
# (OFF_HOURS 45.6% of them against 37.7% of the book; NEW_YORK 24.0% vs 20.0%).
SESSION_SIZE_MULT = {
    "LONDON":    float(os.getenv("LONDON_SIZE_MULT", "1.5")),
    "OFF_HOURS": float(os.getenv("OFF_HOURS_SIZE_MULT", "0.75")),
    "NEW_YORK":  float(os.getenv("NEW_YORK_SIZE_MULT", "1.0")),
    "OVERLAP":   float(os.getenv("OVERLAP_SIZE_MULT", "1.0")),
}
# Neutral 4h context rides 1.5x. Re-validated 2026-08-26 because the original
# justification (line ~293: "trend_4h=neutral scores 84.8% WR") is a
# fantasy-fill number from before the 2026-08-23 execution fix and therefore
# void. Turning the multiplier OFF, three windows:
#            base 1.5x                      off (1.0)
#   2023   668tr 68.9% +95.96R  11.3/26.7 | 668 68.9% +92.73R  11.4/25.3
#   2024   872tr 73.5% +161.14R 28.4/58.0 | 872 73.5% +156.53R 27.8/56.1
#   cur   1172tr 75.9% +369.51R 32.4/127.7|1172 75.9% +355.77R 30.6/123.6
# Removing it costs 3-4% of profit in EVERY window and worsens profit/ulcer in
# every window. Trades and win rate are identical, as they must be — this
# touches size only.
#
# Note the honest evidence is weaker than it looks per-trade: on the current
# window neutral-4h trades run 80.6% / +0.512R against 75.0% / +0.336R, but
# that is 67 trades and t=+1.24, i.e. not significant on its own, and 2023
# gives t=+0.52 on 26 trades. The feature test has almost no power at that
# sample size. What justifies keeping it is the end-to-end money effect
# reproducing with the same sign and size across three independent windows —
# and the fact that profit per unit of ulcer RISES with the multiplier on, so
# the extra size is landing on better-than-average trades rather than simply
# levering the book up.
HTF_NEUTRAL_4H_SIZE_MULT = float(os.getenv("HTF_NEUTRAL_4H_SIZE_MULT", "1.5"))

# --- London boost is conditional on volume (2026-08-26) -----------------------
# The LONDON 1.5x above fires on every London setup. Split by volume it turns
# out the edge lives entirely in the half with real participation, and this
# replicates across three windows:
#
#              London & vol>=2       thin London          everything else
#   2023    57tr 84.2% +0.631     32tr 62.5% +0.030     611tr 67.4% +0.098
#   2024    76tr 73.7% +0.234     45tr 73.3% +0.203     796tr 73.5% +0.180
#   cur     98tr 84.7% +0.631     78tr 73.1% +0.265    1074tr 75.1% +0.286
#
# Two windows three years apart both land on 84% and +0.631R — more than
# double the book's expectancy — and 2024 gives nothing at any volume level
# but never turns negative. Upside in two regimes, neutral in the third, harm
# in none, which is the shape a size boost should have.
#
# It is not a volume proxy: London is UNDER-represented at high volume (10% of
# the vol>3 trades against 15% of the book). Reading: London open on real
# volume is money arriving; London open on a thin tape is just a clock.
#
# Below the threshold the boost is removed rather than added elsewhere, so
# this REDUCES deployed size — a reallocation away from a no-edge subset, not
# extra leverage. That distinction is what separated it from
# COUNTER_STRUCTURE_SIZE_MULT, which raised mean and variance together.
# --- 1h-neutral context (2026-08-26) -----------------------------------------
# HTF_NEUTRAL_4H_SIZE_MULT above boosts a neutral 4h. The comment at line ~293
# spotted BOTH neutral timeframes back in July, but only the 4h one ever got a
# multiplier. On honest data the 1h is the stronger of the two, on two to three
# times the sample:
#
#              trend_1h=neutral            trend_4h=neutral (has the 1.5x)
#   2023     105tr 77.1% +0.237          35tr 68.6% +0.292   (book 68.9%)
#   2024      89tr 86.5% +0.461 t=+3.55  43tr 79.1% +0.287
#   cur      130tr 82.3% +0.443          72tr 83.3% +0.621
#
# 1h-neutral beats the book in all three windows (+8.2/+13.0/+6.4pp win rate),
# including the hostile one where the 4h version does nothing at all.
#
# Found by tools_interactions.py, which ranks pairs by their WEAKEST window:
# trend_1h=neutral occupied five of the top twelve positive pairs, against
# score>=15, low RSI, late extension, low efficiency and FVG entries alike —
# i.e. it is not carried by any one partner.
#
# Default 1.0 = off until measured end-to-end. Adding size to a subset is what
# COUNTER_STRUCTURE_SIZE_MULT did before it turned out to be leverage; the
# difference here is that this subset is positive in the hostile window too,
# which is exactly where counter-structure was empty.
# Measured end-to-end, three windows, against the base that already has the
# London gate in it:
#             base                          + 1h-neutral 1.25
#   2023   668tr 68.9%  +96.55R 12.4/27.6 | 668 68.9% +102.24R 12.8/28.7
#   2024   872tr 73.5% +158.55R 27.3/58.8 | 872 73.5% +166.80R 28.4/61.6
#   cur   1172tr 75.9% +362.62R 38.8/143.1|1172 75.9% +374.50R 40.7/151.0
#
# Profit rises in all three (+5.9/+5.2/+3.3%) AND both risk measures rise in
# all three (+3-5%). No trade-off anywhere, which nothing else measured today
# managed.
#
# That combination is also the test that separates this from
# COUNTER_STRUCTURE_SIZE_MULT. Both add size to a subset, but the risk ratios
# are scale-normalised: leverage lifts profit while pushing the ratios DOWN,
# which is exactly what counter-structure did. Here they go up.
# 1.25 -> 1.5 after sweeping it. 1.5 beats 1.25 on all six numbers:
#            1.25                          1.5
#   2023   668tr +102.24R 12.8/28.7 | 668 +106.30R 12.9/29.1
#   2024   872tr +166.80R 28.4/61.6 | 872 +173.95R 28.7/63.5
#   cur   1172tr +374.50R 40.7/151.0|1172 +384.06R 42.4/157.3
# Ratios still RISING at 1.5, so the concentration cost has not bitten yet.
# The logic: scaling the whole book leaves the ratios flat, scaling a bad
# subset pushes them down, scaling a genuinely better one pushes them up.
# There is a ceiling to that — past some multiplier the subset dominates the
# book and correlation inside it starts setting the drawdown — so this is
# swept rather than assumed.
# Swept to the turn rather than stopped at the first value that worked:
#   mult    2023: profit / worst / ulcer     current: profit / worst / ulcer
#   1.0      +96.55R  12.4 / 27.6             +362.62R  38.8 / 143.1
#   1.25    +102.24R  12.8 / 28.7             +374.50R  40.7 / 151.0
#   1.5     +106.30R  12.9 / 29.1             +384.06R  42.4 / 157.3
#   1.75    +109.77R  13.0 / 29.6             +391.73R  43.9 / 162.3
#   2.0     +111.60R  12.8 / 29.5   <- turns  +398.64R  45.4 / 166.6
#
# In the HOSTILE window both risk measures peak at 1.75 and turn at 2.0 —
# profit keeps climbing there while the ratios fall, which is leverage caught
# in the act. The current window never turns, because a benign regime does not
# punish concentration; that is exactly why the hostile one decides.
#
# Two checks that this is selection and not leverage:
#   * the ceiling binds on only 4.7-7.3% of trades, so SIZE_MULT_MAX is not
#     distorting the top of the sweep;
#   * going 1.0 -> 2.0 raises the book's AVERAGE multiplier from 1.015 to
#     1.044, i.e. +4.4% of deployed size, while current-window profit rises
#     +9.9%. Leverage moves both together; profit outrunning size by 2x means
#     the extra size is landing on genuinely better trades.
#
# Caveat kept deliberately: the backtest cannot see a crash, and a correlated
# subset carried at 1.75x is precisely what hurts there — the same argument
# MAX_SAME_DIRECTION_POSITIONS makes about itself.
HTF_NEUTRAL_1H_SIZE_MULT = float(os.getenv("HTF_NEUTRAL_1H_SIZE_MULT", "1.75"))

# --- Trimming the BEARISH 1h context: REJECTED ------------------------------
# trend_1h sorts cleanly in all three windows, so having sized the top of that
# ordering up it looked obvious to size the bottom down:
#              2023    2024   current      (book 68.9 / 73.5 / 75.9)
#   neutral   77.1%   86.5%   82.3%        boosted, HTF_NEUTRAL_1H_SIZE_MULT
#   bullish   67.8%   73.0%   76.3%        level with the book
#   bearish   66.3%   69.7%   73.2%        below it in all three
#
# At 0.75x:
#              base                          bearish x0.75
#   2023   668tr +109.77R 13.0/29.6 | 668 +101.03R 12.9/28.7   both worse
#   2024   872tr +173.95R 28.7/63.5 | 872 +170.91R 31.1/64.8   both better
#   cur   1173tr +392.82R 44.1/162.8|1173 +367.58R 47.1/165.8  both better
# Profit falls in every window (-8.0/-1.7/-6.4%) and the HOSTILE window, the
# one weighted highest, loses on both risk measures.
#
# Why it fails where the thin-London trim succeeded: thin London ran +0.265R
# against a book of +0.286R — nothing to lose by trimming. A bearish 1h is 29%
# of the book and is still solidly profitable on its own, so trimming it cuts
# live money to buy a small risk improvement.
#
# Mirror of the stocks clean-trend rejection: a large subset can be neither
# boosted nor trimmed. Boost it and you have leverage; trim it and you are
# cutting earnings. Sizing rules need a SMALL subset whose edge is genuinely
# near zero (to trim) or genuinely far above book (to boost).

# Measured end-to-end, three windows:
#              base                          London boost gated on volume
#   2023   668tr 68.9% +95.96R  11.3/26.7 | 668 68.9% +96.55R  12.4/27.6
#   2024   872tr 73.5% +161.14R 28.4/58.0 | 872 73.5% +158.55R 27.3/58.8
#   cur   1172tr 75.9% +369.51R 32.4/127.7|1172 75.9% +362.62R 38.8/143.1
# Profit is flat (-1.9% to +0.6%) while risk falls: both measures improve in
# two windows, the current one by 20% and 12%, and its max drawdown goes
# -15.30R -> -12.79R. Worst case is 2024 at -3.9% on one measure.
LONDON_VOL_MIN       = float(os.getenv("LONDON_VOL_MIN", "2.0"))  # 0 = off

# --- OVERLAP runs the other way (2026-08-27) ---------------------------------
# The session-times-volume rule that works for London and for the stocks bot's
# opening bell does NOT generalise. Checked on every crypto session, lift in
# R/trade over the rest of that window:
#
#   session    volume |    2023      2024     current
#   LONDON     >=2.0  |  +0.533    +0.054     +0.344    <- shipped above
#   LONDON      <2.0  |  -0.120    +0.019     -0.054
#   OVERLAP     <2.0  |  +0.528    +0.036     +0.315    <- INVERTED
#   OVERLAP    >=2.0  |  +0.028    -0.052     -0.021
#   OFF_HOURS   <2.0  |  -0.170    -0.089     -0.101
#   NEW_YORK    <2.0  |  -0.428    -0.026     -0.027
#
# At the London/New-York overlap the CALM half is the good one, in all three
# windows. Reading: the overlap is already the busiest hour, so volume on top
# of it is not money arriving, it is a fight — while a quiet overlap means the
# two sessions agree. The mechanism is not "volume is good" but "volume
# against the norm FOR THAT HOUR", and a flat 2.0 threshold is a lot for
# London and little for the overlap.
#
# Both general versions of that idea were tested and are dead: thin volume as
# a standalone feature is negative in all three windows but decays monotonically
# (-0.179 -> -0.072 -> -0.048) across 36-43% of the book, and volume measured
# against each session's own median gives +0.088 -> +0.027 -> +0.017 while
# splitting the book in half. Neither is usable.
#
# Default 1.0 = off until measured end-to-end. The subset is 4.3% of the book
# (28/33/55 trades per window) and the 2024 lift is only +0.036, so the
# evidence is thin even though the sign holds three times.
# Robust across thresholds rather than perched on one — the lift is positive
# in all three windows at every cut from 1.75 to 3.0:
#   thr    2023            2024            current       share
#   1.75  84.2% +0.354   87.5% +0.217   83.3% +0.290     2.6%
#   2.00  82.1% +0.528   75.8% +0.036   87.3% +0.315     4.3%
#   2.25  87.2% +0.570   75.6% +0.049   85.9% +0.339     5.3%
#   2.50  84.3% +0.417   76.1% +0.106   85.0% +0.282     6.5%
#   3.00  83.3% +0.342   72.4% +0.066   81.4% +0.161     7.9%
#
# 2.5 is chosen by ranking on the WEAKEST window, the same rule the pair
# scanner uses. 2.0 was measured first purely because it is a round number,
# and it is the worst cut on that criterion (+0.036 against +0.106) on half
# the sample. End-to-end 2.5 beat it on all six numbers.
#
# Measured at 1.5x, three windows pinned by date:
#             base                          calm overlap
#   2023   668tr +109.77R 13.0/29.6 | 668 +120.78R 13.9/32.3
#   2024   872tr +173.95R 28.7/63.5 | 872 +184.32R 28.5/63.5
#   cur   1167tr +384.42R 43.1/159.1|1167 +400.95R 46.8/166.2
# Profit up in all three (+10.0/+6.0/+4.3%), both risk measures clearly better
# in two and flat in the third (-0.7% and 0.0%). Max drawdown -12.44R -> -12.09R.
OVERLAP_VOL_MAX        = float(os.getenv("OVERLAP_VOL_MAX", "2.5"))
# Swept upward; no turn appears, but that is the CEILING talking, not the
# market. Share of calm-overlap trades pinned at SIZE_MULT_MAX:
#   1.5  -> 28%      1.75 -> 46%      2.0 -> 61%
# so past 1.5 the nominal multiplier is increasingly not applied as written.
#   mult   2023: profit/worst/ulcer     current: profit/worst/ulcer
#   1.0    +109.77R 13.0 / 29.6          +384.42R 43.1 / 159.1
#   1.5    +120.78R 13.9 / 32.3          +400.95R 46.8 / 166.2
#   1.75   +124.37R 14.2 / 33.0          +405.43R 47.6 / 167.1
#   2.0    +127.46R 14.3 / 33.4          +409.17R 48.1 / 167.9
# Gains flatten accordingly: worst-windows moves +2.2% from 1.5 to 1.75 and
# +0.7% from 1.75 to 2.0. 1.75 takes the meaningful part.
#
# Side effect that got its own look: SIZE_MULT_MAX now binds on 8-9% of the
# book, so the ceiling had quietly become a live parameter without anyone
# deciding that. Swept, and it stays at 2.0:
#   ceiling   2023 worst/ulcer   2024 worst/ulcer   current worst/ulcer
#   2.0       14.2 / 33.0        26.7 / 62.3        47.6 / 167.1
#   2.5       14.1 / 32.6        25.5 / 62.1        52.9 / 173.8
#   3.0       13.8 / 32.3          -                52.8 / 174.3
# Raising it helps only the benign window; both historical windows lose on
# both measures. Principle agrees with the measurement here: this is a safety
# rail whose job is the crash that is NOT in this data — the same argument
# MAX_SAME_DIRECTION_POSITIONS makes about itself — so raising it because a
# calm regime likes it would be tuning the insurance to the weather.
OVERLAP_CALM_SIZE_MULT = float(os.getenv("OVERLAP_CALM_SIZE_MULT", "1.75"))

# --- Active chop rides bigger (2026-08-27) -----------------------------------
# Found by the triple search, and it names no session and no trend — it
# describes a market STATE: high volume, LOW efficiency ratio, high ATR
# percentage. Price is busy and violent but not travelling cleanly.
#
#   volume>=2.11 & eff_ratio<0.4476 & vol_atr_pct>=0.006268
#     2023  114tr 79.8% lift +0.230
#     2024  132tr 77.3% lift +0.148
#     cur   152tr 76.3% lift +0.260
#   398 trades, 14.7% of the book, no size or decay flag.
#
# Win rate runs 10.9 / 3.8 / 0.6 points over each window's book. It agrees with
# a note already in this file at EFF_RATIO_MAX: a very clean trend at entry
# means the move is already extended and the retest being bought is late. The
# triple reaches the same idea from the other side — messy is not bad, messy
# WITH participation is where the edge sits.
#
# Default 1.0 = off until measured end-to-end.
CHOP_VOL_MIN     = float(os.getenv("CHOP_VOL_MIN", "2.11"))
CHOP_EFF_MAX     = float(os.getenv("CHOP_EFF_MAX", "0.4476"))
CHOP_ATR_MIN     = float(os.getenv("CHOP_ATR_MIN", "0.006268"))
# Measured end-to-end, three windows pinned:
#            base                          x1.25                  x1.5
#   2023  +122.76R 14.1/32.5 | +132.73R 14.9/35.0 | +139.95R 15.2/36.0
#   2024  +186.22R 28.3/63.8 | +194.98R 29.5/65.2 | +201.24R 29.3/64.9
#   cur   +399.10R 48.4/167.8| +409.99R 50.0/168.2| +415.63R 51.0/166.3
#
# 1.25 improves profit AND both risk measures in ALL THREE windows — a clean
# sweep, which almost nothing measured over two days has managed. 1.5 earns
# more profit but the cracks start: 2024 gives back a little on both ratios
# and the current window's ulcer turns negative, so the concentration turn
# sits between the two. Taking the value with no damage anywhere, on a subset
# this fresh, beats chasing the extra profit into the first breakage.
CHOP_SIZE_MULT   = float(os.getenv("CHOP_SIZE_MULT", "1.25"))

# --- Open space rides smaller (2026-08-27) -----------------------------------
# room_atr is the distance to the nearest higher-timeframe level. Far from any
# level is WORSE, consistently, in all three windows:
#   thr>=3.5   2023  67tr 62.7% lag -0.131
#              2024  74tr 67.6% lag -0.119
#              cur   98tr 72.4% lag -0.072
#   8.8% of the book; win rate 5-6 points under each window's own book.
#
# Counterintuitive at first — "plenty of room to run" sounds good. It is not:
# with the nearest structure 3.5+ ATR away the setup is in open space with
# nothing to lean on. The level matters as a REFERENCE, not as an obstacle.
#
# 3.5 chosen by ranking on the weakest window (-0.072 against -0.060 at 4.0
# and -0.051 at 3.0); 5.0 lags harder still but holds only 38-46 trades per
# window.
#
# Unlike the thin-London trim this subset IS profitable on its own (+0.026 /
# +0.076 / +0.249 R per trade), so trimming costs real money and has to pay
# for it in risk. Default 1.0 = off until that is measured.
OPEN_SPACE_ROOM_MIN  = float(os.getenv("OPEN_SPACE_ROOM_MIN", "3.5"))
# Measured, three windows pinned:
#            base                       x0.75                  x0.6
#   2023  +132.73R 14.9/35.0 | +131.92R 15.1/36.3 | +130.82R 15.0/36.6
#   2024  +194.98R 29.5/65.2 | +192.71R 31.6/66.8 | +191.32R 32.9/67.6
#   cur   +409.99R 50.0/168.2| +407.36R 50.5/172.2| +404.90R 50.3/174.0
# At 0.75 all SIX numbers improve — three windows by two measures — for 0.6%
# to 1.2% of profit. 0.6 buys more risk reduction at twice the profit cost;
# the milder intervention is taken on a finding this new.
#
# Worth contrasting with the worst-corner trim rejected the same night for
# being immaterial. The magnitudes here are comparable, but there one window
# sat flat and its ulcer went backwards — a mixed bag — while this is six of
# six, and the mechanism is a dimension nothing else in the config touches.
OPEN_SPACE_SIZE_MULT = float(os.getenv("OPEN_SPACE_SIZE_MULT", "0.75"))

# --- Parabolic arc rides smaller (2026-08-28) --------------------------------
# The "parabolic arc" pattern, reduced to a number: how many times steeper the
# recent leg is than the one before it. The claim is that a trend accelerating
# like this has buyers in full control until they exhaust themselves, and then
# it retraces. Measured across three windows, the claim holds directionally:
#   accel    2023 lag   2024 lag   current lag   share
#   >=2.5     -0.027     -0.103      -0.028      23.1%
#   >=4.0     +0.031     -0.056      -0.027      14.9%
#   >=6.0     -0.065     -0.122      -0.034       9.5%
#   >=8.0     -0.098     -0.171      -0.001       7.0%
#
# 6.0 is the cut: negative in all three windows and 9.5% of the book, inside
# the band where trims have worked. 4.0 flips positive in 2023 and 2.5 covers a
# quarter of the book.
#
# Default 1.0 = off until measured end-to-end.
PARABOLIC_ACCEL_MIN  = float(os.getenv("PARABOLIC_ACCEL_MIN", "6.0"))
# Measured, three windows:
#            base                       x0.75                  x0.6
#   2023  +136.43R 17.6/44.1 | +134.12R 19.4/46.7 | +133.03R 20.7/49.6
#   2024  +187.48R 40.5/82.9 | +185.43R 40.6/85.0 | +184.19R 40.7/86.0
#   cur   +379.65R 72.4/198.4| +372.43R 72.5/196.0| +368.00R 72.5/194.2
#
# The hostile window gains most (+10.2% worst-windows, +5.9% ulcer at 0.75),
# 2024 gains on both, the current window is flat. Costs 1-2% of profit and no
# trades at all — the book stays 694/901/1193.
#
# 0.75 rather than 0.6, the milder intervention on a new finding, as with every
# other rule shipped this week. 0.6 gives the hostile window considerably more
# (+17.6% / +12.5%) if a more defensive setting is ever wanted.
#
# Worth recording that the pattern's own claim was PRE-REGISTERED before
# measuring: an accelerating trend should make entries WORSE because buyers
# exhaust themselves. It does, in all three windows, and the effect strengthens
# monotonically with steepness (-0.027 at 2.5x, -0.122 at 6x in 2024). A
# claim that only worked at one threshold would have been a coincidence.
PARABOLIC_SIZE_MULT  = float(os.getenv("PARABOLIC_SIZE_MULT", "0.75"))

# --- Zone age: feature REVIVED, sizing rule REJECTED -------------------------
# zone_age_bars had been declared, plumbed through signal_filter and written to
# every trade export since the feature was created — and was ALWAYS -1, because
# nothing computed it: signal_filter asked ind["bullish_fvg_age"] and the
# indicators never returned such a key. Fixed in detect_fvg/detect_order_block.
#
# True distribution once a SECOND instance of the same bug was fixed (the export
# wrote `int(setup.get(...) or -1)`, and 0 is falsy, so every age-0 trade was
# recorded as "no zone"):
#   age   2023 lag    2024 lag   current lag   share
#    0     -0.114      -0.023      -0.123      53.6%
#    1     -0.160      -0.104      -0.006      15.4%
#    2     +0.084      -0.039      +0.169       6.8%
#    3        -        +0.160      +0.209       3.9%
# Older zones are better; the freshest lag. But age 0 is HALF THE BOOK, so
# trimming it is de-leveraging rather than selection.
#
# Trimming age 1 alone (15.4%, lagging in all three windows) was measured:
#            base                       x0.75
#   2023  +123.39R 15.2/37.2 | +122.13R 15.6/37.5
#   2024  +177.84R 36.0/72.6 | +175.28R 34.4/71.8
#   cur   +377.46R 66.1/195.4| +366.01R 68.2/199.0
# Profit falls in all three, 2024 loses on BOTH measures, and the gains
# elsewhere are 0.8-3.2%. Not worth a rule.
#
# The FIXES stay: the indicator now reports zone age, and _fld() replaced nine
# `x or default` reads where a legitimate zero was being swallowed. That bug
# class cost two contradictory measurements before the arithmetic gave it away
# — identical trade sets cannot produce different numbers.

# --- Thin dead zone rides smaller (2026-08-27) -------------------------------
# DEAD_ZONE is 16% of the book and has never had a multiplier — SESSION_SIZE_MULT
# lists London, off-hours, New York and the overlap, so it falls through at 1.0.
# The session as a whole deserves that: lift -0.002 / +0.051 / -0.040, t near
# zero everywhere. Split by volume it is the familiar shape once more:
#              2023      2024     current
#   vol<2.0   -0.265    -0.070    -0.091
#   vol>=2.0  +0.189    +0.165    -0.001
# Third confirmation of the same mechanism, after London and the stocks bot's
# opening bell: a session without volume behind it is just a clock reading.
#
# The thin half is 7.0% of the book and, crucially, is WEAK IN THE HOSTILE
# WINDOW: absolute -0.106R there against a book of +0.144R.
#
# That last point is the criterion, and it corrects what was written after the
# thin-London trim. "Trim when absolute expectancy is near zero" does not fit
# the evidence — thin London ran +0.265 against a book of +0.286 and trimming
# WORKED, while the bearish-1h subset ran +0.264 against ~+0.30 and it FAILED.
# Near-identical profiles, opposite outcomes. The real difference is what the
# trim does: thin London removed an unjustified BOOST (1.5x back to 1.0x, not
# below base), whereas bearish-1h cut BELOW base. Cutting below base needs the
# subset to be weak specifically in the hostile window, where there is least
# margin — open-space qualified (+0.026 against +0.199 there) and bearish-1h
# did not (+0.191 against +0.144, i.e. ABOVE its book).
DEAD_THIN_VOL_MAX   = float(os.getenv("DEAD_THIN_VOL_MAX", "2.0"))
# Measured, three windows pinned:
#            base                       x0.75                  x0.6
#   2023  +131.92R 15.1/36.3 | +133.10R 15.9/38.0 | +133.80R 16.1/38.9
#   2024  +192.71R 31.6/66.8 | +190.26R 32.0/66.6 | +188.93R 32.3/66.3
#   cur   +407.36R 50.5/172.2| +403.27R 51.3/173.5| +400.46R 51.8/173.7
#
# The HOSTILE window gains profit as well as both risk measures (+5.3% and
# +4.7%), which is exactly what the corrected trim rule predicts: the subset
# is NEGATIVE there (-0.106R against a book of +0.144R), so cutting below base
# is safe. The other two windows give up about 1% of profit for 0.8-1.6% on
# worst-windows; 2024's ulcer gives back 0.3%.
#
# 0.75 rather than 0.6 — the milder intervention, same precedent as elsewhere.
DEAD_THIN_SIZE_MULT = float(os.getenv("DEAD_THIN_SIZE_MULT", "0.75"))

# --- Worst corner of the book: measured, NOT shipped -------------------------
# Top of the triple search's negative list and narrow enough to act on:
#   session=OFF_HOURS & trend_1h=bearish & bos extension>=1.04
#     2023  49tr 57.1%   2024  55tr 60.0%   cur 101tr 68.3%
#   205 trades, 7.6% of the book, 7-14 points of win rate below each window.
#
# Trimming it to 0.75x:
#            base                    x0.75
#   2023  +132.73R 14.9/35.0 | +132.93R 14.9/34.8
#   2024  +194.98R 29.5/65.2 | +194.66R 30.5/66.2
#   cur   +409.99R 50.0/168.2| +407.87R 50.4/168.3
#
# This PASSES the usual bar — profit flat, both measures better in two windows
# — and is still not worth shipping. The moves are 0.1% to 3.4%, indistinct
# from noise, and the mechanism explains why: two of the three conditions
# already carry trims, so this corner is riding at 0.75 x 0.75 = 0.5625 before
# the new rule touches it. There is little left to cut.
#
# Recorded as a criterion, not just a result: past "does it work" there is
# "does it earn its complexity". The sizing stack grew by four rules in one
# night, and each further one adds surface to maintain and another chance of
# interacting with something else. A rule that does almost nothing is worse
# than no rule.
LONDON_THIN_SIZE_MULT = float(os.getenv("LONDON_THIN_SIZE_MULT", "1.0"))
# Per-coin tier multiplier, 2026-08-24. Coins ranked by expectancy on the
# 923-trade book, bottom third 0.75 / top third 1.25, middle untouched.
# Kept SEPARATE from SYMBOL_SIZE_MULT so the BTC trim (measured on its own,
# see memory btc-size-trim) and this ranking can be reverted independently.
#
# The strength is the whole point. Validated OUT OF SAMPLE — fit on one half of
# the window, scored on the other, both directions:
#   0.75/1.25  →  +6.3% and +2.9%   both halves better
#   0.50/1.50  →  -10.6% and -2.2%  both halves WORSE
# Aggressive multipliers bet on the per-coin number, which is ~50 trades of
# noise; mild ones bet only on the ordering, which holds. Fitting on the full
# window and scoring on it flatters this to +15.2% — that figure is an artefact.
# LAB sits in the top tier on 22 trades; that is an outlier the mild multiplier
# is allowed to carry, not a conviction.
#
# 🔴 2026-08-27 — audited against the historical windows, and most of the top
# tier CANNOT be checked at all: LAB, TAO, ZEC and SEI did not trade in 2023.
# No backtest fixes that; the data does not exist. Of the two that can be
# checked, LINK agrees in one window only and ETH CONTRADICTS outright —
# +0.590R per trade in the current window against -0.208R in the hostile one.
#
# ETH's boost removed on that basis, three windows:
#             base                          ETH at 1.0
#   2023   668tr +124.37R 14.2/33.0 | 668 +122.76R 14.1/32.5
#   2024   872tr +186.00R 26.7/62.3 | 872 +186.22R 28.3/63.8
#   cur   1167tr +405.43R 47.6/167.1|1167 +399.10R 48.4/167.8
# Profit is flat (-1.3/+0.1/-1.6%) while both risk measures improve in two
# windows and give back 0.7%/1.5% in the third — a free risk reduction.
#
# The other three boosts are LEFT IN PLACE deliberately. Turning the whole top
# tier off was measured too: worst-windows improves in all three windows but
# ulcer only in one, and profit falls 2-7% — no clean verdict either way. And
# that test is weak by construction, since in the historical windows the thing
# under test barely exists. Unvalidatable is not the same as contradicted, so
# only the contradicted one was removed.
SYMBOL_TIER_MULT = _parse_symbol_size_mult(os.getenv(
    "SYMBOL_TIER_MULT",
    "BTCUSDT:0.75,ADAUSDT:0.75,SEIUSDT:0.75,SOLUSDT:0.75,DOTUSDT:0.75,"
    "LINKUSDT:1.25,TAOUSDT:1.25,ZECUSDT:1.25,LABUSDT:1.25"))
# Ceiling on the PRODUCT of every size multiplier. They stack: a top-tier coin
# in LONDON with a neutral 4h reaches 1.25*1.5*1.5 = 2.81x, a concentration
# nothing here was measured at. Each multiplier was validated on its own; their
# product was not.
# 2026-08-29 RE-MEASURED, and the sweep recorded above is WRONG. It was run
# while the kill-switch replay was peeking at future outcomes, so every risk
# ratio it compared was three to four times too kind, and unevenly so. Redone on
# uncapped exports (SIZE_MULT_MAX=100, then clamped offline — the ceiling is a
# pure clamp, so one export answers every candidate exactly):
#   cap    2023 net  worst/ulcer      2024 net  worst/ulcer     current net  worst/ulcer
#   2.0    +112.20   12.1 / 24.8      +180.76   27.0 / 59.0      +419.62   29.1 / 126.6
#   2.5    +116.21   12.5 / 25.0      +184.66   28.1 / 61.0      +435.02   28.5 / 124.5
#   3.0    +118.48   12.7 / 25.1      +186.54   28.4 / 61.6      +440.63   27.7 / 121.3
# The old claim was "raising helps only the benign window; both historical
# windows lose on both measures". The truth is the reverse: BOTH historical
# windows, the hostile one included, gain on both ratios, and it is the CURRENT
# window that gives them back. Profit rises everywhere (+3.3% in total at 2.5).
#
# Why it would work: the cap binds on 3-5% of the book, and those trades earn
# 2.6-4x the book's per-trade edge at unit size (+0.361 against +0.088 in 2023,
# +0.427 against +0.164 in 2024), 18 of 29 of them OVERLAP — the rule with the
# largest measured edge. The ceiling is clamping exactly the trades that have
# earned size.
#
# NOT RAISED ANYWAY, and deliberately. This is a rail whose job is the crash
# that is not in this data, and that argument never depended on the risk
# figures, so correcting them removes a reason to keep 2.0 without supplying a
# reason to leave it. +3.3% of profit against a 25% wider worst-case position is
# the account owner's call, not a measurement's — same class as TP1_R_MULT. The
# numbers above are here so the decision can be made on true ones.
SIZE_MULT_MAX = float(os.getenv("SIZE_MULT_MAX", "2.0"))
# Extension trim, 2026-08-25. bos_extension_atr is how far price has travelled
# from the level where structure broke, in ATR — i.e. how LATE the entry is.
# It is the one microstructure feature that survived: expectancy falls monotone
# across its quartiles with both window halves agreeing at every step —
#   <=0.53 +0.366R (75.8%) · 0.53-1.02 +0.352 · 1.02-1.49 +0.260 · >1.49 +0.214
#   (68.3%) — against a base of +0.298R.
# Trimmed rather than filtered: the worst bucket is still PROFITABLE, and
# dropping profitable trades drops profit (see EFF_RATIO_MAX, tested and
# rejected the same way).
# Threshold and multiplier are deliberately mild. The equal-risk response has a
# stable plateau at 0.9-1.2 under BOTH x0.75 and x0.6, and falls off a cliff at
# 1.3 under both; 1.2 sits inside the plateau away from that edge. The gain is
# mostly drawdown reduction driven by ~96 trades in a narrow band, so treat the
# expectancy edge as the real part and the headline as optimistic.
EXTENSION_ATR_THRESHOLD = float(os.getenv("EXTENSION_ATR_THRESHOLD", "1.2"))
EXTENSION_SIZE_MULT     = float(os.getenv("EXTENSION_SIZE_MULT", "0.75"))

# Volatility boost, under test 2026-08-25. vol_atr_pct is ATR as a share of
# price. The top quintile is the strongest single group found anywhere in this
# book — 184 сд, 77.2% WR, +0.494R against a +0.298R base, both halves
# +0.517/+0.474, and no loss clustering (9 consecutive against 9.6 expected).
# Held to a HIGHER bar than usual because the response is NOT monotone (the
# fourth quintile is the worst of the five) and because I have no mechanism for
# it: the obvious one argues the other way — in high volatility the structural
# stop wants to be wider, hits the RISK_MAX_PCT clamp, and should therefore sit
# tighter than intended. Default 1.0 until it earns its place.
VOL_ATR_BOOST_THRESHOLD = float(os.getenv("VOL_ATR_BOOST_THRESHOLD", "0.0104"))
# Shipped at 1.25 on 2026-08-25. Against the OFF_HOURS-trimmed book:
#   base       +313.2R  окна 3.54 (88.4)  ulcer 1.60 (195.5)
#   x1.25      +336.3R  окна 3.67 (91.6)  ulcer 1.65 (203.7)
#   x1.5       +354.9R  окна 3.92 (90.6)  ulcer 1.72 (206.5)
# Both ratios improve at both multipliers and profit rises 7.4%/13.3%; risk
# rises less than profit does, which is what a boost has to prove. The one
# blemish is the first half's worst-windows ratio, 45.7 -> 45.4, i.e. -0.7%
# — noise, but the reason for taking the mild multiplier and not 1.5.
# Re-tested 2026-08-28 against removal (mult=1.0) on all three pinned windows.
# KEPT — the current window vetoes removal:
#             profit    worst/ulcer ratios
#   2023   +136.43R ->  +132.09R    17.6/44.1 -> 19.6/47.7   removal better
#   2024   +185.43R ->  +180.72R    40.6/85.0 -> 41.4/88.0   removal better
#   2026   +379.65R ->  +351.01R    72.4/198  -> 64.3/185    removal WORSE, -7.5% profit
#
# 🔑 This rule also refutes a tempting way of auditing the sizing book. Per-trade
# R at equal size says the boosted subset is 79% BELOW book in 2023 (+0.032 vs
# +0.150) and 26% below in 2024 — on that view alone it looks like leverage on a
# weak subset and should go. It should not: the boost was fitted against
# DRAWDOWN (losses arrive in clusters, see the memory note on that), and removing
# it deepens drawdown in two of three windows. A rule fitted to one measure must
# be judged on that measure. Do not re-litigate this from per-trade R.
VOL_ATR_BOOST_MULT      = float(os.getenv("VOL_ATR_BOOST_MULT", "1.25"))

# --- Risk-normalised sizing (added 2026-08-22 after a live incident) ---
# Position size was fixed regardless of how far away the stop sat, so a trade
# with 1R = 3.0% of price risked 2.5x the money of one with 1R = 1.2% for the
# same "size". Stop width across 1773 backtest trades: min 1.20%, median 1.47%,
# max 3.00%, with 20.4% of trades pinned at the 3.0% ceiling.
#
# At 10x that means a 1R loss costs 12% of the trade's margin on a tight stop
# and 30% on a wide one. On 2026-08-22 DOGE and PUMP both had 1R = 3.0% — the
# ceiling — so each cost 60% of its margin when the 2R backstop fired
# (-2.12 and -2.11 USDC on ~3.4 USDC of margin).
#
# 🔑 THE DEEPER PROBLEM: every backtest figure in this project is in R, and
# summing R assumes each R is worth the same money. The live bot broke that
# assumption, so live dollar outcomes carry variance the backtest never showed.
# This is the same class of live-vs-backtest gap as the missing gates — the
# model was right, the execution did not match it.
#
# Scaling is DOWNWARD ONLY: a wide stop gets less size, a tight stop is left
# alone rather than levered up. That is strictly risk-reducing and needs no
# further validation to be safe. The symmetric version (sizing tight stops UP
# to hit the reference exactly) would raise exposure and has NOT been measured.
# --- 2026-08-28: the backtest deliberately does NOT mirror this block -------
# Mirroring it looks obviously right and is WRONG. backtest.py sums outcomes in
# R, which already assumes every trade risks the same money; this block is what
# MAKES that assumption true in live. Multiplying R by the same factor a second
# time in the backtest deflates every figure for a risk reduction the R unit
# has already priced in. Written, run, caught, reverted the same night.
#
# Measured over the three pinned windows, money at risk per trade (reference
# units) spans only 0.800 to 1.050, and return per unit of risk differs from
# the flat-R backtest by +0.97% / +1.19% / +1.20%. The accounting is faithful
# to ~1%; there is no gap here worth closing.
#
# ⚠️ TRAP that produced a whole night of false findings before this was caught:
# `cost_r` in the trade export is multiplied by the size multiplier
# (backtest.py: `cost_r *= _sz`), so recovering stop width as
# `round_trip / cost_r` yields stop_width / size_mult, NOT stop width. Small
# values then mean "heavily boosted", not "tight stop" — and boosted trades win
# more BY CONSTRUCTION, so the artefact reproduces across every window, symbol
# and volatility band and looks like the strongest signal in the book.
# Use the `entry` and `sl` columns. Note also that `size_mult` and `risk_mult`
# are DIFFERENT columns; dividing by risk_mult does not remove the size effect.
#
# For reference, the real distribution: stop width is bounded at 1.20% and
# 3.50% of price (median 1.77-2.13%), and per-trade R by stop width is a weak
# inverted U with no consistent peak — the TIGHTEST bucket is the worst in all
# three windows. There is no edge to harvest from stop width.
RISK_NORMALIZED_SIZING = os.getenv("RISK_NORMALIZED_SIZING", "1") != "0"
# Reference 1R width. At or below this, size is untouched; above it, size is cut
# proportionally. Set to the measured median so the typical trade is unaffected.
RISK_REFERENCE_PCT     = float(os.getenv("RISK_REFERENCE_PCT", "0.015"))
# Floor on the multiplier, so an extreme stop cannot shrink a position below the
# exchange minimum and silently drop the trade.
RISK_SIZE_MULT_MIN     = float(os.getenv("RISK_SIZE_MULT_MIN", "0.45"))

# --- Profit sweep: take money off leverage ------------------------------------
# Everything the bot earns currently becomes collateral for the next 10x trade,
# so profit is never actually banked — it just raises the stake. This offers a
# slice of realised profit for withdrawal into unleveraged spot.
#
# The value here is the DE-LEVERAGING, not the buying. Moving money out from
# under 10x is the one risk control that works without needing any edge. Which
# coin it then sits in is a separate decision the bot has no measured basis for
# — see the note on recommendations below.
#
# Realised PnL is read from OKX (positions-history), never computed from our own
# prices: the engine and the exchange exit at different moments, so our numbers
# are the wrong basis for moving real money.
#
# ⚠️ The bot NEVER buys on its own. It sends an offer with buttons and only acts
# on an explicit press. Declining still advances the watermark, otherwise the
# same profit is re-offered forever.
PROFIT_SWEEP_ENABLED       = os.getenv("PROFIT_SWEEP_ENABLED", "1") != "0"
PROFIT_SWEEP_THRESHOLD_USD = float(os.getenv("PROFIT_SWEEP_THRESHOLD_USD", "20"))
PROFIT_SWEEP_PCT           = float(os.getenv("PROFIT_SWEEP_PCT", "10"))
# Do not pester: minimum hours between offers to the same user.
PROFIT_SWEEP_MIN_GAP_H     = float(os.getenv("PROFIT_SWEEP_MIN_GAP_H", "24"))
# OKX spot minimum order is a few dollars; below this a sweep cannot execute.
PROFIT_SWEEP_MIN_ORDER_USD = float(os.getenv("PROFIT_SWEEP_MIN_ORDER_USD", "2"))
#
# NO RECOMMENDATION ENGINE. The measured edge in this project is a 15m return to
# a zone with a stop and a target, holding hours. It says nothing about which
# coin to buy and hold, and the one time buy-and-hold momentum WAS measured on
# crypto it came out at roughly zero to negative (alts pump and dump — see the
# TUSA FINANCE work). A picker built on that would emit confident suggestions
# with nothing behind them, and would be believed because the rest of this bot's
# numbers are real. The offer therefore shows FACTS per coin (24h volume,
# spread, how this bot has traded it) and lets the user choose.

# --- Approach quality: did price come INTO the zone, or are we chasing? ---
# `approach_pct` (main._attach_approach) is how far price already moved in the
# trade's direction over the previous APPROACH_LOOKBACK_BARS. Collected as
# telemetry only — it does NOT size or filter anything.
#
# 🔴 A sizing rule WAS built on this on 2026-08-20 and reverted the same hour.
# The first measurement looked spectacular — stop rate rising monotonically from
# 6.4% to 25% as pre-entry run-up increased, stable across both halves, 39% of
# trades qualifying for a 1.5x size-up worth +29% profit. All of it was an
# artifact: the analysis located each trade's candle by the `entry_bar` INDEX
# from an exported CSV, while the candle array had been refetched and shifted
# since that export. It was comparing entry prices against unrelated bars.
#
# Caught because a fresh export reproduced none of it. Re-measured with bars
# located by TIMESTAMP, the effect is gone:
#   ниже +0.1%    295 сд  12.2% стопов  +0.624R
#   +0.1..+0.4%   296 сд  15.9%         +0.539R
#   +0.4..+0.7%   295 сд  14.6%         +0.571R
#   +0.7..+1.1%   296 сд  15.5%         +0.461R
#   +1.1..+1.7%   295 сд  12.9%         +0.464R
#   +1.7%+        296 сд  15.5%         +0.452R
# No monotonicity, no consistent split. And there is little to exploit anyway:
# the honest distribution is tight — 10th pct -0.04%, median +0.73%, 90th +2.17%.
#
# 🔑 RULE THIS COST: never index candles by a bar number stored in an artifact.
# Bar indices are only valid against the exact array that produced them; a
# refetch shifts them. Locate by timestamp.
APPROACH_LOOKBACK_BARS  = int(os.getenv("APPROACH_LOOKBACK_BARS", "24"))   # 6h of 15m
AUTOTRADE_CONTACT           = os.getenv("AUTOTRADE_CONTACT", "@sanja_tusagang")
# Fernet key for encrypting user API keys at rest — generate once:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# and set AUTOTRADE_ENC_KEY on the host. Keys are unreadable without it.

# --- Reject cooldown + kill-switch (added after 8-SL chop cluster 2026-07-10) ---
# After Claude rejects a setup, don't re-ask the same symbol+direction while
# price is still in the same zone (1 ATR) — stops "ask every scan until yes".
REJECT_COOLDOWN_HOURS = float(os.getenv("REJECT_COOLDOWN_HOURS", "3"))
# N consecutive SL among today's closed signals → pause new signals until the
# next Riga day. 0 = off.
# 3 -> 2 on 2026-08-26. The kill switch turned out to be the single most
# effective risk control in the system, and the reason is the whole lesson
# of this session: the ~50 trades it blocks at a threshold of 3 average
# almost exactly ZERO, so no analysis of mean expectancy can ever find
# them — but they land inside the worst stretches. Turning it off entirely
# leaves profit unchanged (+334.8R against +336.3R) while worst-windows go
# 3.67 -> 6.15 and ulcer 1.65 -> 2.19.
# Tightening to 2 improves everything except trade count, and passes all
# six checks (both measures, both halves, whole window):
#   kill 3   967 сд  75.4%  +336.3R  окна 3.67 (91.6)  ulcer 1.65 (203.7)
#   kill 2   901 сд  76.8%  +342.1R  окна 2.50 (136.8) ulcer 1.38 (248.6)
# kill 1 was also tested: 79.8% WR — close to the 80% the user keeps asking
# for — but only 650 trades and +285.7R, so it buys the win rate with a
# third of the book and 15% of the profit.
# 2 -> 5 on 2026-08-29. WIDENED, not removed. The backtest replay of this gate
# walked trades in ENTRY order while reading each trade's eventual outcome, so
# it paused the day at the entry of the second trade that would LATER stop out
# — knowledge the live bot cannot have, since live reads get_today_sl_streak()
# over CLOSED signals. Losses cluster on this book, so that peek deleted the
# rest of each bad patch and made a streak of 2 look protective.
# Replayed honestly on raw pre-gate exports, a streak of 2 is the worst setting
# available: fewer trades, less profit AND more risk than no brake at all
# (current window: 1454sd +386.71R worst 18.97 against 1554sd +424.24R worst
# 18.05). 5 recovers essentially all of that — it never fires at all in 2023,
# and keeps 99.6% of the no-brake profit in the current window — while leaving
# a circuit breaker for a catastrophic day that is not in this sample. That is
# the whole reason it is not simply set to 0.
KILL_SWITCH_SL_STREAK = int(os.getenv("KILL_SWITCH_SL_STREAK", "5"))
