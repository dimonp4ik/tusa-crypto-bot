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
SMC_FVG_MIN_PCT       = 0.0005
SMC_OB_LOOKBACK       = 30
SMC_MIN_CONFIRMATIONS = int(os.getenv("SMC_MIN_CONFIRMATIONS", "2"))
SMC_BOS_MIN_VOLUME    = float(os.getenv("SMC_BOS_MIN_VOLUME", "1.5"))
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
EFF_RATIO_MIN      = float(os.getenv("EFF_RATIO_MIN", "0.15"))
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

# --- Risk sizing overlays (DEFAULT ON) -----------------------------------------
# Does not filter trades. Raises risk_mult for contexts that repeatedly showed
# stronger R/trade: OB entries, optimal RSI/vol, strong relative coin momentum.
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
# but cuts winners in strong-trend month (June: +0.625→+0.416). Net slightly
# WORSE for us — cuts 37% of trades without lifting win rate. KEPT OFF.
# Enable only as a conservative/range-market mode after re-validation.
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
SL_ATR_BUFFER = float(os.getenv("SL_ATR_BUFFER", "1.0"))
RISK_MIN_PCT  = float(os.getenv("RISK_MIN_PCT", "0.012"))  # min SL distance = 1.2%
RISK_MAX_PCT  = float(os.getenv("RISK_MAX_PCT", "0.035"))  # max SL distance = 3.5%
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
TP1_R_MULT    = float(os.getenv("TP1_R_MULT", "0.6"))      # TP1 = entry ± risk * 0.6
TP2_R_MULT    = float(os.getenv("TP2_R_MULT", "2.0"))      # TP2 = entry ± risk * 2.0 (was 3.0 — unreachable)

# Runner exit after TP1: trail the remaining 50% by ATR instead of fixed TP2.
# Backtest (10 sym, 2880x15m): +21% net R, -27% max drawdown, same win rate vs
# fixed TP2. Trailing stop = peak ∓ TRAIL_ATR_MULT×ATR, floored at breakeven.
TRAIL_RUNNER_ENABLED = os.getenv("TRAIL_RUNNER_ENABLED", "1") != "0"
TRAIL_ATR_MULT       = float(os.getenv("TRAIL_ATR_MULT", "0.05"))  # base trail; post_tp1_v2 overrides per-context
# 2026-08-24: 0.25 -> 0.05, monotone across the whole range on the honest
# model and confirmed in both halves. The runner earns, but gives it back on
# the pullback; exiting at the first failure to extend keeps +0.029R per
# trail exit over 702 of them. NOT the same as banking at TP1 — TP1_CLOSE_FRAC
# 0.5 was tested and is far worse (+184R vs +236R).

# Exit profile: "post_tp1_v2" keeps the FULL position past TP1 (TP1_CLOSE_FRAC=0)
# and trails by an ATR multiple chosen from the TP1-acceptance candle — strong
# follow-through trails wide (let it run), weak/rejected trails tight (lock).
# Validated 3 windows on our cache (90/180/365d): net R +80/+91/+124% with LOWER
# drawdown, win rate / trades / SL count UNCHANGED — it only changes how winners
# are harvested, never which trades are taken. "fixed" = legacy 50%-at-TP1 + BE.
TP1_CLOSE_FRAC = max(0.0, min(1.0, float(os.getenv("TP1_CLOSE_FRAC", "0.0"))))
EXIT_PROFILE   = os.getenv("EXIT_PROFILE", "post_tp1_v2").strip().lower()
POST_TP1_STRONG_TRAIL_ATR_MULT = float(os.getenv("POST_TP1_STRONG_TRAIL_ATR_MULT", "0.05"))
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
KNN_RISK_OVERLAY   = os.getenv("KNN_RISK_OVERLAY", "1") != "0"
KNN_DEEP_CANDLES   = int(os.getenv("KNN_DEEP_CANDLES", "1000"))   # 1 Bybit page
KNN_MAX_HISTORY    = int(os.getenv("KNN_MAX_HISTORY", "800"))     # analog pool cap
KNN_SHAPE_LEN      = int(os.getenv("KNN_SHAPE_LEN", "12"))        # query window (3h)
KNN_HORIZON        = int(os.getenv("KNN_HORIZON", "16"))          # forward bars (4h)
KNN_K              = int(os.getenv("KNN_K", "40"))                # neighbours
KNN_MIN_HISTORY    = int(os.getenv("KNN_MIN_HISTORY", "120"))     # min bars to score
KNN_HIGH_SCORE     = float(os.getenv("KNN_HIGH_SCORE", "0.55"))   # size-up threshold
KNN_HIGH_MULT      = float(os.getenv("KNN_HIGH_MULT", "1.20"))    # size-up multiplier
KNN_LOW_SCORE      = float(os.getenv("KNN_LOW_SCORE", "0.50"))    # size-down threshold
KNN_LOW_MULT       = float(os.getenv("KNN_LOW_MULT", "0.80"))     # size-down multiplier
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
STOP_CLOSE_CONFIRM = os.getenv("STOP_CLOSE_CONFIRM", "1") != "0"
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
MAX_SAME_DIRECTION_POSITIONS = int(os.getenv("MAX_SAME_DIRECTION_POSITIONS", "8"))

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
BACKTEST_FEE_RATE       = float(os.getenv("BACKTEST_FEE_RATE", "0.0005"))
BACKTEST_SLIPPAGE_RATE  = float(os.getenv("BACKTEST_SLIPPAGE_RATE", "0.0005"))
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
COUNTER_STRUCTURE_SIZE_MULT = float(os.getenv("COUNTER_STRUCTURE_SIZE_MULT", "1.5"))

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
SYMBOL_TIER_MULT = _parse_symbol_size_mult(os.getenv(
    "SYMBOL_TIER_MULT",
    "BTCUSDT:0.75,ADAUSDT:0.75,SEIUSDT:0.75,SOLUSDT:0.75,DOTUSDT:0.75,"
    "LINKUSDT:1.25,ETHUSDT:1.25,TAOUSDT:1.25,ZECUSDT:1.25,LABUSDT:1.25"))
# Ceiling on the PRODUCT of every size multiplier. They stack: a top-tier coin
# in LONDON with a neutral 4h reaches 1.25*1.5*1.5 = 2.81x, a concentration
# nothing here was measured at. Each multiplier was validated on its own; their
# product was not.
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
KILL_SWITCH_SL_STREAK = int(os.getenv("KILL_SWITCH_SL_STREAK", "2"))
