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
SIGNAL_COOLDOWN_HOURS = 3  # 15m swing signals hold 2-8h — 3h cooldown per coin/direction

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
SMC_SWING_LOOKBACK    = 5
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
MTF_MIN_SCORE = int(os.getenv("MTF_MIN_SCORE", "14"))

# 2026-07-25: filter-variant experiment only (src/filter_variants.py variant D).
# Setups scoring [SHADOW_MIN_SCORE, MTF_MIN_SCORE) are NOT real signals (never
# sent, never traded — see _shadow_only flag in signal_filter.py) but ARE sent
# to Claude + logged, so variant D (soft score>=12) gets real verdicts instead
# of mirroring variant A. Real signal gate (MTF_MIN_SCORE) is untouched.
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
OVERLAP_BEARISH_1H_GUARD = os.getenv("OVERLAP_BEARISH_1H_GUARD", "1") != "0"

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
EFF_RATIO_LOOKBACK = int(os.getenv("EFF_RATIO_LOOKBACK", "20"))
EFF_RATIO_MIN      = float(os.getenv("EFF_RATIO_MIN", "0.15"))
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
FVG_LONDON_BTC_UP_FILTER  = os.getenv("FVG_LONDON_BTC_UP_FILTER", "1") != "0"
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
STABILITY_FILTERS_ENABLED   = os.getenv("STABILITY_FILTERS_ENABLED", "1") != "0"
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

# Epoch for the LIVE tier of Claude's self-feedback history (unix ts, 0 = off).
# The live tier looks back 30 days, which on 2026-08-01 still reached into the
# pre-parity-fix bot: those trades were entered by a filter with a 50-candle 1h
# lookback (Strong1h could never fire), a 3% entry-drift allowance and a stop
# that fired on wicks. Their record — 17 sent, 3W/14SL — describes software
# that no longer exists, yet it dominated every prompt: Claude quoted it in 6
# of 6 rejections on 31.07, and all 6 of those rejected setups went on to hit
# TP1 or TP2. Rows before this timestamp are excluded from the live tier only;
# admin stats and the global over-strictness corrector still see everything.
# 1785456000 = 2026-07-31 00:00 UTC, the day the parity fixes shipped.
LIVE_HIST_EPOCH_TS = float(os.getenv("LIVE_HIST_EPOCH_TS", "1785456000"))

# --- Structure-based stops/takes (swing mode, 15m, ~20x leverage) ---
# SL sits at swing invalidation (recent swing low/high) + ATR buffer, then
# clamped to safe leverage bounds. TPs are R-multiples for swing-sized moves.
#   risk%  ~1.2–3.0% of price  → on 20x = 24–60% margin at risk per stop
#   TP1 = 1.5R (1.8–4.5% move → 36–90% on 20x), close 50%, move SL to BE
#   TP2 = 3.0R (3.6–9%   move → 72–180% on 20x), let winner run
ATR_PERIOD    = 14
SL_ATR_BUFFER = float(os.getenv("SL_ATR_BUFFER", "0.5"))   # buffer beyond swing, in ATR
RISK_MIN_PCT  = float(os.getenv("RISK_MIN_PCT", "0.012"))  # min SL distance = 1.2%
RISK_MAX_PCT  = float(os.getenv("RISK_MAX_PCT", "0.03"))   # max SL distance = 3.0% (20x safe)
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
TP1_R_MULT    = float(os.getenv("TP1_R_MULT", "0.7"))      # TP1 = entry ± risk * 0.7
TP2_R_MULT    = float(os.getenv("TP2_R_MULT", "2.0"))      # TP2 = entry ± risk * 2.0 (was 3.0 — unreachable)

# Runner exit after TP1: trail the remaining 50% by ATR instead of fixed TP2.
# Backtest (10 sym, 2880x15m): +21% net R, -27% max drawdown, same win rate vs
# fixed TP2. Trailing stop = peak ∓ TRAIL_ATR_MULT×ATR, floored at breakeven.
TRAIL_RUNNER_ENABLED = os.getenv("TRAIL_RUNNER_ENABLED", "1") != "0"
TRAIL_ATR_MULT       = float(os.getenv("TRAIL_ATR_MULT", "0.25"))  # base trail; post_tp1_v2 overrides per-context

# Exit profile: "post_tp1_v2" keeps the FULL position past TP1 (TP1_CLOSE_FRAC=0)
# and trails by an ATR multiple chosen from the TP1-acceptance candle — strong
# follow-through trails wide (let it run), weak/rejected trails tight (lock).
# Validated 3 windows on our cache (90/180/365d): net R +80/+91/+124% with LOWER
# drawdown, win rate / trades / SL count UNCHANGED — it only changes how winners
# are harvested, never which trades are taken. "fixed" = legacy 50%-at-TP1 + BE.
TP1_CLOSE_FRAC = max(0.0, min(1.0, float(os.getenv("TP1_CLOSE_FRAC", "0.0"))))
EXIT_PROFILE   = os.getenv("EXIT_PROFILE", "post_tp1_v2").strip().lower()
POST_TP1_STRONG_TRAIL_ATR_MULT = float(os.getenv("POST_TP1_STRONG_TRAIL_ATR_MULT", "0.35"))
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

# --- Sniper tag (VALIDATED 2026-08-09, walk-forward) ---------------------------
# A LABEL on the normal stream, never a filter: it does not add, remove or alter
# a single signal. It marks the subset that historically resolved best, so the
# position can be sized differently by hand.
#
# Two conditions, both declared from mechanism BEFORE looking at the test years,
# and both of which came out OPPOSITE to the strategy's own stated thesis:
#   1. stop distance < SNIPER_MAX_STOP_ATR of the instrument's own ATR. The
#      thesis says a tight stop is riskier (noise knocks you out); the data says
#      the opposite, because a stop that is few ATRs away puts TP1 (0.7R) within
#      easy reach in ATR terms. What matters is target reachability, not stop
#      safety.
#   2. 1h and 4h do NOT both agree with the direction. The thesis treats full
#      HTF alignment as the best case; the data says those win LESS. Consistent
#      with the counter-structure result (see claude_analyzer's Str note): this
#      system earns on pullback entries into a zone, not on trend continuation.
#
# Threshold 2.09 = the 25th percentile of stop/ATR measured on 2022-2024 ONLY,
# frozen before the 2025-2026 check. Result, defined-then-tested:
#   train 2022-24  85.7% WR / +0.641R   |   test 2025-26  84.4% WR / +0.637R
# By year (never negative, never below the base): 2023 88.1%/+0.647 ·
# 2024 85.2%/+0.670 · 2025 83.1%/+0.645 · 2026 85.5%/+0.631, against a base of
# 81.1%/+0.461. ~65 trades a year.
#
# ⚠️ This is NOT the 90-100% the idea started from, and it cannot be: an earlier
# search over ~600 feature combinations DID find 90.5% on 2022-24, and every one
# of those rules collapsed to 80.6-83.5% (i.e. the base) on 2025-26. Searching
# for a win rate manufactures it. These two conditions survive precisely because
# they were fixed in advance.
SNIPER_TAG_ENABLED  = os.getenv("SNIPER_TAG_ENABLED", "1") != "0"
SNIPER_MAX_STOP_ATR = float(os.getenv("SNIPER_MAX_STOP_ATR", "2.09"))

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
# The exchange-side stop stays in place as a DISASTER backstop, just widened to
# this multiple of R. It is what protects the position when the bot itself is
# down (deploy, restart, network) — without it a close-confirmed stop would
# leave the position naked and 10x leverage liquidates ~9% away. At 2.0R it
# would not have fired once in either backtest window, so it costs nothing in
# normal operation and caps the tail at -2R instead of a liquidation.
STOP_EXCHANGE_BACKSTOP_R = float(os.getenv("STOP_EXCHANGE_BACKSTOP_R", "2.0"))

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
BACKTEST_FEE_RATE       = float(os.getenv("BACKTEST_FEE_RATE", "0.001"))
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
KILL_SWITCH_SL_STREAK = int(os.getenv("KILL_SWITCH_SL_STREAK", "3"))
