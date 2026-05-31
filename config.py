import os
from dotenv import load_dotenv

load_dotenv()

# --- Required secrets (set in Render environment variables) ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# --- Admin panel: Telegram user IDs that can access /admin in DM ---
ADMIN_IDS = {671071896, 1254580347, 1678694315}

# --- Scan settings ---
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))
TOP_COINS_COUNT = int(os.getenv("TOP_COINS_COUNT", "45"))
TIMEFRAME = "15m"          # 15m candle → swing signals, hold 2-8h
KLINES_LIMIT = 200         # 200 × 15m = ~50 hours of data for SMC

# --- Symbol quality filter ---
# ALLOWED_SYMBOLS="" (default) → auto top-volume mode, top 45 by 24h USDT volume.
# Set ALLOWED_SYMBOLS=BTC-USDT,ETH-USDT,... in .env for strict whitelist.
MIN_24H_QUOTE_VOLUME_USDT = float(os.getenv("MIN_24H_QUOTE_VOLUME_USDT", "3000000"))
MAX_SPREAD_PCT            = float(os.getenv("MAX_SPREAD_PCT", "0.20"))

def _parse_symbol_list(value, default=None):
    if not value:
        return list(default or [])
    return [s.strip().upper() for s in value.split(",") if s.strip()]

ALLOWED_SYMBOLS = _parse_symbol_list(os.getenv("ALLOWED_SYMBOLS", ""))
BLOCKED_SYMBOLS = _parse_symbol_list(os.getenv("BLOCKED_SYMBOLS", ""))

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

# --- KuCoin (accessible from cloud/US servers) ---
KUCOIN_BASE_URL = "https://api.kucoin.com"
QUOTE_ASSET = "USDT"
TIMEFRAME_KUCOIN = "15min"
KLINES_INTERVAL_SEC = 15 * 60

# --- 1h candles for trend direction ---
TIMEFRAME_1H_KUCOIN = "1hour"
KLINES_1H_LIMIT = 50
KLINES_1H_INTERVAL_SEC = 3600

# --- 4h candles for higher timeframe bias ---
TIMEFRAME_4H_KUCOIN = "4hour"
KLINES_4H_LIMIT = 30
KLINES_4H_INTERVAL_SEC = 4 * 3600

# --- Trading hours filter (UTC) ---
TRADING_HOURS_START = 7    # 07:00 UTC — London open
TRADING_HOURS_END   = 23   # 23:00 UTC — NY close
TRADE_WEEKENDS      = False

# --- SMC settings ---
SMC_SWING_LOOKBACK    = 5
SMC_FVG_MIN_PCT       = 0.0005
SMC_OB_LOOKBACK       = 30
SMC_MIN_CONFIRMATIONS = int(os.getenv("SMC_MIN_CONFIRMATIONS", "2"))
SMC_BOS_MIN_VOLUME    = float(os.getenv("SMC_BOS_MIN_VOLUME", "1.5"))
SMC_RSI_LONG_MAX      = float(os.getenv("SMC_RSI_LONG_MAX", "72"))   # skip overextended longs
SMC_RSI_SHORT_MIN     = float(os.getenv("SMC_RSI_SHORT_MIN", "28"))  # skip overextended shorts
MAX_SETUPS_TO_CLAUDE  = int(os.getenv("MAX_SETUPS_TO_CLAUDE", "8"))  # only strongest go to Claude

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
MTF_MIN_SCORE = int(os.getenv("MTF_MIN_SCORE", "9"))

# --- Signal-quality filters (backtested on a PINNED 20-coin / ~21-day set) ---
# All three were A/B-tested apples-to-apples and DEFAULT OFF — none beat baseline:
# №1 Volatility regime: lower bound cut only ~3 dead trades → +0.11R vs +0.11R
#    (zero net effect). Upper ceiling actively hurt (cut TP2 runners). Off.
#    NOTE: an earlier "+0.13R" win was a measurement artifact — the live
#    get_top_coins() list reshuffled between runs (SUI drifting in/out). Fixed
#    by pinning BACKTEST_SYMBOLS in backtest.py.
VOL_REGIME_FILTER = os.getenv("VOL_REGIME_FILTER", "0") != "0"
VOL_MIN_ATR_PCT   = float(os.getenv("VOL_MIN_ATR_PCT", "0.0015"))  # <0.15% range = too dead
VOL_MIN_RATIO     = float(os.getenv("VOL_MIN_RATIO", "0.55"))      # cur/median below = collapsed
VOL_MAX_RATIO     = float(os.getenv("VOL_MAX_RATIO", "99"))        # ceiling OFF (hurt R in backtest)
VOL_REGIME_LOOKBACK = int(os.getenv("VOL_REGIME_LOOKBACK", "50"))

# №3 Strong BOS and №4 Structural-only confirmation were BOTH backtested and
# DROPPED (default off): each lowered win rate (37.5% → 35.0%) and Expected R
# (+0.12R → +0.03R). Strong-BOS pushed entries late (momentum spent → SL);
# structural-only cut valid reversals. Flags kept for experimentation.
REQUIRE_STRONG_BOS = os.getenv("REQUIRE_STRONG_BOS", "0") != "0"
STRONG_BOS_VOL_MULT = float(os.getenv("STRONG_BOS_VOL_MULT", "1.3"))  # x SMC_BOS_MIN_VOLUME
REQUIRE_STRONG_CONFIRM = os.getenv("REQUIRE_STRONG_CONFIRM", "0") != "0"

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
# №B Strict HTF alignment — DROPPED (default off). Backtested: 232tr +0.04R/+8R,
#    half of baseline. Cutting counter-trend also cut winners. Flag kept for experiments.
REQUIRE_STRICT_HTF = os.getenv("REQUIRE_STRICT_HTF", "0") != "0"

# --- Claude tiered analysis (cascade: cheap LIGHT gate + rare deep HEAVY) ---
# LIGHT  : Haiku validates every passed setup in ONE cached batch call (JSON via tool).
# HEAVY  : Sonnet re-checks only top setups (score >= HEAVY_MIN_SCORE) with coin memory.
# Caching: static rules block cached 1h → cheap re-reads on the 5-min scan loop.
CLAUDE_LIGHT_MODEL        = os.getenv("CLAUDE_LIGHT_MODEL", "claude-haiku-4-5")
CLAUDE_HEAVY_MODEL        = os.getenv("CLAUDE_HEAVY_MODEL", "claude-sonnet-4-5")
CLAUDE_HEAVY_MIN_SCORE    = int(os.getenv("CLAUDE_HEAVY_MIN_SCORE", "12"))   # score >= → HEAVY 2nd opinion
CLAUDE_HEAVY_MAX_PER_SCAN = int(os.getenv("CLAUDE_HEAVY_MAX_PER_SCAN", "3")) # cost cap per scan
CLAUDE_MEMORY_LIMIT       = int(os.getenv("CLAUDE_MEMORY_LIMIT", "8"))       # recent outcomes per coin (HEAVY)
CLAUDE_MAX_RISK_SCORE     = int(os.getenv("CLAUDE_MAX_RISK_SCORE", "8"))     # counter-arg auto-reject if risk >= this
CLAUDE_CACHE_TTL          = os.getenv("CLAUDE_CACHE_TTL", "1h")              # prompt cache TTL ("5m" or "1h")

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
TP1_R_MULT    = float(os.getenv("TP1_R_MULT", "1.5"))      # TP1 = entry ± risk * 1.5
TP2_R_MULT    = float(os.getenv("TP2_R_MULT", "3.0"))      # TP2 = entry ± risk * 3.0

# --- BTC correlation filter ---
BTC_BLOCK_THRESHOLD_PCT = 1.0

# --- News filter (per-coin keywords) ---
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
NEWS_BLOCK_KEYWORDS = ["hack", "exploit", "scam", "lawsuit", "sec ", "ban", "delist", "rug"]

# --- Global macro news agent (Groq free tier) ---
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
NEWS_LOOKBACK_HOURS = 2

# --- Auto-block symbols with bad recent stats ---
AUTO_BLOCK_ENABLED           = os.getenv("AUTO_BLOCK_ENABLED", "1") != "0"
AUTO_BLOCK_LOOKBACK_TRADES   = int(os.getenv("AUTO_BLOCK_LOOKBACK_TRADES", "20"))
AUTO_BLOCK_MIN_TRADES        = int(os.getenv("AUTO_BLOCK_MIN_TRADES", "8"))
AUTO_BLOCK_MAX_PROFIT_FACTOR = float(os.getenv("AUTO_BLOCK_MAX_PROFIT_FACTOR", "0.80"))
AUTO_BLOCK_MAX_WIN_RATE      = float(os.getenv("AUTO_BLOCK_MAX_WIN_RATE", "35"))
AUTO_BLOCK_DAYS              = int(os.getenv("AUTO_BLOCK_DAYS", "7"))

# --- Database ---
DB_PATH = "signals.db"

# --- Backtest ---
BACKTEST_CANDLES        = int(os.getenv("BACKTEST_CANDLES", "1152"))  # 1152 × 15m ≈ 12 days
BACKTEST_TP_WINDOW      = int(os.getenv("BACKTEST_TP_WINDOW", "48"))
BACKTEST_TOP_COINS      = int(os.getenv("BACKTEST_TOP_COINS", "20"))
BACKTEST_FEE_RATE       = float(os.getenv("BACKTEST_FEE_RATE", "0.001"))
BACKTEST_SLIPPAGE_RATE  = float(os.getenv("BACKTEST_SLIPPAGE_RATE", "0.0005"))
BACKTEST_USE_BTC_FILTER = os.getenv("BACKTEST_USE_BTC_FILTER", "1") != "0"
