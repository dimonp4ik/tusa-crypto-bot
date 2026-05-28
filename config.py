import os
from dotenv import load_dotenv

load_dotenv()

# --- Required secrets (set in Render environment variables) ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# --- Scan settings ---
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))
TOP_COINS_COUNT = int(os.getenv("TOP_COINS_COUNT", "45"))
TIMEFRAME = "5m"           # 5m candle → signals resolve in 30-90 min
KLINES_LIMIT = 200         # 200 × 5m = ~16 hours of data for SMC

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
SIGNAL_COOLDOWN_HOURS = 1  # 5m signals resolve in 30-90 min — 1h cooldown per coin/direction

# --- KuCoin (accessible from cloud/US servers) ---
KUCOIN_BASE_URL = "https://api.kucoin.com"
QUOTE_ASSET = "USDT"
TIMEFRAME_KUCOIN = "5min"
KLINES_INTERVAL_SEC = 5 * 60

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

# --- Multi-timeframe score gate (max ~15) ---
MTF_MIN_SCORE = int(os.getenv("MTF_MIN_SCORE", "9"))

# --- ATR-based stops/takes ---
ATR_PERIOD   = 14
ATR_SL_MULT  = 1.5   # SL  = entry ± ATR * 1.5
ATR_TP1_MULT = 1.5   # TP1 = entry ± ATR * 1.5  (close 50%, move SL to BE)
ATR_TP2_MULT = 3.0   # TP2 = entry ± ATR * 3.0  (1:2 R:R on remaining 50%)

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
BACKTEST_CANDLES        = int(os.getenv("BACKTEST_CANDLES", "2000"))
BACKTEST_TP_WINDOW      = int(os.getenv("BACKTEST_TP_WINDOW", "24"))
BACKTEST_TOP_COINS      = int(os.getenv("BACKTEST_TOP_COINS", "20"))
BACKTEST_FEE_RATE       = float(os.getenv("BACKTEST_FEE_RATE", "0.001"))
BACKTEST_SLIPPAGE_RATE  = float(os.getenv("BACKTEST_SLIPPAGE_RATE", "0.0005"))
BACKTEST_USE_BTC_FILTER = os.getenv("BACKTEST_USE_BTC_FILTER", "1") != "0"
