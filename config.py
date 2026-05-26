import os
from dotenv import load_dotenv

load_dotenv()

# --- Required secrets (set in Render environment variables) ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# --- Scan settings ---
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "10"))
TOP_COINS_COUNT = 50
TIMEFRAME = "15m"          # candle size for analysis
KLINES_LIMIT = 100         # number of candles to fetch per coin

# --- Technical filter thresholds ---
RSI_OVERSOLD = 35          # below = potential LONG
RSI_OVERBOUGHT = 65        # above = potential SHORT
VOLUME_SPIKE_MULTIPLIER = 1.8   # current volume must be 1.8x the 20-period average
MIN_SIGNALS_TO_PASS = 2    # coin needs at least 2 bullish or 2 bearish signals

# --- Signal deduplication ---
SIGNAL_COOLDOWN_HOURS = 4  # don't resend same direction for same coin within 4h

# --- KuCoin (accessible from cloud/US servers) ---
KUCOIN_BASE_URL = "https://api.kucoin.com"
QUOTE_ASSET = "USDT"
TIMEFRAME_KUCOIN = "15min"      # KuCoin interval format
KLINES_INTERVAL_SEC = 15 * 60   # 15 minutes in seconds

# --- 1h candles for trend direction ---
TIMEFRAME_1H_KUCOIN = "1hour"
KLINES_1H_LIMIT = 50
KLINES_1H_INTERVAL_SEC = 3600

# --- SMC settings ---
SMC_SWING_LOOKBACK = 5      # candles each side to confirm swing point
SMC_FVG_MIN_PCT = 0.0005    # minimum FVG size (0.05%)
SMC_OB_LOOKBACK = 30        # candles back to search for order blocks
