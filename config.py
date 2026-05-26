import os
from dotenv import load_dotenv

load_dotenv()

# --- Required secrets (set in Render environment variables) ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# --- Scan settings ---
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "10"))
TOP_COINS_COUNT = 100      # top 100 by volume → more setups per scan
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

# --- 4h candles for higher timeframe bias ---
TIMEFRAME_4H_KUCOIN = "4hour"
KLINES_4H_LIMIT = 30
KLINES_4H_INTERVAL_SEC = 4 * 3600

# --- Trading hours filter (UTC) ---
TRADING_HOURS_START = 7    # 07:00 UTC — London open
TRADING_HOURS_END   = 23   # 23:00 UTC — NY close
TRADE_WEEKENDS      = False  # skip Saturday and Sunday

# --- SMC settings ---
SMC_SWING_LOOKBACK    = 5       # candles each side to confirm swing point
SMC_FVG_MIN_PCT       = 0.0005  # minimum FVG size (0.05%)
SMC_OB_LOOKBACK       = 30      # candles back to search for order blocks
SMC_MIN_CONFIRMATIONS = 1       # 1 confirmation from FVG/OB/Sweep (more signals)
SMC_BOS_MIN_VOLUME    = 1.5     # BOS candle volume must be >= 1.5x average
