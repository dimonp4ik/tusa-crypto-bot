import os
from dotenv import load_dotenv

load_dotenv()

# --- Required secrets (set in Render environment variables) ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# --- Scan settings ---
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))
TOP_COINS_COUNT = 30       # top 30 by volume — most volatile, best quality
TIMEFRAME = "5m"           # 5m candle → signals resolve in 30-90 min
KLINES_LIMIT = 200         # 200 × 5m = ~16 hours of data for SMC

# --- Technical filter thresholds ---
RSI_OVERSOLD = 35          # below = potential LONG
RSI_OVERBOUGHT = 65        # above = potential SHORT
VOLUME_SPIKE_MULTIPLIER = 1.8   # current volume must be 1.8x the 20-period average
MIN_SIGNALS_TO_PASS = 2    # coin needs at least 2 bullish or 2 bearish signals

# --- Signal deduplication ---
SIGNAL_COOLDOWN_HOURS = 1  # 5m signals resolve fast — 1h cooldown per coin/direction

# --- KuCoin (accessible from cloud/US servers) ---
KUCOIN_BASE_URL = "https://api.kucoin.com"
QUOTE_ASSET = "USDT"
TIMEFRAME_KUCOIN = "5min"       # KuCoin interval format
KLINES_INTERVAL_SEC = 5 * 60    # 5 minutes in seconds

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
SMC_MIN_CONFIRMATIONS = 2       # 2 confirmations required (FVG/OB/Sweep/Div/Wick/Stoch)
SMC_BOS_MIN_VOLUME    = 1.5     # BOS candle volume must be >= 1.5x average

# --- ATR-based stops/takes ---
ATR_PERIOD       = 14
ATR_SL_MULT      = 1.0   # SL = ATR * 1.0  (tight — 5m moves fast)
ATR_TP1_MULT     = 1.0   # TP1 = ATR * 1.0 (close 50%, move SL to BE)
ATR_TP2_MULT     = 2.0   # TP2 = ATR * 2.0 (close rest — 1:2 R:R)

# --- BTC correlation filter ---
BTC_BLOCK_THRESHOLD_PCT = 1.0  # if BTC moved >1% against direction → skip signal

# --- News filter (CryptoPanic per-coin) ---
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")  # optional
NEWS_BLOCK_KEYWORDS = ["hack", "exploit", "scam", "lawsuit", "sec ", "ban", "delist", "rug"]

# --- Global macro news agent (Groq free tier) ---
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
NEWS_LOOKBACK_HOURS = 2  # fetch headlines from last 2 hours

# --- Database ---
DB_PATH = "signals.db"

# --- Backtest ---
BACKTEST_CANDLES = 2000   # ~7 days of 5m data
BACKTEST_TP_WINDOW = 24   # 24 × 5m = 2 hours — if not hit in 2h, expired
