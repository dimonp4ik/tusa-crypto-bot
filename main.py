"""
Crypto Signal Bot — entry point.

Flow every N minutes:
  1. Fetch top 50 USDT pairs from Bybit (by 24h volume)
  2. Run technical filter (EMA + RSI + Volume + Breakout)
  3. Send only strong setups (~3-8 coins) to Claude Haiku
  4. Claude returns LONG / SHORT / NO TRADE
  5. Telegram receives only actionable signals
"""

import logging
import os
import time
import threading

from flask import Flask, request as flask_request
from apscheduler.schedulers.background import BackgroundScheduler

import requests as _requests

from config import SCAN_INTERVAL_MINUTES, SIGNAL_COOLDOWN_HOURS, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from src.binance_client import get_top_coins, get_klines
from src.signal_filter import analyze_coin
from src.claude_analyzer import analyze_with_claude
from src.telegram_notifier import send_signal, send_status

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Flask (keeps Render dyno alive) ──────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def health():
    return "Crypto Signal Bot is running.", 200


@app.route("/status")
def status():
    return f"Scanning every {SCAN_INTERVAL_MINUTES} min. Signal cache: {len(_signal_cache)} entries.", 200


# ── Telegram webhook — handles incoming messages ──────────────────────────────
@app.route("/webhook", methods=["POST"])
def webhook():
    data = flask_request.get_json(silent=True)
    if not data:
        return "ok", 200

    message = data.get("message") or data.get("channel_post")
    if not message:
        return "ok", 200

    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "").strip().lower()

    if not chat_id or not text:
        return "ok", 200

    # "привет" — проверка что бот живой
    if "привет" in text:
        _reply(chat_id,
               "👋 Привет! Бот работает.\n"
               f"⏱ Сканирую каждые {SCAN_INTERVAL_MINUTES} мин.\n"
               f"📊 Монет в кэше: {len(_signal_cache)}")

    # /status — подробный статус
    elif text in ("/status", "/старт", "/start"):
        _reply(chat_id,
               f"🤖 *TUSA CRYPTO BOT*\n"
               f"✅ Работает\n"
               f"⏱ Интервал: {SCAN_INTERVAL_MINUTES} мин\n"
               f"📊 Сигналов в кэше: {len(_signal_cache)}\n"
               f"💾 Данные: KuCoin\n"
               f"🧠 AI: Claude Haiku")

    return "ok", 200


def _reply(chat_id: int, text: str):
    """Send a reply to a specific chat."""
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"Reply failed: {e}")


# ── Signal deduplication cache ────────────────────────────────────────────────
# Prevents sending the same signal for the same coin repeatedly.
# Format: { "BTCUSDT": ("LONG", 1714000000.0) }
_signal_cache: dict[str, tuple[str, float]] = {}


def _is_duplicate(symbol: str, direction: str) -> bool:
    if symbol in _signal_cache:
        cached_dir, cached_ts = _signal_cache[symbol]
        age_hours = (time.time() - cached_ts) / 3600
        if cached_dir == direction and age_hours < SIGNAL_COOLDOWN_HOURS:
            return True
    return False


def _cache_signal(symbol: str, direction: str):
    _signal_cache[symbol] = (direction, time.time())


# ── Main scanning function ────────────────────────────────────────────────────
def run_scan():
    log.info("=== Scan started ===")

    try:
        # Step 1: top 50 coins
        coins = get_top_coins()
        log.info(f"Fetched {len(coins)} coins from Binance")

        setups = []

        # Step 2: technical filter
        for symbol in coins:
            try:
                df = get_klines(symbol)
                setup = analyze_coin(df, symbol)
                if setup:
                    log.info(
                        f"  Setup: {symbol:12s}  {setup['direction']}  "
                        f"B={setup['bullish_score']} S={setup['bearish_score']}  "
                        f"signals={setup['signals']}"
                    )
                    setups.append(setup)
                time.sleep(0.12)  # stay well within Bybit rate limits
            except Exception as e:
                log.warning(f"  Skip {symbol}: {e}")

        log.info(f"Pre-filter: {len(setups)} setups passed from {len(coins)} coins")

        # Step 3–4: Claude analysis + Telegram
        sent_count = 0
        for setup in setups:
            try:
                if _is_duplicate(setup["symbol"], setup["direction"]):
                    log.info(f"  Duplicate skip: {setup['symbol']} {setup['direction']}")
                    continue

                analysis = analyze_with_claude(setup)
                log.info(
                    f"  Claude: {setup['symbol']} → {analysis['decision']} "
                    f"({analysis.get('confidence','?')}) — {analysis.get('reason','')}"
                )

                if analysis["decision"] != "NO TRADE":
                    if send_signal(analysis):
                        _cache_signal(setup["symbol"], analysis["decision"])
                        sent_count += 1
                        log.info(f"  Signal sent: {setup['symbol']} {analysis['decision']}")

            except Exception as e:
                log.error(f"  Error processing {setup['symbol']}: {e}")

        log.info(f"=== Scan complete — {sent_count} signal(s) sent ===\n")

    except Exception as e:
        log.error(f"Scan failed: {e}")


# ── Self-ping (keeps Render free tier awake) ──────────────────────────────────
def _self_ping():
    """Ping own health endpoint every 4 minutes so Render never sleeps."""
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not render_url:
        log.info("RENDER_EXTERNAL_URL not set — self-ping disabled (local run)")
        return
    while True:
        time.sleep(240)  # 4 minutes
        try:
            _requests.get(f"{render_url}/", timeout=10)
            log.info("Self-ping OK")
        except Exception as e:
            log.warning(f"Self-ping failed: {e}")


# ── Webhook setup ────────────────────────────────────────────────────────────
def _setup_webhook():
    """Register Telegram webhook so bot can receive messages."""
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not render_url:
        log.info("RENDER_EXTERNAL_URL not set — webhook skipped (local run)")
        return
    webhook_url = f"{render_url}/webhook"
    try:
        resp = _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook",
            json={"url": webhook_url},
            timeout=10,
        )
        log.info(f"Webhook set: {webhook_url} → {resp.json().get('description', '?')}")
    except Exception as e:
        log.warning(f"Webhook setup failed: {e}")


# ── Startup ───────────────────────────────────────────────────────────────────
def start_bot():
    log.info("Starting Crypto Signal Bot...")

    try:
        send_status(
            "🤖 *Crypto Signal Bot Online*\n"
            f"Сканирую топ-50 монет каждые {SCAN_INTERVAL_MINUTES} минут."
        )
    except Exception as e:
        log.warning(f"Could not send startup message: {e}")

    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(run_scan, "interval", minutes=SCAN_INTERVAL_MINUTES)
    scheduler.start()
    log.info(f"Scheduler running — interval {SCAN_INTERVAL_MINUTES} min")

    # Register Telegram webhook
    _setup_webhook()

    # First scan immediately
    threading.Thread(target=run_scan, daemon=True).start()

    # Self-ping to keep Render awake
    threading.Thread(target=_self_ping, daemon=True).start()


start_bot()  # runs at module load — works with gunicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
