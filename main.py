"""
Crypto Signal Bot — entry point.

Flow every N minutes:
  1. Fetch top 30 USDT pairs from KuCoin (by 24h volume)
  2. Run SMC technical filter (BOS + FVG + OB + multi-timeframe)
  3. Send only strong setups to Claude Sonnet
  4. Claude returns LONG / SHORT / NO TRADE
  5. Telegram receives only actionable signals
"""

import logging
import os
import time
import threading
from datetime import datetime, timezone

from flask import Flask, request as flask_request
from apscheduler.schedulers.background import BackgroundScheduler

import requests as _requests

from config import (
    SCAN_INTERVAL_MINUTES, SIGNAL_COOLDOWN_HOURS, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    TRADING_HOURS_START, TRADING_HOURS_END, TRADE_WEEKENDS,
)
from src.binance_client import (
    get_top_coins, get_klines, get_klines_1h, get_klines_4h,
    get_btc_change_1h, get_funding_rate,
)
from src.signal_filter import analyze_coin_smc
from src.claude_analyzer import analyze_batch_with_claude
from src.telegram_notifier import send_signal, send_status, send_news_alert, calculate_tp_sl
from src.news_filter import check_news_sentiment
from src.news_agent import get_market_news, detect_major_events, fetch_recent_headlines
from src.db import init_db, get_open_signals, update_signal_status, get_stats

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

    # /stats — статистика побед/поражений
    elif text in ("/stats", "/статистика"):
        try:
            s7  = get_stats(days=7)
            s30 = get_stats(days=30)
            _reply(chat_id,
                   f"📈 *СТАТИСТИКА*\n\n"
                   f"*За 7 дней:*\n"
                   f"  Всего: {s7['total']}\n"
                   f"  TP1: {s7['tp1_hit']}  TP2: {s7['tp2_hit']}  SL: {s7['sl_hit']}\n"
                   f"  Win rate: *{s7['win_rate']}%*\n\n"
                   f"*За 30 дней:*\n"
                   f"  Всего: {s30['total']}\n"
                   f"  TP1: {s30['tp1_hit']}  TP2: {s30['tp2_hit']}  SL: {s30['sl_hit']}\n"
                   f"  Win rate: *{s30['win_rate']}%*")
        except Exception as e:
            _reply(chat_id, f"Ошибка статистики: {e}")

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

# ── News alert deduplication cache ────────────────────────────────────────────
# Prevents re-sending the same major event alert for 6 hours.
# Format: { "event name": timestamp_sent }
_news_alert_cache: dict[str, float] = {}
_NEWS_ALERT_COOLDOWN_HOURS = 6


def _is_alert_duplicate(name: str) -> bool:
    if name in _news_alert_cache:
        age_hours = (time.time() - _news_alert_cache[name]) / 3600
        if age_hours < _NEWS_ALERT_COOLDOWN_HOURS:
            return True
    return False


def _is_duplicate(symbol: str, direction: str) -> bool:
    if symbol in _signal_cache:
        cached_dir, cached_ts = _signal_cache[symbol]
        age_hours = (time.time() - cached_ts) / 3600
        if cached_dir == direction and age_hours < SIGNAL_COOLDOWN_HOURS:
            return True
    return False


def _cache_signal(symbol: str, direction: str):
    _signal_cache[symbol] = (direction, time.time())


# ── Open-signal monitor (updates TP/SL hits in DB) ────────────────────────────
def _check_open_signals():
    """For each OPEN signal in DB, fetch current price and update status."""
    open_signals = get_open_signals()
    if not open_signals:
        return

    for sig in open_signals:
        try:
            df = get_klines(sig["symbol"], limit=2)
            current = df["close"][-1]
            high    = df["high"][-1]
            low     = df["low"][-1]

            direction = sig["direction"]
            tp1, tp2, sl = sig["tp1"], sig["tp2"], sig["sl"]
            opened_at = sig["opened_at"]

            new_status = None
            exit_px    = current

            if direction == "LONG":
                if low <= sl:
                    new_status, exit_px = "SL_HIT", sl
                elif high >= tp2:
                    new_status, exit_px = "TP2_HIT", tp2
                elif high >= tp1:
                    new_status, exit_px = "TP1_HIT", tp1
            else:  # SHORT
                if high >= sl:
                    new_status, exit_px = "SL_HIT", sl
                elif low <= tp2:
                    new_status, exit_px = "TP2_HIT", tp2
                elif low <= tp1:
                    new_status, exit_px = "TP1_HIT", tp1

            # Expire after 24h with no result
            age_hours = (time.time() - opened_at) / 3600
            if new_status is None and age_hours > 24:
                new_status = "EXPIRED"

            if new_status:
                update_signal_status(sig["id"], new_status, exit_px)
                log.info(f"  Signal #{sig['id']} {sig['symbol']} → {new_status}")

        except Exception as e:
            log.warning(f"  Could not check signal #{sig['id']}: {e}")


# ── Main scanning function ────────────────────────────────────────────────────
def run_scan():
    now_utc = datetime.now(timezone.utc)

    # Weekend filter (Mon=0 ... Sun=6)
    if not TRADE_WEEKENDS and now_utc.weekday() >= 5:
        log.info(f"Weekend ({now_utc.strftime('%A')}) — scan skipped")
        return

    # Trading hours filter (UTC)
    utc_hour = now_utc.hour
    if not (TRADING_HOURS_START <= utc_hour < TRADING_HOURS_END):
        log.info(f"Outside trading hours (UTC {utc_hour:02d}:xx) — scan skipped")
        return

    log.info("=== Scan started (SMC mode) ===")

    try:
        # Step 0a: Global macro news (Groq free tier)
        news = get_market_news()
        log.info(
            f"News: {news['sentiment']} — {news['summary']} "
            f"({news['headline_count']} headlines)"
        )
        if news["pause"]:
            log.warning("News agent: PAUSE — extreme market event, skipping scan")
            send_status(f"⚠️ *СТОП* — новостной агент остановил скан:\n_{news['summary']}_")
            return

        # Step 0a-2: Detect and broadcast high-impact macro events
        try:
            headlines = fetch_recent_headlines()
            events = detect_major_events(headlines)
            for ev in events:
                if not _is_alert_duplicate(ev["name"]):
                    if send_news_alert(ev):
                        _news_alert_cache[ev["name"]] = time.time()
                        log.info(f"News alert sent: {ev['name']} ({ev['direction']} {ev['level']}x)")
        except Exception as e:
            log.warning(f"Major event check failed: {e}")

        # Step 0b: BTC 1h change for correlation filter
        btc_change = get_btc_change_1h()
        log.info(f"BTC 1h change: {btc_change:+.2f}%")

        # Check open signals from previous scans → update TP/SL hits
        _check_open_signals()

        # Step 1: top 30 coins by volume
        coins = get_top_coins()
        log.info(f"Fetched {len(coins)} coins from KuCoin")

        setups = []

        # Step 2: SMC filter — BOS + confirmation + 1h/4h trend + BTC correlation
        for symbol in coins:
            try:
                df_15m = get_klines(symbol)
                df_1h  = get_klines_1h(symbol)
                df_4h  = get_klines_4h(symbol)
                setup  = analyze_coin_smc(df_15m, df_1h, symbol, df_4h, btc_change)
                if setup:
                    log.info(
                        f"  SMC setup: {symbol:12s}  {setup['direction']}  "
                        f"4h={setup['trend_4h']} 1h={setup['trend_1h']}  "
                        f"signals={setup['signals']}"
                    )
                    setups.append(setup)
                time.sleep(0.2)  # 2 API calls per coin — small delay
            except Exception as e:
                log.warning(f"  Skip {symbol}: {e}")

        log.info(f"SMC filter: {len(setups)} setups from {len(coins)} coins")

        # Step 3: remove duplicates
        fresh = [s for s in setups if not _is_duplicate(s["symbol"], s["direction"])]
        log.info(f"After dedup: {len(fresh)} fresh setups")

        # Step 3b: news + funding enrichment
        enriched = []
        for s in fresh:
            # News check — block on bad news
            news = check_news_sentiment(s["symbol"])
            if not news["safe"]:
                log.info(f"  Skip {s['symbol']} — {news['reason']}")
                continue
            # Funding rate (best effort)
            s["funding_rate"] = get_funding_rate(s["symbol"])
            enriched.append(s)

        log.info(f"After news/funding: {len(enriched)} setups → sending to Claude")

        if not enriched:
            log.info("=== Scan complete — 0 signal(s) sent ===\n")
            return

        # Step 4: ONE batch call to Claude Haiku (+ news context)
        try:
            analyses = analyze_batch_with_claude(enriched, news_context=news)
        except Exception as e:
            log.error(f"Claude batch call failed: {e}")
            return

        # Step 5: Send signals to Telegram
        sent_count = 0
        for analysis in analyses:
            try:
                # Attach news context to each analysis for Telegram message
                analysis["news_sentiment"] = news.get("sentiment", "")
                analysis["news_summary"]   = news.get("summary", "")

                log.info(
                    f"  Claude: {analysis['symbol']} → {analysis['decision']} "
                    f"({analysis.get('confidence','?')}) — {analysis.get('reason','')}"
                )
                if analysis["decision"] != "NO TRADE":
                    if send_signal(analysis):
                        _cache_signal(analysis["symbol"], analysis["decision"])
                        sent_count += 1
                        log.info(f"  Signal sent: {analysis['symbol']} {analysis['decision']}")
            except Exception as e:
                log.error(f"  Error sending {analysis.get('symbol','?')}: {e}")

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

    # Initialise signal-tracking DB
    try:
        init_db()
        log.info("Database initialised")
    except Exception as e:
        log.warning(f"DB init failed: {e}")

    try:
        send_status(
            "🤖 *Crypto Signal Bot Online*\n"
            f"Сканирую топ-30 монет каждые {SCAN_INTERVAL_MINUTES} мин (Пн-Пт, 10:00–02:00 по Риге)."
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
