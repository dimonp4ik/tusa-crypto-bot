"""
Crypto Signal Bot — entry point.

Flow every N minutes:
  1. Fetch top 45 USDT pairs from KuCoin (by 24h volume)
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
    MAX_SETUPS_TO_CLAUDE, ALLOWED_SYMBOLS, KLINES_INTERVAL_SEC, SIGNAL_EXPIRY_HOURS,
    CLAUDE_HEAVY_MIN_SCORE, CLAUDE_HEAVY_MAX_PER_SCAN, CLAUDE_MEMORY_LIMIT,
)
from src.binance_client import (
    get_top_coins, get_klines, get_klines_1h, get_klines_4h,
    get_btc_change_1h, get_funding_rate,
)
from src.signal_filter import analyze_coin_smc
from src.claude_analyzer import analyze_batch_with_claude, analyze_heavy
from src.telegram_notifier import send_signal, send_status, send_news_alert, send_signal_update, calculate_tp_sl, send_morning_digest
from src.news_filter import check_news_sentiment
from src.news_agent import get_market_news, detect_major_events, fetch_recent_headlines, get_daily_digest
from src.db import (
    init_db, get_open_signals, update_signal_status, get_stats,
    auto_block_bad_symbols, is_symbol_auto_blocked, get_active_symbol_blocks,
    get_recent_outcomes, unblock_symbol, get_symbols_performance,
)
from config import ADMIN_IDS

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


# ── Admin panel helpers ───────────────────────────────────────────────────────

# Persistent bottom-bar keyboards — set once via /start, stay forever in DM.
_USER_KB = {
    "keyboard": [
        [{"text": "📋 Открытые сделки"}, {"text": "📈 Результаты"}],
        [{"text": "❓ Помощь"}],
    ],
    "resize_keyboard": True,
    "is_persistent":   True,
}
_ADMIN_KB = {
    "keyboard": [
        [{"text": "🛠 Админ панель"}],
        [{"text": "📋 Открытые сделки"}, {"text": "📈 Результаты"}],
        [{"text": "❓ Помощь"}],
    ],
    "resize_keyboard": True,
    "is_persistent":   True,
}

# Inline keyboard shown inside the panel message.
_ADMIN_KEYBOARD = {
    "inline_keyboard": [[
        {"text": "📊 Статистика",       "callback_data": "adm_stats"},
        {"text": "📋 Открытые сделки",  "callback_data": "adm_open"},
    ], [
        {"text": "🚫 Авто-блок",        "callback_data": "adm_blocks"},
        {"text": "🏆 Топ монет",        "callback_data": "adm_top"},
        {"text": "💀 Худшие монеты",    "callback_data": "adm_worst"},
    ]]
}


def _send_persistent_menu(chat_id: int, is_admin: bool = False):
    """Send the persistent bottom-bar keyboard. Admins get extra admin button."""
    kb   = _ADMIN_KB if is_admin else _USER_KB
    text = ("✅ Меню активировано.\n🛠 Админ панель доступна."
            if is_admin else
            "✅ Меню активировано. Кнопки внизу всегда доступны.")
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "reply_markup": kb},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"_send_persistent_menu failed: {e}")


def _send_keyboard(chat_id: int, text: str):
    """Send message with admin inline keyboard."""
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id, "text": text,
                "parse_mode": "Markdown",
                "reply_markup": _ADMIN_KEYBOARD,
            },
            timeout=10,
        )
    except Exception as e:
        log.warning(f"_send_keyboard failed: {e}")


def _answer_callback(callback_id: str, text: str = ""):
    """Acknowledge a button press (stops Telegram spinner)."""
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
            json={"callback_query_id": callback_id, "text": text},
            timeout=5,
        )
    except Exception:
        pass


def _edit_message(chat_id: int, message_id: int, text: str):
    """Edit an existing message and re-attach the keyboard."""
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText",
            json={
                "chat_id": chat_id, "message_id": message_id,
                "text": text, "parse_mode": "Markdown",
                "reply_markup": _ADMIN_KEYBOARD,
            },
            timeout=10,
        )
    except Exception as e:
        log.warning(f"_edit_message failed: {e}")


def _handle_admin_callback(callback_id: str, chat_id: int,
                           message_id: int, data: str):
    """Dispatch inline-button presses for the admin panel."""
    _answer_callback(callback_id)

    if data == "adm_stats":
        try:
            s7  = get_stats(days=7)
            s30 = get_stats(days=30)
            txt = (
                f"📈 *СТАТИСТИКА*\n\n"
                f"*За 7 дней:*\n"
                f"  Сигналов: {s7['total']}  Закрыто: {s7['closed']}\n"
                f"  TP1: {s7['tp1_hit']} ({s7['tp1_rate']}%)  TP2: {s7['tp2_hit']}\n"
                f"  BE: {s7['breakeven']}  SL: {s7['sl_hit']}  Expired: {s7['expired']}\n"
                f"  Win rate: *{s7['win_rate']}%*\n\n"
                f"*За 30 дней:*\n"
                f"  Сигналов: {s30['total']}  Закрыто: {s30['closed']}\n"
                f"  TP1: {s30['tp1_hit']} ({s30['tp1_rate']}%)  TP2: {s30['tp2_hit']}\n"
                f"  BE: {s30['breakeven']}  SL: {s30['sl_hit']}  Expired: {s30['expired']}\n"
                f"  Win rate: *{s30['win_rate']}%*"
            )
        except Exception as e:
            txt = f"Ошибка: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_open":
        try:
            sigs = get_open_signals()
            if not sigs:
                txt = "📋 *Открытые сделки*\n\nНет активных позиций."
            else:
                lines = ["📋 *Открытые сделки*\n"]
                for s in sigs:
                    import time as _t
                    age_h = round((_t.time() - s["opened_at"]) / 3600, 1)
                    lines.append(
                        f"• *{s['symbol']}* {s['direction']} "
                        f"@ {s['entry_price']}  [{s['status']}]  {age_h}ч"
                    )
                txt = "\n".join(lines)
        except Exception as e:
            txt = f"Ошибка: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_blocks":
        try:
            blocks = get_active_symbol_blocks()
            if not blocks:
                txt = "🚫 *Авто-блок*\n\nЗаблокированных монет нет."
                _edit_message(chat_id, message_id, txt)
            else:
                import time as _t
                lines = ["🚫 *Авто-блок*\n"]
                keyboard_rows = []
                for b in blocks:
                    until = datetime.fromtimestamp(b["blocked_until"]).strftime("%d.%m %H:%M")
                    lines.append(f"• *{b['symbol']}* до {until}\n  _{b['reason']}_")
                    keyboard_rows.append([{
                        "text": f"✅ Разблокировать {b['symbol']}",
                        "callback_data": f"adm_unblock_{b['symbol']}",
                    }])
                # add back-row with main buttons
                keyboard_rows.append([
                    {"text": "« Назад", "callback_data": "adm_back"}
                ])
                try:
                    _requests.post(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText",
                        json={
                            "chat_id": chat_id, "message_id": message_id,
                            "text": "\n".join(lines), "parse_mode": "Markdown",
                            "reply_markup": {"inline_keyboard": keyboard_rows},
                        },
                        timeout=10,
                    )
                except Exception as e:
                    log.warning(f"blocks keyboard failed: {e}")
        except Exception as e:
            _edit_message(chat_id, message_id, f"Ошибка: {e}")

    elif data.startswith("adm_unblock_"):
        symbol = data[len("adm_unblock_"):]
        try:
            unblock_symbol(symbol)
            txt = f"✅ *{symbol}* разблокирована.\n\nНажми 🚫 Авто-блок чтобы обновить список."
        except Exception as e:
            txt = f"Ошибка разблокировки: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_top":
        try:
            perfs = get_symbols_performance(days=30)
            top = [p for p in perfs if p["total_r"] > 0][:8]
            if not top:
                txt = "🏆 *Топ монет (30д)*\n\nНет прибыльных монет с данными."
            else:
                lines = ["🏆 *Топ монет (30д)*\n"]
                for i, p in enumerate(top, 1):
                    lines.append(
                        f"{i}. *{p['symbol']}*  {p['total_r']:+.2f}R  "
                        f"win {p['win_rate']}%  ({p['trades']} сд)"
                    )
                txt = "\n".join(lines)
        except Exception as e:
            txt = f"Ошибка: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_worst":
        try:
            perfs = get_symbols_performance(days=30)
            worst = [p for p in reversed(perfs) if p["trades"] >= 2][:8]
            if not worst:
                txt = "💀 *Худшие монеты (30д)*\n\nНедостаточно данных."
            else:
                lines = ["💀 *Худшие монеты (30д)*\n"]
                for i, p in enumerate(worst, 1):
                    lines.append(
                        f"{i}. *{p['symbol']}*  {p['total_r']:+.2f}R  "
                        f"win {p['win_rate']}%  ({p['trades']} сд)"
                    )
                txt = "\n".join(lines)
        except Exception as e:
            txt = f"Ошибка: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_back":
        _edit_message(chat_id, message_id, "🛠 *TUSA Admin Panel*\nВыбери раздел:")


# ── Telegram webhook — handles incoming messages ──────────────────────────────
@app.route("/webhook", methods=["POST"])
def webhook():
    data = flask_request.get_json(silent=True)
    if not data:
        return "ok", 200

    # ── Inline button press ───────────────────────────────────────────────────
    cb = data.get("callback_query")
    if cb:
        user_id    = cb.get("from", {}).get("id")
        chat_id    = cb.get("message", {}).get("chat", {}).get("id")
        message_id = cb.get("message", {}).get("message_id")
        cb_data    = cb.get("data", "")
        cb_id      = cb.get("id")
        if user_id in ADMIN_IDS:
            _handle_admin_callback(cb_id, chat_id, message_id, cb_data)
        else:
            _answer_callback(cb_id, "Нет доступа.")
        return "ok", 200

    message = data.get("message") or data.get("channel_post")
    if not message:
        return "ok", 200

    chat_id = message.get("chat", {}).get("id")
    user_id = message.get("from", {}).get("id")
    text = message.get("text", "").strip().lower()

    if not chat_id or not text:
        return "ok", 200

    # "привет" — проверка что бот живой
    if "привет" in text:
        _reply(chat_id,
               "👋 Привет! Бот работает.\n"
               f"⏱ Сканирую каждые {SCAN_INTERVAL_MINUTES} мин.\n"
               f"📊 Монет в кэше: {len(_signal_cache)}")

    # /start → постоянное меню (у админов расширенное)
    elif text == "/start":
        _send_persistent_menu(chat_id, is_admin=(user_id in ADMIN_IDS))

    # 🛠 Кнопка "Админ панель" → инлайн-панель
    elif text == "🛠 админ панель":
        if user_id in ADMIN_IDS:
            _send_keyboard(chat_id, "🛠 *TUSA Admin Panel*\nВыбери раздел:")
        else:
            _reply(chat_id, "Нет доступа.")

    # 📋 Открытые сделки
    elif text == "📋 открытые сделки":
        try:
            sigs = get_open_signals()
            if not sigs:
                _reply(chat_id, "📋 *Открытые сделки*\n\nНет активных позиций.")
            else:
                import time as _t
                lines = ["📋 *Открытые сделки*\n"]
                for s in sigs:
                    age_h = round((_t.time() - s["opened_at"]) / 3600, 1)
                    icon  = "🟡" if s["status"] == "TP1_PARTIAL" else "🟢"
                    lines.append(
                        f"{icon} *{s['symbol']}* {s['direction']} "
                        f"@ {s['entry_price']}  _{age_h}ч_"
                    )
                _reply(chat_id, "\n".join(lines))
        except Exception as e:
            _reply(chat_id, f"Ошибка: {e}")

    # 📈 Результаты
    elif text == "📈 результаты":
        try:
            s7  = get_stats(days=7)
            s30 = get_stats(days=30)
            _reply(chat_id,
                   f"📈 *Результаты бота*\n\n"
                   f"*За 7 дней:*\n"
                   f"  Сигналов: {s7['total']}  •  Win rate: *{s7['win_rate']}%*\n"
                   f"  TP1: {s7['tp1_hit']}  TP2: {s7['tp2_hit']}  SL: {s7['sl_hit']}\n\n"
                   f"*За 30 дней:*\n"
                   f"  Сигналов: {s30['total']}  •  Win rate: *{s30['win_rate']}%*\n"
                   f"  TP1: {s30['tp1_hit']}  TP2: {s30['tp2_hit']}  SL: {s30['sl_hit']}")
        except Exception as e:
            _reply(chat_id, f"Ошибка: {e}")

    # ❓ Помощь
    elif text == "❓ помощь":
        _reply(chat_id,
               "❓ *Как читать сигналы*\n\n"
               "*Направление:*\n"
               "  📈 LONG — ожидаем рост, покупаем\n"
               "  📉 SHORT — ожидаем падение, продаём\n\n"
               "*Уровни:*\n"
               "  🎯 *TP1* — первая цель. Закрываем 50% позиции\n"
               "  🎯 *TP2* — вторая цель. Закрываем остаток\n"
               "  🛑 *SL* — стоп-лосс. Выход если цена пошла против\n\n"
               "*Исходы сделки:*\n"
               "  ✅ TP1 / TP2 — прибыль\n"
               "  ⚖️ BE — безубыток (TP1 взят, остаток закрыт в ноль)\n"
               "  ❌ SL — убыток\n"
               "  ⏱ Expired — время вышло, сделка закрыта без результата\n\n"
               "*Win rate* — % прибыльных от закрытых.\n"
               "Норма для SMC стратегии: 35–45% при высоком R\\:R.")

    # /admin — запасной вариант текстом
    elif text in ("/admin", "/панель"):
        if user_id in ADMIN_IDS:
            _send_keyboard(chat_id, "🛠 *TUSA Admin Panel*\nВыбери раздел:")
        else:
            _reply(chat_id, "Нет доступа.")

    # /status — подробный статус
    elif text in ("/status", "/старт"):
        _reply(chat_id,
               f"🤖 *TUSA CRYPTO BOT*\n"
               f"✅ Работает\n"
               f"⏱ Интервал: {SCAN_INTERVAL_MINUTES} мин\n"
               f"📊 Сигналов в кэше: {len(_signal_cache)}\n"
               f"💾 Данные: KuCoin\n"
               f"🧠 AI: Claude Sonnet")

    # /stats — статистика побед/поражений
    elif text in ("/stats", "/статистика"):
        try:
            s7  = get_stats(days=7)
            s30 = get_stats(days=30)
            blocks = get_active_symbol_blocks()
            blocks_line = ", ".join(b["symbol"] for b in blocks[:6]) if blocks else "нет"
            _reply(chat_id,
                   f"📈 *СТАТИСТИКА*\n\n"
                   f"*За 7 дней:*\n"
                   f"  Сигналов: {s7['total']}  Закрыто: {s7['closed']}\n"
                   f"  TP1: {s7['tp1_hit']} ({s7['tp1_rate']}%)  TP2: {s7['tp2_hit']}\n"
                   f"  BE: {s7['breakeven']}  SL: {s7['sl_hit']}  Expired: {s7['expired']}\n"
                   f"  Win rate: *{s7['win_rate']}%*\n\n"
                   f"*За 30 дней:*\n"
                   f"  Сигналов: {s30['total']}  Закрыто: {s30['closed']}\n"
                   f"  TP1: {s30['tp1_hit']} ({s30['tp1_rate']}%)  TP2: {s30['tp2_hit']}\n"
                   f"  BE: {s30['breakeven']}  SL: {s30['sl_hit']}  Expired: {s30['expired']}\n"
                   f"  Win rate: *{s30['win_rate']}%*\n\n"
                   f"🚫 Авто-блок: {blocks_line}")
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


def _setup_rank(setup: dict) -> tuple:
    """Rank setups before Claude so only the strongest spend LLM tokens."""
    mtf_score    = int(setup.get("mtf_score", 0) or 0)
    confirmations = sum(1 for k in ("fvg", "order_block", "liq_sweep") if setup.get(k))
    volume_score  = float(setup.get("volume_ratio", 0.0))
    zone_bonus    = 1 if setup.get("entry_source") in ("OB", "FVG") else 0
    return (mtf_score, zone_bonus, confirmations, volume_score)


# ── Open-signal monitor (updates TP/SL hits in DB) ────────────────────────────
def _slice_candles_from_open(candles: dict, after_ts: float) -> dict:
    """Return only candles that opened after after_ts to avoid counting pre-entry moves."""
    idxs = [i for i, ts in enumerate(candles.get("time", [])) if float(ts) >= float(after_ts)]
    return {k: [v[i] for i in idxs] for k, v in candles.items()}


def _check_open_signals():
    """For each OPEN signal in DB, fetch current price and update status."""
    active_signals = get_open_signals()
    if not active_signals:
        return

    now = time.time()

    for sig in active_signals:
        try:
            opened_at  = float(sig["opened_at"])
            age_hours  = (now - opened_at) / 3600
            # 15m candles = 4 per hour; fetch enough to cover the signal's age
            candle_lim = max(8, min(220, int(age_hours * 4) + 6))
            df_all     = get_klines(sig["symbol"], limit=candle_lim)

            status    = sig["status"]
            direction = sig["direction"]
            entry     = float(sig["entry_price"])
            tp1, tp2, sl = float(sig["tp1"]), float(sig["tp2"]), float(sig["sl"])

            # Inspect only candles that opened after signal time (OPEN)
            # or after TP1 was recorded (TP1_PARTIAL) to avoid pre-entry moves
            monitor_from = opened_at if status == "OPEN" else float(sig.get("tp1_hit_at") or opened_at)
            df = _slice_candles_from_open(df_all, monitor_from)

            new_status = None
            exit_px    = df_all["close"][-1] if df_all.get("close") else entry

            for i in range(len(df.get("close", []))):
                high  = float(df["high"][i])
                low   = float(df["low"][i])
                close = float(df["close"][i])
                exit_px = close

                if status == "OPEN":
                    if direction == "LONG":
                        if low <= sl:             new_status, exit_px = "SL_HIT",     sl;  break
                        if high >= tp2:           new_status, exit_px = "TP2_HIT",    tp2; break
                        if high >= tp1:           new_status, exit_px = "TP1_PARTIAL", tp1; break
                    else:
                        if high >= sl:            new_status, exit_px = "SL_HIT",     sl;  break
                        if low <= tp2:            new_status, exit_px = "TP2_HIT",    tp2; break
                        if low <= tp1:            new_status, exit_px = "TP1_PARTIAL", tp1; break

                elif status == "TP1_PARTIAL":
                    # Remaining 50% — SL moved to breakeven (entry)
                    if direction == "LONG":
                        if low <= entry:          new_status, exit_px = "BREAKEVEN",  entry; break
                        if high >= tp2:           new_status, exit_px = "TP2_HIT",    tp2;   break
                    else:
                        if high >= entry:         new_status, exit_px = "BREAKEVEN",  entry; break
                        if low <= tp2:            new_status, exit_px = "TP2_HIT",    tp2;   break

            if new_status is None and age_hours > SIGNAL_EXPIRY_HOURS:
                new_status = "TP1_EXPIRED" if status == "TP1_PARTIAL" else "EXPIRED"

            if new_status:
                update_signal_status(sig["id"], new_status, exit_px)
                log.info(f"  Signal #{sig['id']} {sig['symbol']} → {new_status}")
                try:
                    send_signal_update(sig, new_status, exit_px)
                except Exception as _e:
                    log.warning(f"  Update notification failed #{sig['id']}: {_e}")

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

        # Auto-block symbols with consistently bad stats (local DB, no API calls)
        new_blocks = auto_block_bad_symbols()
        for b in new_blocks:
            log.info(f"  Auto-blocked: {b['reason']}")

        # Step 1: top 45 liquid coins (quality filtered)
        coins = get_top_coins()
        before_blocks = len(coins)
        coins = [s for s in coins if not is_symbol_auto_blocked(s)]
        if len(coins) != before_blocks:
            log.info(f"Auto-block: skipped {before_blocks - len(coins)} blocked symbol(s)")
        mode = "whitelist" if ALLOWED_SYMBOLS else "auto top-volume"
        log.info(f"Fetched {len(coins)} coins from KuCoin ({mode})")

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

        # Sort by quality score, keep only top MAX_SETUPS_TO_CLAUDE (saves tokens)
        enriched.sort(key=_setup_rank, reverse=True)
        if len(enriched) > MAX_SETUPS_TO_CLAUDE:
            log.info(f"Token saver: top {MAX_SETUPS_TO_CLAUDE} of {len(enriched)} → Claude")
            enriched = enriched[:MAX_SETUPS_TO_CLAUDE]

        log.info(f"After news/funding/ranking: {len(enriched)} setups → sending to Claude")

        if not enriched:
            log.info("=== Scan complete — 0 signal(s) sent ===\n")
            return

        # Step 4: LIGHT tier — ONE batch call to Claude Haiku (cached rules + news)
        try:
            analyses = analyze_batch_with_claude(enriched, news_context=news)
        except Exception as e:
            log.error(f"Claude LIGHT batch call failed: {e}")
            return

        # Step 4b: HEAVY tier — Sonnet second opinion on the strongest survivors.
        # Only setups the LIGHT gate approved (LONG/SHORT, not LOW) with a high
        # mtf_score qualify; capped per scan to protect the budget. Coin memory
        # (recent outcomes) is injected so Sonnet learns from this symbol's past.
        heavy_done = 0
        for analysis in analyses:
            if heavy_done >= CLAUDE_HEAVY_MAX_PER_SCAN:
                break
            decision = analysis.get("decision", "NO TRADE")
            conf     = analysis.get("confidence", "LOW").upper()
            score    = int(analysis.get("mtf_score", 0) or 0)
            if decision in ("LONG", "SHORT") and conf != "LOW" and score >= CLAUDE_HEAVY_MIN_SCORE:
                try:
                    history = get_recent_outcomes(analysis["symbol"], limit=CLAUDE_MEMORY_LIMIT)
                    heavy = analyze_heavy(analysis, news_context=news, history=history)
                    for k in ("decision", "confidence", "risk_score", "trend_strength", "reason", "counter"):
                        if k in heavy:
                            analysis[k] = heavy[k]
                    heavy_done += 1
                    log.info(
                        f"  HEAVY: {analysis['symbol']} → {analysis['decision']} "
                        f"({analysis.get('confidence','?')}) risk={analysis.get('risk_score','?')} "
                        f"— {analysis.get('reason','')}"
                    )
                except Exception as e:
                    log.warning(f"  HEAVY check failed {analysis.get('symbol','?')}: {e}")

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
                decision   = analysis.get("decision", "NO TRADE")
                direction  = analysis.get("direction")
                confidence = analysis.get("confidence", "LOW").upper()

                # Guard: Claude must confirm setup direction, not flip it
                if decision in ("LONG", "SHORT") and decision != direction:
                    log.warning(f"  Skip {analysis['symbol']} — Claude flipped side blocked")
                    continue

                # Skip LOW confidence signals
                if confidence == "LOW":
                    log.info(f"  Skip {analysis['symbol']} — LOW confidence")
                    continue

                if decision != "NO TRADE":
                    if send_signal(analysis):
                        _cache_signal(analysis["symbol"], direction)
                        sent_count += 1
                        log.info(f"  Signal sent: {analysis['symbol']} {direction}")
            except Exception as e:
                log.error(f"  Error sending {analysis.get('symbol','?')}: {e}")

        log.info(f"=== Scan complete — {sent_count} signal(s) sent ===\n")

    except Exception as e:
        log.error(f"Scan failed: {e}")


# ── Morning digest ────────────────────────────────────────────────────────────
def run_morning_digest():
    """Fetch last 18h headlines, ask Groq to rank top 5, send to Telegram."""
    log.info("=== Morning digest started ===")
    try:
        digest = get_daily_digest()
        send_morning_digest(digest)
        log.info(
            f"Morning digest sent — {len(digest.get('items', []))} items, "
            f"overall={digest.get('overall')}"
        )
    except Exception as e:
        log.error(f"Morning digest failed: {e}")


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

    # Dedup guard: only send once per 60s per container (prevents
    # double-message during Render zero-downtime deploys where old + new
    # instances briefly overlap).
    _flag = "/tmp/tusa_started"
    try:
        skip = False
        if os.path.exists(_flag):
            if time.time() - os.path.getmtime(_flag) < 60:
                skip = True
        if not skip:
            open(_flag, "w").close()
            send_status(
                "🤖 *Crypto Signal Bot Online*\n"
                f"Сканирую топ-45 монет каждые {SCAN_INTERVAL_MINUTES} мин "
                f"(Пн-Пт, 10:00–02:00 по Риге)."
            )
    except Exception as e:
        log.warning(f"Could not send startup message: {e}")

    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(run_scan, "interval", minutes=SCAN_INTERVAL_MINUTES)
    scheduler.add_job(
        run_morning_digest, "cron",
        day_of_week="mon-fri", hour=10, minute=0,
        timezone="Europe/Riga",
    )
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
