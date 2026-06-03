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
from src.news_agent import (
    get_market_news, detect_major_events, fetch_recent_headlines,
    get_daily_digest, get_upcoming_high_impact_events, get_day_events,
)
from config import EVENT_WARN_HOURS
from src.db import (
    init_db, get_open_signals, update_signal_status, get_stats,
    auto_block_bad_symbols, is_symbol_auto_blocked, get_active_symbol_blocks,
    get_recent_outcomes, unblock_symbol, get_symbols_performance,
    upsert_user, get_user_by_id, get_all_users,
    add_dynamic_admin, remove_dynamic_admin, get_dynamic_admins, is_dynamic_admin,
    delete_signal, get_recent_signals,
    get_claude_spend_stats,
)
from config import ADMIN_IDS

# ── Admin helpers ─────────────────────────────────────────────────────────────

def _is_admin(user_id: int) -> bool:
    """True for config super-admins OR dynamically added DB admins."""
    return user_id in ADMIN_IDS or is_dynamic_admin(user_id)


def _is_super_admin(user_id: int) -> bool:
    """True only for hardcoded config admins (can manage other admins)."""
    return user_id in ADMIN_IDS


# State: super-admin is typing a new admin's Telegram ID.
# { chat_id: True } — cleared on next message regardless of content.
_pending_add_admin: dict = {}

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
        [{"text": "📰 Новости на сегодня"}],
        [{"text": "❓ Помощь"}],
    ],
    "resize_keyboard": True,
    "is_persistent":   True,
}
_ADMIN_KB = {
    "keyboard": [
        [{"text": "🛠 Админ панель"}],
        [{"text": "📋 Открытые сделки"}, {"text": "📈 Результаты"}],
        [{"text": "📰 Новости на сегодня"}],
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
    ], [
        {"text": "👥 Пользователи",     "callback_data": "adm_users"},
        {"text": "👮 Админы",           "callback_data": "adm_admins"},
    ], [
        {"text": "🗑 Управление сделками", "callback_data": "adm_deals"},
        {"text": "💰 Бюджет Claude",       "callback_data": "adm_budget"},
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
                           message_id: int, data: str, user_id: int = 0):
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
                    until = datetime.fromtimestamp(b["blocked_until"], tz=_riga_tz()).strftime("%d.%m %H:%M")
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

    elif data == "adm_users":
        try:
            users = get_all_users(limit=50)
            if not users:
                txt = "👥 *Пользователи*\n\nНикто ещё не писал боту."
            else:
                lines = [f"👥 *Пользователи* — {len(users)} чел.\n"]
                for u in users:
                    parts = []
                    fn = u.get("first_name") or ""
                    ln = u.get("last_name") or ""
                    name = (fn + (" " + ln if ln else "")).strip() or "—"
                    uname = f"@{u['username']}" if u.get("username") else f"`{u['user_id']}`"
                    last = datetime.fromtimestamp(u["last_seen"], tz=_riga_tz()).strftime("%d.%m %H:%M")
                    lines.append(f"• {name} {uname} — {last} ({u.get('message_count', 1)} сообщ.)")
                txt = "\n".join(lines)
        except Exception as e:
            txt = f"Ошибка: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_admins":
        try:
            dynamic = get_dynamic_admins()
            lines = ["👮 *Управление админами*\n"]
            lines.append("🔒 *Супер-админы (config):*")
            for aid in sorted(ADMIN_IDS):
                lines.append(f"  `{aid}`")
            if dynamic:
                lines.append("\n➕ *Добавленные:*")
                for a in dynamic:
                    fn   = a.get("first_name") or ""
                    un   = f" @{a['username']}" if a.get("username") else ""
                    lines.append(f"  • {fn}{un} `{a['user_id']}`")
            else:
                lines.append("\n_Добавленных админов нет._")

            kb_rows = []
            if _is_super_admin(user_id):
                for a in dynamic:
                    label = a.get("first_name") or str(a["user_id"])
                    kb_rows.append([{
                        "text": f"❌ Удалить {label}",
                        "callback_data": f"adm_rm_admin_{a['user_id']}",
                    }])
                kb_rows.append([{
                    "text": "➕ Добавить администратора",
                    "callback_data": "adm_add_admin",
                }])
            kb_rows.append([{"text": "« Назад", "callback_data": "adm_back"}])
            try:
                _requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText",
                    json={
                        "chat_id": chat_id, "message_id": message_id,
                        "text": "\n".join(lines), "parse_mode": "Markdown",
                        "reply_markup": {"inline_keyboard": kb_rows},
                    },
                    timeout=10,
                )
            except Exception as e:
                log.warning(f"adm_admins keyboard failed: {e}")
        except Exception as e:
            _edit_message(chat_id, message_id, f"Ошибка: {e}")

    elif data.startswith("adm_rm_admin_"):
        if not _is_super_admin(user_id):
            _edit_message(chat_id, message_id, "⛔ Только супер-администратор может удалять.")
            return
        try:
            rm_id = int(data[len("adm_rm_admin_"):])
            remove_dynamic_admin(rm_id)
            txt = f"✅ Администратор `{rm_id}` удалён.\n\nНажми 👮 Админы чтобы обновить список."
        except Exception as e:
            txt = f"Ошибка удаления: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_add_admin":
        if not _is_super_admin(user_id):
            _edit_message(chat_id, message_id, "⛔ Только супер-администратор может добавлять.")
            return
        _pending_add_admin[chat_id] = True
        try:
            _requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": "Отправь *Telegram ID* нового администратора.\nПример: `123456789`",
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception as e:
            log.warning(f"add_admin prompt failed: {e}")

    elif data == "adm_deals":
        try:
            sigs = get_recent_signals(limit=20)
            if not sigs:
                txt = "🗑 *Управление сделками*\n\nСделок в базе нет."
                _edit_message(chat_id, message_id, txt)
            else:
                _RIGA = _riga_tz()
                STATUS_ICON = {
                    "OPEN": "🟢", "TP1_PARTIAL": "🟡", "TP2_HIT": "✅",
                    "BREAKEVEN": "⚖️", "SL_HIT": "❌", "EXPIRED": "⏱",
                    "TP1_EXPIRED": "⏱", "TP1_HIT": "✅",
                }
                lines = [f"🗑 *Управление сделками* (последние {len(sigs)})\n"]
                kb_rows = []
                import time as _t
                for s in sigs:
                    icon   = STATUS_ICON.get(s["status"], "•")
                    opened = datetime.fromtimestamp(s["opened_at"], tz=_RIGA).strftime("%d.%m %H:%M")
                    lines.append(
                        f"{icon} `#{s['id']}` *{s['symbol']}* {s['direction']} "
                        f"@ {s['entry_price']}  [{s['status']}]  {opened}"
                    )
                    kb_rows.append([{
                        "text": f"🗑 Удалить #{s['id']} {s['symbol']} {s['direction']}",
                        "callback_data": f"adm_del_sig_{s['id']}",
                    }])
                kb_rows.append([{"text": "« Назад", "callback_data": "adm_back"}])
                try:
                    _requests.post(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText",
                        json={
                            "chat_id": chat_id, "message_id": message_id,
                            "text": "\n".join(lines), "parse_mode": "Markdown",
                            "reply_markup": {"inline_keyboard": kb_rows},
                        },
                        timeout=10,
                    )
                except Exception as e:
                    log.warning(f"adm_deals keyboard failed: {e}")
        except Exception as e:
            _edit_message(chat_id, message_id, f"Ошибка: {e}")

    elif data.startswith("adm_del_sig_"):
        try:
            sig_id = int(data[len("adm_del_sig_"):])
            removed = delete_signal(sig_id)
            if removed:
                txt = f"✅ Сделка `#{sig_id}` удалена из базы."
            else:
                txt = f"⚠️ Сделка `#{sig_id}` не найдена (уже удалена?)."
        except Exception as e:
            txt = f"Ошибка удаления: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_budget":
        try:
            from config import CLAUDE_DAILY_BUDGET_USD
            s = get_claude_spend_stats()
            remaining = max(0.0, round(CLAUDE_DAILY_BUDGET_USD - s["today_usd"], 4))
            bar_filled = int((s["today_usd"] / CLAUDE_DAILY_BUDGET_USD) * 10) if CLAUDE_DAILY_BUDGET_USD else 0
            bar_filled = min(bar_filled, 10)
            bar = "█" * bar_filled + "░" * (10 - bar_filled)
            txt = (
                f"💰 *Бюджет Claude*\n\n"
                f"Лимит: ${CLAUDE_DAILY_BUDGET_USD:.2f}/день\n"
                f"[{bar}] ${s['today_usd']:.4f}\n"
                f"Осталось сегодня: *${remaining:.4f}*\n\n"
                f"*За сегодня:* {s['today_calls']} вызовов · ${s['today_usd']:.4f}\n"
                f"*За 7 дней:* {s['week_calls']} вызовов · ${s['week_usd']:.4f}\n"
                f"*Всего:* {s['total_calls']} вызовов · ${s['total_usd']:.4f}"
            )
        except Exception as e:
            txt = f"Ошибка: {e}"
        _edit_message(chat_id, message_id, txt)

    elif data == "adm_back":
        _edit_message(chat_id, message_id, "🛠 *TUSA Admin Panel*\nВыбери раздел:")


def _num(s):
    """Parse FF numeric string ('0.3%', '187K', '<0.1') → float or None."""
    if not s:
        return None
    t = s.strip().replace(",", "").replace("%", "").replace("<", "").replace(">", "")
    mult = 1.0
    if t and t[-1] in ("K", "k"):
        mult, t = 1e3, t[:-1]
    elif t and t[-1] in ("M", "m"):
        mult, t = 1e6, t[:-1]
    elif t and t[-1] in ("B", "b"):
        mult, t = 1e9, t[:-1]
    try:
        return float(t) * mult
    except Exception:
        return None


# (keyword substring, RU title, short RU explanation) — first match wins,
# so put more specific phrases before generic ones.
_RU_EVENTS = [
    ("ism manufacturing prices", "Цены в промышленности (ISM)",
     "ценовое давление у производителей — сигнал по инфляции"),
    ("ism manufacturing", "Деловая активность в промышленности (ISM)",
     "настроения производителей; выше 50 = рост экономики"),
    ("ism services", "Деловая активность в услугах (ISM)",
     "настроения в секторе услуг; выше 50 = рост"),
    ("non-farm", "Занятость вне сельского хозяйства (NFP)",
     "ключевой отчёт по рынку труда США"),
    ("nonfarm", "Занятость вне сельского хозяйства (NFP)",
     "ключевой отчёт по рынку труда США"),
    ("adp", "Занятость в частном секторе (ADP)",
     "предвестник NFP по найму в частном секторе"),
    ("unemployment rate", "Уровень безработицы", "доля безработных"),
    ("unemployment claims", "Заявки на пособие по безработице",
     "число новых заявок за неделю"),
    ("jobless claims", "Заявки на пособие по безработице",
     "число новых заявок за неделю"),
    ("core cpi", "Базовая инфляция (Core CPI)",
     "рост цен без еды и энергии"),
    ("cpi", "Инфляция (CPI)", "рост потребительских цен"),
    ("core ppi", "Базовые цены производителей (Core PPI)", "оптовая инфляция"),
    ("ppi", "Цены производителей (PPI)", "оптовая инфляция"),
    ("core pce", "Базовый PCE", "любимый показатель инфляции ФРС"),
    ("pce", "Расходы на личное потребление (PCE)", "инфляция и траты"),
    ("retail sales", "Розничные продажи", "потребительский спрос"),
    ("gdp", "ВВП", "темп роста экономики"),
    ("federal funds rate", "Решение ФРС по ставке",
     "главное событие — ставка ФРС"),
    ("interest rate decision", "Решение по процентной ставке",
     "уровень ключевой ставки"),
    ("fomc statement", "Заявление ФРС", "сопроводительный текст к ставке"),
    ("fomc meeting minutes", "Протокол заседания ФРС",
     "детали обсуждения ставки"),
    ("fomc economic projections", "Экономические прогнозы ФРС",
     "ожидания ФРС по ставке и инфляции"),
    ("powell", "Выступление главы ФРС Пауэлла",
     "намёки на курс по ставке"),
    ("bailey", "Выступление главы Банка Англии",
     "намёки на курс по ставке"),
    ("lagarde", "Выступление главы ЕЦБ", "намёки на курс по ставке"),
    ("fomc member", "Выступление члена ФРС", "намёки на курс по ставке"),
    ("fed chair", "Выступление главы ФРС", "намёки на курс по ставке"),
    ("member", "Выступление представителя ЦБ", "намёки на курс по ставке"),
    ("speaks", "Выступление представителя ЦБ", "намёки на курс по ставке"),
    ("consumer confidence", "Индекс доверия потребителей",
     "настроения покупателей"),
    ("consumer sentiment", "Индекс настроений потребителей",
     "настроения покупателей"),
    ("durable goods", "Заказы на товары длительного пользования",
     "спрос на дорогие товары"),
    ("trade balance", "Торговый баланс", "экспорт минус импорт"),
    ("building permits", "Разрешения на строительство",
     "активность в недвижимости"),
    ("crude oil inventories", "Запасы нефти", "влияет на цену нефти"),
    ("manufacturing pmi", "PMI в промышленности",
     "деловая активность; выше 50 = рост"),
    ("services pmi", "PMI в услугах", "деловая активность; выше 50 = рост"),
    ("flash manufacturing pmi", "Предв. PMI в промышленности",
     "ранняя оценка деловой активности"),
    ("flash services pmi", "Предв. PMI в услугах",
     "ранняя оценка деловой активности"),
    ("pmi", "Индекс деловой активности (PMI)",
     "выше 50 = рост, ниже = спад"),
    ("bank holiday", "Банковский выходной", "биржи/банки закрыты"),
]


def _ru_event(title: str):
    """Map an English FF title → (ru_title, ru_note). Falls back to original."""
    low = title.lower()
    for kw, ru, note in _RU_EVENTS:
        if kw in low:
            return ru, note
    return title, ""


# Market impact one-liners shown after actual result (better / worse than forecast).
# Tuple: (note_if_better, note_if_worse)
_MARKET_NOTES: dict = {
    "core cpi":        ("Базовая инфляция выше → ФРС не снижает ставку → крипта ↓",
                        "Базовая инфляция ниже → ФРС ближе к снижению → крипта ↑"),
    "cpi":             ("Инфляция выше ожиданий → ФРС жёстче → крипта/акции ↓",
                        "Инфляция ниже ожиданий → путь к снижению ставки → крипта ↑"),
    "core ppi":        ("Оптовая инфляция выше → давление сохраняется → осторожно",
                        "Оптовая инфляция ниже → хороший сигнал для рынков"),
    "ppi":             ("Цены производителей выше → инфляционное давление → крипта ↓",
                        "Цены производителей ниже → меньше инфляции → позитив"),
    "core pce":        ("PCE выше → ФРС не спешит со снижением → риски для крипты",
                        "PCE ниже → снижение ставки ближе → позитив для крипты"),
    "pce":             ("Расходы выше ожиданий → инфляционное давление",
                        "Расходы ниже → потребитель экономит → осторожно"),
    "non-farm":        ("Рынок труда силён → доллар ↑, крипта под давлением",
                        "Занятость слабее → доллар ↓, позитив для крипты"),
    "nonfarm":         ("Рынок труда силён → доллар ↑, крипта под давлением",
                        "Занятость слабее → доллар ↓, позитив для крипты"),
    "adp":             ("Частный найм активен → рынок труда силён → доллар ↑",
                        "Частный найм слабее → сигнал охлаждения экономики"),
    "unemployment rate": ("Безработица выше → экономика охлаждается",
                          "Безработица ниже → сильный рынок труда → доллар ↑"),
    "unemployment":    ("Заявки выросли → рынок труда слабеет",
                        "Заявки упали → рынок труда устойчив"),
    "jobless":         ("Заявки выросли → рынок труда слабеет",
                        "Заявки упали → рынок труда устойчив"),
    "gdp":             ("ВВП выше ожиданий → экономика сильнее → доллар ↑",
                        "ВВП ниже ожиданий → риски замедления → осторожно"),
    "retail sales":    ("Потребитель тратит больше → рост экономики",
                        "Розничные продажи слабее → потребитель экономит"),
    "federal funds":   ("Ставка выше ожиданий → доллар ↑, крипта ↓",
                        "Ставка ниже ожиданий → доллар ↓, крипта ↑"),
    "interest rate":   ("Ставка выше ожиданий → доллар ↑, крипта ↓",
                        "Ставка ниже ожиданий → доллар ↓, крипта ↑"),
    "ism manufacturing": ("Промышленность растёт → позитив для экономики",
                          "Промышленность сокращается → риски рецессии"),
    "ism services":    ("Сектор услуг растёт → позитив",
                        "Сектор услуг замедляется → осторожно"),
    "manufacturing pmi": ("Промышленность выше 50 → рост → позитив",
                          "Промышленность ниже 50 → сжатие → осторожно"),
    "services pmi":    ("Услуги выше 50 → рост экономики",
                        "Услуги ниже 50 → замедление"),
    "consumer confidence": ("Потребители оптимистичны → рост трат → позитив",
                            "Потребители пессимистичны → спад трат → осторожно"),
    "durable goods":   ("Спрос на товары высок → экономика активна",
                        "Спрос на товары слаб → инвестиции снижаются"),
}


def _market_note(event_title: str, is_better: bool) -> str:
    """Brief market impact note for a past event. Empty string if unknown."""
    low = event_title.lower()
    for kw, (note_b, note_w) in _MARKET_NOTES.items():
        if kw in low:
            return note_b if is_better else note_w
    return ""


# Crypto digest cache — get_daily_digest() hits Groq + RSS, cache 30 min so
# repeated button presses don't spam the API.
_digest_cache = {"at": 0.0, "items": []}


def _cached_digest() -> list:
    import time as _t
    now = _t.time()
    if now - _digest_cache["at"] > 1800:
        try:
            _digest_cache["items"] = get_daily_digest().get("items", [])
        except Exception:
            pass
        _digest_cache["at"] = now
    return _digest_cache["items"]


def _riga_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Europe/Riga")
    except Exception:
        from datetime import timezone as _tz, timedelta as _td
        return _tz(_td(hours=3))


def _format_day_news() -> str:
    """'📰 Новости на сегодня' — economic calendar + crypto headlines, ≤10 total."""
    RIGA = _riga_tz()

    data   = get_day_events(max_events=10)
    events = data["events"]
    d      = data["date"]
    crypto = _cached_digest()                      # ≤5 AI-picked crypto items

    # Budget: ≤10 total. Reserve up to 4 slots for crypto, grow if calendar small.
    n_crypto = min(len(crypto), 4)
    n_macro  = min(len(events), 10 - n_crypto)
    n_crypto = min(len(crypto), 10 - n_macro)
    events, crypto = events[:n_macro], crypto[:n_crypto]

    header = f"📰 *Новости на сегодня* ({d.strftime('%d.%m')})"
    if data["weekend_rolled"]:
        header += "\n_Выходной — календарь на понедельник._"
    lines = [header]

    # ── Economic calendar ──
    if events:
        lines.append("\n🗓 *Экономический календарь*")
        for e in events:
            flag = "🔴" if e["impact"] == "high" else "🟡"
            when = ("весь день" if e["all_day"] or not e["when_utc"]
                    else e["when_utc"].astimezone(RIGA).strftime("%H:%M по Риге"))
            cc       = f"{e['country']} " if e["country"] else ""
            ru, note = _ru_event(e["title"])
            lines.append(f"{flag} *{cc}{ru}* — {when}")
            if note:
                lines.append(f"   📖 {note}")
            f_, p_, a_ = e["forecast"], e["previous"], e["actual"]
            extra_prev = f" / пред {p_}" if p_ else ""
            if e["passed"] and a_:
                # Event passed AND actual value published
                af, ff = _num(a_), _num(f_)
                if af is not None and ff is not None:
                    is_better = af > ff
                    tag = ("📈 лучше прогноза" if is_better else
                           "📉 хуже прогноза"  if af < ff else "➡️ по прогнозу")
                else:
                    is_better = None
                    tag = "✅ вышло"
                lines.append(f"   факт *{a_}* / прогноз {f_ or '—'}{extra_prev} → {tag}")
                if is_better is not None:
                    impact = _market_note(e["title"], is_better)
                    if impact:
                        lines.append(f"   💡 _{impact}_")
            elif e["passed"]:
                # Event passed but actual not published yet
                lines.append(f"   ✅ прошло · прогноз {f_ or '—'}{extra_prev} · _факт не опубликован_")
            else:
                # Upcoming event
                lines.append(f"   🔮 прогноз {f_ or '—'}{extra_prev}")

    # ── Crypto headlines ──
    if crypto:
        dir_emoji = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}
        lines.append("\n🪙 *Крипто-новости*")
        for it in crypto:
            em = dir_emoji.get(it.get("direction", "NEUTRAL"), "➡️")
            lines.append(f"{em} *{it.get('title', '')}*")
            expl = it.get("explanation", "")
            if expl:
                lines.append(f"   {expl}")

    if not events and not crypto:
        return header + "\n\nВажных событий и новостей нет. Спокойно 🌤"

    lines.append("\n🔴 высокая важность  🟡 средняя")
    return "\n".join(lines)


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
        if _is_admin(user_id):
            _handle_admin_callback(cb_id, chat_id, message_id, cb_data, user_id)
        else:
            _answer_callback(cb_id, "Нет доступа.")
        return "ok", 200

    message = data.get("message") or data.get("channel_post")
    if not message:
        return "ok", 200

    chat_id  = message.get("chat", {}).get("id")
    user_id  = message.get("from", {}).get("id")
    from_obj = message.get("from", {})
    text_raw = message.get("text", "").strip()
    text     = text_raw.lower()

    if not chat_id:
        return "ok", 200

    # Track every user who interacts with the bot
    if user_id:
        try:
            upsert_user(
                user_id,
                username=from_obj.get("username"),
                first_name=from_obj.get("first_name"),
                last_name=from_obj.get("last_name"),
            )
        except Exception as _ue:
            log.warning(f"upsert_user failed: {_ue}")

    if not text:
        return "ok", 200

    # DM = positive chat_id (private chat with bot)
    is_dm = isinstance(chat_id, int) and chat_id > 0

    # ── Pending "add admin" state — super-admin just typed a new admin's ID ──
    if is_dm and _is_super_admin(user_id) and _pending_add_admin.pop(chat_id, False):
        raw_id = text_raw.strip()
        if raw_id.lstrip("-").isdigit():
            new_id = int(raw_id)
            # Try to look up name from users table (may already have interacted)
            u_info = get_user_by_id(new_id) or {}
            add_dynamic_admin(
                new_id,
                username=u_info.get("username"),
                first_name=u_info.get("first_name"),
                added_by=user_id,
            )
            _reply(chat_id,
                   f"✅ Администратор `{new_id}` добавлен.\n"
                   f"Ему нужно написать /start чтобы получить панель.")
        else:
            _reply(chat_id, "❌ Не похоже на Telegram ID. Нужно число, например `123456789`.")
        return "ok", 200

    # /start → постоянное меню (у админов расширенное, только в ЛС)
    if text == "/start":
        _send_persistent_menu(chat_id, is_admin=(is_dm and _is_admin(user_id)))

    # 🛠 Кнопка "Админ панель" → инлайн-панель (только ЛС)
    elif text == "🛠 админ панель":
        if is_dm and _is_admin(user_id):
            _send_keyboard(chat_id, "🛠 *TUSA Admin Panel*\nВыбери раздел:")
        elif not is_dm:
            pass  # silence in group chats
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

    # 📰 Новости на сегодня
    elif text == "📰 новости на сегодня":
        try:
            _reply(chat_id, _format_day_news())
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

    # /admin — запасной вариант текстом (только ЛС)
    elif text in ("/admin", "/панель"):
        if is_dm and _is_admin(user_id):
            _send_keyboard(chat_id, "🛠 *TUSA Admin Panel*\nВыбери раздел:")
        elif not is_dm:
            pass
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

    # TP/SL monitoring moved to dedicated 1-min job (_monitor_open_signals)

    # Weekend filter (Mon=0 ... Sun=6)
    if not TRADE_WEEKENDS and now_utc.weekday() >= 5:
        log.info(f"Weekend ({now_utc.strftime('%A')}) — new-signal scan skipped")
        return

    # Trading hours filter (UTC)
    utc_hour = now_utc.hour
    if not (TRADING_HOURS_START <= utc_hour < TRADING_HOURS_END):
        log.info(f"Outside trading hours (UTC {utc_hour:02d}:xx) — new-signal scan skipped")
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
        log.info(f"Fetched {len(coins)} coins ({mode})")

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
            # Funding rate — fetch + hard filter crowded positions
            fr = get_funding_rate(s["symbol"])
            s["funding_rate"] = fr
            if fr is not None:
                if s["direction"] == "LONG"  and fr >  0.0005:   # >+0.05% = crowded longs
                    log.info(f"  Skip {s['symbol']} LONG — funding {fr*100:+.3f}% crowded")
                    continue
                if s["direction"] == "SHORT" and fr < -0.0005:   # <-0.05% = crowded shorts
                    log.info(f"  Skip {s['symbol']} SHORT — funding {fr*100:+.3f}% crowded")
                    continue
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

        # Upcoming high-impact macro events (CPI/FOMC/NFP) — warn on signals
        event_warning = ""
        try:
            events = get_upcoming_high_impact_events(EVENT_WARN_HOURS)
            if events:
                ev = events[0]
                cc = f"{ev['country']} " if ev.get("country") else ""
                event_warning = (
                    f"{cc}{ev['title']} через {ev['hours_until']}ч — "
                    f"высокая волатильность, осторожно"
                )
        except Exception as e:
            log.warning(f"Calendar check failed: {e}")

        # Step 5: Send signals to Telegram
        sent_count = 0
        for analysis in analyses:
            try:
                # Attach news context to each analysis for Telegram message
                analysis["news_sentiment"] = news.get("sentiment", "")
                analysis["news_summary"]   = news.get("summary", "")
                analysis["event_warning"]  = event_warning

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
def _app_url() -> str:
    """Return the public URL of this deployment from any known env var."""
    for key in ("APP_URL", "RENDER_EXTERNAL_URL", "RAILWAY_PUBLIC_DOMAIN"):
        val = os.environ.get(key, "").strip().rstrip("/")
        if val:
            # RAILWAY_PUBLIC_DOMAIN gives just the domain, add https://
            if not val.startswith("http"):
                val = f"https://{val}"
            return val
    return ""


def _self_ping():
    """Ping own health endpoint every 4 minutes to keep the service alive."""
    url = _app_url()
    if not url:
        log.info("No APP_URL set — self-ping disabled (local run)")
        return
    while True:
        time.sleep(240)  # 4 minutes
        try:
            _requests.get(f"{url}/", timeout=10)
            log.info("Self-ping OK")
        except Exception as e:
            log.warning(f"Self-ping failed: {e}")


# ── Webhook setup ────────────────────────────────────────────────────────────
def _setup_webhook():
    """Register Telegram webhook so bot can receive messages."""
    url = _app_url()
    if not url:
        log.info("No APP_URL set — webhook skipped (local run)")
        return
    webhook_url = f"{url}/webhook"
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
def _monitor_open_signals():
    """Lightweight 1-min job: check open trades for TP1/SL/BE hits. 24/7."""
    try:
        _check_open_signals()
    except Exception as e:
        log.warning(f"Open-signal monitor failed: {e}")


def start_bot():
    log.info("Starting Crypto Signal Bot...")
    # Proxy diagnostics — shows in Railway logs so we can verify env vars loaded
    _prx = os.environ.get("BYBIT_HTTPS_PROXY", "")
    _prx_base = os.environ.get("BYBIT_PROXY_BASE", "")
    log.info(f"Bybit proxy: HTTPS_PROXY={'SET ('+_prx[:30]+'...)' if _prx else 'NOT SET'} PROXY_BASE={'SET' if _prx_base else 'NOT SET'}")

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

    # Signal scan — every 5 min aligned to candle closes.
    # 15m candles close at :00/:15/:30/:45 → scan at :01/:16/:31/:46 (+1 min buffer).
    scheduler.add_job(
        run_scan, "cron",
        minute="1,6,11,16,21,26,31,36,41,46,51,56",
        timezone="UTC",
    )

    # TP/SL monitor — every 1 min, 24/7, lightweight (only price checks).
    scheduler.add_job(
        _monitor_open_signals, "cron",
        minute="*",
        timezone="UTC",
    )

    scheduler.add_job(
        run_morning_digest, "cron",
        day_of_week="mon-fri", hour=10, minute=0,
        timezone="Europe/Riga",
    )
    scheduler.start()
    log.info("Scheduler: signal scan every 5 min (:01/:06/...), TP/SL monitor every 1 min")

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
