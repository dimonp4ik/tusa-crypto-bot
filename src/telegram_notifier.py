import requests
from datetime import datetime, timezone
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    ATR_SL_MULT, ATR_TP1_MULT, ATR_TP2_MULT,
)

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Max SL distance fallback when ATR unavailable
MAX_SL_PERCENT = 0.05


def calculate_tp_sl(price: float, direction: str, atr: float = 0.0,
                    recent_high: float = 0.0, recent_low: float = 0.0):
    """
    ATR-based TP1, TP2, SL.

    LONG:
      SL  = price - ATR * 1.5
      TP1 = price + ATR * 1.5  (1:1 → close 50%, move SL to BE)
      TP2 = price + ATR * 3.0  (1:2 → close rest)

    Fallback to recent_high/low if ATR is zero.
    """
    if atr and atr > 0:
        if direction == "LONG":
            sl  = price - atr * ATR_SL_MULT
            tp1 = price + atr * ATR_TP1_MULT
            tp2 = price + atr * ATR_TP2_MULT
        else:  # SHORT
            sl  = price + atr * ATR_SL_MULT
            tp1 = price - atr * ATR_TP1_MULT
            tp2 = price - atr * ATR_TP2_MULT
    else:
        # Legacy fallback
        if direction == "LONG":
            sl = recent_low or price * 0.97
            if (price - sl) / price > MAX_SL_PERCENT:
                sl = price * 0.97
            risk = price - sl
            tp1 = price + risk
            tp2 = price + risk * 2
        else:
            sl = recent_high or price * 1.03
            if (sl - price) / price > MAX_SL_PERCENT:
                sl = price * 1.03
            risk = sl - price
            tp1 = price - risk
            tp2 = price - risk * 2

    return round(tp1, 8), round(tp2, 8), round(sl, 8)


def _format_price(price: float) -> str:
    if price >= 1000:  return f"{price:,.2f}"
    if price >= 1:     return f"{price:.4f}"
    return f"{price:.6f}"


def send_signal(analysis: dict) -> bool:
    """Format and send a trading signal. Returns True on success."""
    decision = analysis["decision"]
    if decision == "NO TRADE":
        return False

    price     = analysis["current_price"]
    direction = analysis["direction"]
    atr       = analysis.get("atr", 0.0)
    rec_high  = analysis.get("recent_high", price * 1.03)
    rec_low   = analysis.get("recent_low",  price * 0.97)

    tp1, tp2, sl = calculate_tp_sl(price, direction, atr, rec_high, rec_low)

    arrow     = "🟢 ЛОНГ" if decision == "LONG" else "🔴 ШОРТ"
    conf_icon = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "⚠️"}.get(analysis.get("confidence", ""), "⚡")
    conf_ru   = {"HIGH": "ВЫСОКАЯ", "MEDIUM": "СРЕДНЯЯ", "LOW": "НИЗКАЯ"}.get(analysis.get("confidence", ""), "—")

    session_icons = {
        "LONDON":    "🇬🇧 London",
        "NEW_YORK":  "🇺🇸 New York",
        "OVERLAP":   "🔥 London/NY",
        "OFF_HOURS": "🌙 Off-hours",
    }
    session_str  = session_icons.get(analysis.get("session", ""), "")
    signals_text = "\n".join(f"  • {s}" for s in analysis["signals"])
    timestamp    = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")

    btc_change   = analysis.get("btc_change", 0)
    btc_line     = f"₿ BTC за час: `{btc_change:+.2f}%`\n" if btc_change else ""
    news_sent    = analysis.get("news_sentiment", "")
    news_summary = analysis.get("news_summary", "")
    news_icon    = {"BULLISH": "📰🟢", "BEARISH": "📰🔴"}.get(news_sent, "")
    news_line    = f"{news_icon} _{news_summary}_\n" if news_sent and news_summary and news_sent != "NEUTRAL" else ""

    message = (
        f"{arrow} — *{analysis['symbol']}*\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Цена входа:  `{_format_price(price)}`\n"
        f"🎯 TP1 (50%):   `{_format_price(tp1)}`  _→ перенеси SL в б/у_\n"
        f"🎯 TP2 (50%):   `{_format_price(tp2)}`\n"
        f"❌ Стоп лосс:   `{_format_price(sl)}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📊 RSI: `{analysis['rsi']}`   📈 Объём: `{analysis['volume_ratio']}x`\n"
        f"{btc_line}"
        f"{news_line}"
        f"\n*Сигналы:*\n{signals_text}\n\n"
        f"{conf_icon} Уверенность: *{conf_ru}*\n"
        f"📝 _{analysis.get('reason', '')}_\n\n"
        f"🕐 {session_str}  ⏰ {timestamp}"
    )

    if _send_message(message):
        # Log to DB
        try:
            from src.db import log_signal
            log_signal(analysis, tp1, tp2, sl)
        except Exception as e:
            print(f"[DB] log_signal failed: {e}")
        return True
    return False


def send_status(text: str) -> bool:
    return _send_message(text)


def _send_message(text: str) -> bool:
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"[Telegram] Ошибка отправки: {e}")
        return False
