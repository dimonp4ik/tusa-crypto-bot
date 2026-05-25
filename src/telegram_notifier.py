import requests
from datetime import datetime, timezone
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Максимальный стоп-лосс от цены входа (5%)
MAX_SL_PERCENT = 0.05


def _calculate_tp_sl(price: float, direction: str, recent_high: float, recent_low: float):
    """
    LONG:  SL = недавний минимум,  TP = цена + 2 * риск  (соотношение 2:1)
    SHORT: SL = недавний максимум, TP = цена - 2 * риск  (соотношение 2:1)
    Если SL дальше 5% от цены — ограничиваем 3%.
    """
    if direction == "LONG":
        sl = recent_low
        if price > 0 and (price - sl) / price > MAX_SL_PERCENT:
            sl = price * 0.97
        risk = price - sl
        tp = price + risk * 2

    else:  # SHORT
        sl = recent_high
        if price > 0 and (sl - price) / price > MAX_SL_PERCENT:
            sl = price * 1.03
        risk = sl - price
        tp = price - risk * 2

    return round(tp, 6), round(sl, 6)


def _format_price(price: float) -> str:
    """Убираем лишние нули, оставляем до 6 знаков."""
    if price >= 1000:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    else:
        return f"{price:.6f}"


def send_signal(analysis: dict) -> bool:
    """Форматирует и отправляет торговый сигнал в Telegram. Возвращает True при успехе."""
    decision = analysis["decision"]
    if decision == "NO TRADE":
        return False

    price = analysis["current_price"]
    direction = analysis["direction"]
    recent_high = analysis.get("recent_high", price * 1.03)
    recent_low = analysis.get("recent_low", price * 0.97)

    tp, sl = _calculate_tp_sl(price, direction, recent_high, recent_low)

    # Иконки
    arrow     = "🟢 ЛОНГ" if decision == "LONG" else "🔴 ШОРТ"
    conf_icon = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "⚠️"}.get(
        analysis.get("confidence", ""), "⚡"
    )
    conf_ru   = {"HIGH": "ВЫСОКАЯ", "MEDIUM": "СРЕДНЯЯ", "LOW": "НИЗКАЯ"}.get(
        analysis.get("confidence", ""), "—"
    )

    signals_text = "\n".join(f"  • {s}" for s in analysis["signals"])
    timestamp = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")

    message = (
        f"{arrow} — *{analysis['symbol']}*\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Цена входа:   `{_format_price(price)}`\n"
        f"✅ Тейк профит:  `{_format_price(tp)}`\n"
        f"❌ Стоп лосс:    `{_format_price(sl)}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📊 RSI: `{analysis['rsi']}`   "
        f"📈 Объём: `{analysis['volume_ratio']}x`\n\n"
        f"*Сигналы:*\n{signals_text}\n\n"
        f"{conf_icon} Уверенность: *{conf_ru}*\n"
        f"📝 _{analysis.get('reason', '')}_\n\n"
        f"⏰ {timestamp}"
    )

    return _send_message(message)


def send_status(text: str) -> bool:
    """Отправляет системное сообщение."""
    return _send_message(text)


def _send_message(text: str) -> bool:
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
            },
            timeout=15,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"[Telegram] Ошибка отправки: {e}")
        return False
