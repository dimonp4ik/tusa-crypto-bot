import requests
from datetime import datetime, timezone
import sys
import os

try:
    from zoneinfo import ZoneInfo
    _RIGA = ZoneInfo("Europe/Riga")
except Exception:
    from datetime import timedelta
    _RIGA = timezone(timedelta(hours=3))  # fallback UTC+3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    SL_ATR_BUFFER, RISK_MIN_PCT, RISK_MAX_PCT, TP1_R_MULT, TP2_R_MULT,
)

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def calculate_tp_sl(price: float, direction: str, atr: float = 0.0,
                    recent_high: float = 0.0, recent_low: float = 0.0,
                    tp1_level: float = None, tp2_level: float = None):
    """
    Structure-based SL + smart structural TP for swing trading (15m, ~20x leverage).

    SL  — placed at swing invalidation (recent_low/high) + ATR buffer, clamped
          to RISK_MIN_PCT..RISK_MAX_PCT of price for safe leverage.

    TP1 — nearest confirmed swing high/low (tp1_level) when it gives ≥ 1.5R.
          Falls back to price ± risk * TP1_R_MULT.

    TP2 — next swing level (tp2_level) when it's further than TP1.
          Falls back to price ± risk * TP2_R_MULT.

    This way targets align with real market structure, not arbitrary multiples.
    """
    min_risk = price * RISK_MIN_PCT
    max_risk = price * RISK_MAX_PCT
    buf      = atr * SL_ATR_BUFFER if (atr and atr > 0) else 0.0

    if direction == "LONG":
        struct_sl = (recent_low - buf) if recent_low and recent_low > 0 else price - max_risk
        risk = price - struct_sl
        risk = min(max(risk, min_risk), max_risk)
        sl   = price - risk

        # TP1: structural swing high if valid (min 1.0R away, above price)
        if tp1_level and tp1_level > price * 1.001 and (tp1_level - price) >= risk * 1.0:
            tp1 = tp1_level
        else:
            tp1 = price + risk * TP1_R_MULT

        # TP2: next structural level above TP1 AND at least 1.5R from entry
        if tp2_level and tp2_level > tp1 * 1.001 and (tp2_level - price) >= risk * 1.5:
            tp2 = tp2_level
        else:
            tp2 = price + risk * TP2_R_MULT
            if tp2 <= tp1:        # ensure TP2 > TP1
                tp2 = tp1 * 1.02

    else:  # SHORT
        struct_sl = (recent_high + buf) if recent_high and recent_high > 0 else price + max_risk
        risk = struct_sl - price
        risk = min(max(risk, min_risk), max_risk)
        sl   = price + risk

        # TP1: structural swing low if valid (min 1.0R away, below price)
        if tp1_level and tp1_level < price * 0.999 and (price - tp1_level) >= risk * 1.0:
            tp1 = tp1_level
        else:
            tp1 = price - risk * TP1_R_MULT

        # TP2: next structural level below TP1 AND at least 1.5R from entry
        if tp2_level and tp2_level < tp1 * 0.999 and (price - tp2_level) >= risk * 1.5:
            tp2 = tp2_level
        else:
            tp2 = price - risk * TP2_R_MULT
            if tp2 >= tp1:        # ensure TP2 < TP1
                tp2 = tp1 * 0.98

    return round(tp1, 8), round(tp2, 8), round(sl, 8)


def _format_price(price: float) -> str:
    if price >= 1000:  return f"{price:,.2f}"
    if price >= 1:     return f"{price:.4f}"
    return f"{price:.6f}"


def recommend_leverage(price: float, sl: float, tp1: float, tp2: float,
                       direction: str, mtf_score: int) -> dict:
    """
    Recommend Bybit leverage based on SL distance and setup quality.

    Logic:
      max_safe_lev = 70% / sl_pct  (SL stays above liquidation with safety buffer)
      quality_mult  from MTF score (50–100% of max_safe)
      Rounds down to nearest Bybit tier [5,10,15,20,25,30,40,50]

    Returns leverage, liquidation price, and profit/loss % at that leverage.
    """
    if price <= 0:
        return {"leverage": 10, "max_safe": 10, "rating": "ХОРОШИЙ ✅",
                "liq": 0.0, "tp1_profit": 0.0, "tp2_profit": 0.0, "sl_loss": 0.0}

    sl_pct = abs(price - sl) / price
    if sl_pct <= 0:
        sl_pct = 0.03

    max_safe = min(50, max(5, int(0.70 / sl_pct)))

    if mtf_score >= 13:
        quality = 1.00; rating = "ИДЕАЛ 🔥"
    elif mtf_score >= 11:
        quality = 0.75; rating = "СИЛЬНЫЙ ⚡"
    else:
        quality = 0.50; rating = "ХОРОШИЙ ✅"

    rec = max(5, int(max_safe * quality))
    tiers = [5, 10, 15, 20, 25, 30, 40, 50]
    final = max(t for t in tiers if t <= rec)

    # Liquidation price at recommended leverage (Bybit isolated, ~0.9/lev distance)
    if direction == "LONG":
        liq = price * (1 - 0.9 / final)
    else:
        liq = price * (1 + 0.9 / final)

    tp1_profit = abs(tp1 - price) / price * final * 100
    tp2_profit = abs(tp2 - price) / price * final * 100
    sl_loss    = abs(sl  - price) / price * final * 100

    return {
        "leverage":   final,
        "max_safe":   max_safe,
        "rating":     rating,
        "liq":        round(liq, 8),
        "tp1_profit": round(tp1_profit, 0),
        "tp2_profit": round(tp2_profit, 0),
        "sl_loss":    round(sl_loss, 0),
    }


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

    tp1, tp2, sl = calculate_tp_sl(
        price, direction, atr, rec_high, rec_low,
        tp1_level=analysis.get("tp1_level"),
        tp2_level=analysis.get("tp2_level"),
    )

    mtf_score = int(analysis.get("mtf_score", 9) or 9)
    lev_info  = recommend_leverage(price, sl, tp1, tp2, direction, mtf_score)

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
    timestamp    = datetime.now(_RIGA).strftime("%d.%m.%Y %H:%M (Рига)")

    btc_change   = analysis.get("btc_change", 0)
    btc_line     = f"₿ BTC за час: `{btc_change:+.2f}%`\n" if btc_change else ""
    news_sent    = analysis.get("news_sentiment", "")
    news_summary = analysis.get("news_summary", "")
    news_icon    = {"BULLISH": "📰🟢", "BEARISH": "📰🔴"}.get(news_sent, "")
    news_line    = f"{news_icon} _{news_summary}_\n" if news_sent and news_summary and news_sent != "NEUTRAL" else ""
    event_warn   = analysis.get("event_warning", "")
    event_line   = f"⚠️ {event_warn}\n" if event_warn else ""

    lev = lev_info["leverage"]
    premium_badge = "  💎 *PREMIUM*" if analysis.get("premium") else ""
    message = (
        f"{arrow} — *{analysis['symbol']}*{premium_badge}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Вход:        `{_format_price(price)}`\n"
        f"🎯 TP1 (50%):   `{_format_price(tp1)}`  → SL в б/у\n"
        f"🎯 TP2 (50%):   `{_format_price(tp2)}`\n"
        f"❌ Стоп лосс:   `{_format_price(sl)}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ Плечо: *{lev}x*  ({lev_info['rating']})\n"
        f"   TP1 `+{lev_info['tp1_profit']:.0f}%`  TP2 `+{lev_info['tp2_profit']:.0f}%`  SL `-{lev_info['sl_loss']:.0f}%`\n"
        f"   Ликвидация x{lev}: `{_format_price(lev_info['liq'])}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📊 RSI: `{analysis['rsi']}`   📈 Объём: `{analysis['volume_ratio']}x`\n"
        f"{btc_line}"
        f"{news_line}"
        f"{event_line}"
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


def send_signal_update(sig: dict, new_status: str, exit_price: float) -> bool:
    """
    Send TP/SL hit notification for a tracked signal.
    sig = row dict from get_open_signals() (has symbol, direction, entry_price, tp1, tp2, sl).
    """
    symbol    = sig["symbol"]
    direction = sig["direction"]
    entry     = float(sig["entry_price"])
    tp2       = float(sig["tp2"])
    sl        = float(sig["sl"])

    # Price move from entry to exit (positive = profit for LONG)
    move_pct = (exit_price - entry) / entry * 100 if entry > 0 else 0.0
    if direction == "SHORT":
        move_pct = -move_pct

    # Estimate leverage from SL distance (mirrors recommend_leverage logic)
    sl_pct = abs(entry - sl) / entry if entry > 0 else 0.02
    if sl_pct > 0:
        raw = int(0.70 / sl_pct)
        tiers = [5, 10, 15, 20, 25, 30, 40, 50]
        lev = max(t for t in tiers if t <= max(5, raw))
    else:
        lev = 10
    lev_profit = round(move_pct * lev, 0)

    arrow = "🟢" if direction == "LONG" else "🔴"
    timestamp = datetime.now(_RIGA).strftime("%d.%m.%Y %H:%M (Рига)")
    sign = "+" if lev_profit >= 0 else ""

    if new_status == "TP1_PARTIAL":
        icon  = "✅"
        title = "TP1 ДОСТИГНУТ"
        body  = (
            f"Закрыто 50% по `{_format_price(exit_price)}`\n"
            f"Движение: `{sign}{move_pct:.2f}%`  (x{lev}: `{sign}{lev_profit:.0f}%`)\n"
            f"🔄 SL перенесён в безубыток: `{_format_price(entry)}`\n"
            f"Ждём TP2: `{_format_price(tp2)}`"
        )
    elif new_status == "TP2_HIT":
        icon  = "🎯"
        title = "TP2 ДОСТИГНУТ"
        body  = (
            f"Закрыто 50% по `{_format_price(exit_price)}`\n"
            f"Движение: `{sign}{move_pct:.2f}%`  (x{lev}: `{sign}{lev_profit:.0f}%`)\n"
            f"✅ Сделка полностью закрыта"
        )
    elif new_status == "BREAKEVEN":
        icon  = "🔄"
        title = "БЕЗУБЫТОК"
        body  = (
            f"TP1 был взят, остаток закрыт по входу\n"
            f"Цена: `{_format_price(exit_price)}`"
        )
    elif new_status == "SL_HIT":
        icon  = "❌"
        title = "СТОП ЛОСС"
        body  = (
            f"Закрыто по `{_format_price(exit_price)}`\n"
            f"Движение: `{move_pct:.2f}%`  (x{lev}: `{lev_profit:.0f}%`)"
        )
    elif new_status == "EXPIRED":
        icon  = "⌛"
        title = "ИСТЁК (48ч)"
        body  = f"Цена: `{_format_price(exit_price)}`  — цель не достигнута"
    elif new_status == "TP1_EXPIRED":
        icon  = "⏳"
        title = "TP1 ИСТЁК"
        body  = (
            f"TP1 был взят, TP2 не достигнут за 48ч\n"
            f"Цена: `{_format_price(exit_price)}`"
        )
    else:
        return False

    message = (
        f"{icon} *{title}* — {arrow} *{symbol}*\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Вход: `{_format_price(entry)}`\n"
        f"{body}\n"
        f"⏰ {timestamp}"
    )
    return _send_message(message)


def send_news_alert(event: dict) -> bool:
    """
    Send investing.com-style high-impact news alert.
    event keys: name, direction (BULLISH/BEARISH), level (1-3), explanation
    """
    direction = event.get("direction", "NEUTRAL")
    level     = min(max(int(event.get("level", 1)), 1), 3)
    name      = event.get("name", "")
    expl      = event.get("explanation", "")

    if direction == "BULLISH":
        icons     = "🐂" * level
        impact_ru = "БЫЧЬЕ"
        dir_icon  = "📈"
    elif direction == "BEARISH":
        icons     = "🐻" * level
        impact_ru = "МЕДВЕЖЬЕ"
        dir_icon  = "📉"
    else:
        icons     = "⚪" * level
        impact_ru = "НЕЙТРАЛЬНОЕ"
        dir_icon  = "➡️"

    timestamp = datetime.now(_RIGA).strftime("%d.%m.%Y %H:%M (Рига)")

    message = (
        f"⚡ *ВАЖНАЯ НОВОСТЬ*\n"
        f"🏦 {name}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"{dir_icon} Влияние: {icons} *{impact_ru}*\n"
        f"_{expl}_\n"
        f"⏰ {timestamp}"
    )
    return _send_message(message)


def send_morning_digest(digest: dict) -> bool:
    """Format and send the daily morning news digest."""
    items   = digest.get("items", [])
    overall = digest.get("overall", "NEUTRAL")
    theme   = digest.get("key_theme", "")

    if not items:
        return _send_message("🌅 *УТРЕННИЙ ДАЙДЖЕСТ*\nНовостей за последние 18 часов не найдено.")

    _RU_MONTHS = ["янв","фев","мар","апр","май","июн","июл","авг","сен","окт","ноя","дек"]
    _RU_DAYS   = ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"]
    now_riga = datetime.now(_RIGA)
    date_str = f"{now_riga.day} {_RU_MONTHS[now_riga.month-1]} ({_RU_DAYS[now_riga.weekday()]})"

    overall_map = {"BULLISH": "🟢 БЫЧИЙ", "BEARISH": "🔴 МЕДВЕЖИЙ", "NEUTRAL": "⚪ НЕЙТРАЛЬНЫЙ"}
    overall_str = overall_map.get(overall, "⚪ НЕЙТРАЛЬНЫЙ")

    dir_icons = {"BULLISH": "📈", "BEARISH": "📉", "NEUTRAL": "➡️"}

    def _utc_to_riga(time_str: str) -> str:
        """Convert 'HH:MM' (UTC) string to Riga time string."""
        if not time_str or time_str == "?":
            return "?"
        try:
            clean = time_str.replace("UTC", "").replace("utc", "").strip()
            hh, mm = int(clean.split(":")[0]), int(clean.split(":")[1])
            dt_utc = datetime.now(timezone.utc).replace(hour=hh, minute=mm, second=0, microsecond=0)
            return dt_utc.astimezone(_RIGA).strftime("%H:%M")
        except Exception:
            return time_str

    lines = [
        f"🌅 *УТРЕННИЙ ДАЙДЖЕСТ* — {date_str}",
        "━━━━━━━━━━━━━━━━━━━",
        "📰 *ТОП НОВОСТЕЙ*\n",
    ]

    for i, item in enumerate(items, 1):
        direction = item.get("direction", "NEUTRAL")
        icon      = dir_icons.get(direction, "➡️")
        t_riga    = _utc_to_riga(item.get("time_utc", "?"))
        title     = item.get("title", "")
        expl      = item.get("explanation", "")
        impact    = item.get("impact", "")
        lines.append(
            f"{i}\\. {icon} *{title}*\n"
            f"   ⏰ {t_riga} (Рига)  _{expl}_\n"
            f"   📊 _{impact}_"
        )

    lines += [
        "\n━━━━━━━━━━━━━━━━━━━",
        f"📊 Общий фон: *{overall_str}*",
    ]
    if theme:
        lines.append(f"🔑 _{theme}_")

    return _send_message("\n".join(lines))


def send_status(text: str) -> bool:
    return _send_message(text)


def _send_message(text: str) -> bool:
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"[Telegram] HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.status_code == 200
    except Exception as e:
        print(f"[Telegram] Ошибка отправки: {e}")
        return False
