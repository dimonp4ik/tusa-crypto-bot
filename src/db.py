"""
SQLite database for tracking signal performance.

Lifecycle:
  OPEN         -> signal is live, TP1 not reached yet
  TP1_PARTIAL  -> TP1 hit, 50% position closed, remaining 50% has SL moved to breakeven
  TP2_HIT      -> final target reached after TP1
  BREAKEVEN    -> TP1 hit, remaining 50% closed at entry price
  SL_HIT       -> initial stop hit before TP1
  EXPIRED      -> no TP1/SL within 24h
  TP1_EXPIRED  -> TP1 hit, then rest expired before TP2/BE
"""

import sqlite3
import time as time_mod
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DB_PATH, AUTO_BLOCK_ENABLED, AUTO_BLOCK_LOOKBACK_TRADES, AUTO_BLOCK_MIN_TRADES,
    AUTO_BLOCK_MAX_PROFIT_FACTOR, AUTO_BLOCK_MAX_WIN_RATE, AUTO_BLOCK_DAYS,
)

ACTIVE_STATUSES = ("OPEN", "TP1_PARTIAL")
FINAL_STATUSES  = ("TP2_HIT", "BREAKEVEN", "SL_HIT", "EXPIRED", "TP1_EXPIRED", "TP1_HIT")
TP1_STATUSES    = ("TP1_PARTIAL", "TP2_HIT", "BREAKEVEN", "TP1_EXPIRED", "TP1_HIT")
PROFIT_STATUSES = ("TP2_HIT", "BREAKEVEN", "TP1_EXPIRED", "TP1_HIT")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _ensure_column(conn, table: str, column: str, ddl: str) -> None:
    """SQLite-safe migration helper."""
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db():
    """Create tables if missing and migrate older DBs in place."""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT NOT NULL,
                direction     TEXT NOT NULL,
                entry_price   REAL NOT NULL,
                tp1           REAL NOT NULL,
                tp2           REAL NOT NULL,
                sl            REAL NOT NULL,
                opened_at     REAL NOT NULL,
                status        TEXT NOT NULL DEFAULT 'OPEN',
                closed_at     REAL,
                exit_price    REAL,
                confidence    TEXT,
                reason        TEXT,
                tp1_hit_at    REAL,
                tp1_exit_price REAL,
                entry_low     REAL,
                entry_high    REAL,
                entry_source  TEXT,
                market_price  REAL,
                mtf_score     INTEGER,
                mtf_score_max INTEGER
            )
        """)
        # Migrate older DBs
        for col, ddl in {
            "tp1_hit_at":    "REAL",
            "tp1_exit_price": "REAL",
            "entry_low":     "REAL",
            "entry_high":    "REAL",
            "entry_source":  "TEXT",
            "market_price":  "REAL",
            "mtf_score":     "INTEGER",
            "mtf_score_max": "INTEGER",
        }.items():
            _ensure_column(c, "signals", col, ddl)

        c.execute("""
            CREATE TABLE IF NOT EXISTS symbol_blocks (
                symbol        TEXT PRIMARY KEY,
                blocked_until REAL NOT NULL,
                reason        TEXT,
                created_at    REAL NOT NULL,
                stats_json    TEXT
            )
        """)

        # ── User tracking ────────────────────────────────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id       INTEGER PRIMARY KEY,
                username      TEXT,
                first_name    TEXT,
                last_name     TEXT,
                first_seen    REAL NOT NULL,
                last_seen     REAL NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 1
            )
        """)

        # ── Dynamic admins (added via bot; super-admins stay in config.py) ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                user_id    INTEGER PRIMARY KEY,
                username   TEXT,
                first_name TEXT,
                added_by   INTEGER,
                added_at   REAL NOT NULL
            )
        """)

        # ── Persistent bot state (survives restarts) ─────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS bot_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # ── Claude API budget tracking ────────────────────────────────────────
        # One row per API call. Queried by summing today's spend.
        c.execute("""
            CREATE TABLE IF NOT EXISTS claude_usage (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           REAL NOT NULL,
                tier         TEXT NOT NULL,   -- 'LIGHT' | 'HEAVY'
                input_tok    INTEGER NOT NULL DEFAULT 0,
                output_tok   INTEGER NOT NULL DEFAULT 0,
                cache_write  INTEGER NOT NULL DEFAULT 0,
                cache_read   INTEGER NOT NULL DEFAULT 0,
                cost_usd     REAL NOT NULL DEFAULT 0.0,
                ok           INTEGER NOT NULL DEFAULT 1  -- 0 = failed/timeout
            )
        """)


def get_bot_state(key: str) -> str | None:
    """Read a persistent bot state value. Returns None if key not set."""
    with _conn() as c:
        row = c.execute("SELECT value FROM bot_state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None


def set_bot_state(key: str, value: str) -> None:
    """Write a persistent bot state value (upsert)."""
    with _conn() as c:
        c.execute("""
            INSERT INTO bot_state (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (key, value))


def delete_signal(signal_id: int) -> bool:
    """Hard-delete a signal row by ID. Returns True if a row was removed."""
    with _conn() as c:
        cur = c.execute("DELETE FROM signals WHERE id = ?", (signal_id,))
        return cur.rowcount > 0


def get_recent_signals(limit: int = 20) -> list:
    """Return the most recent signals (any status) for admin review."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM signals ORDER BY opened_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def log_signal(analysis: dict, tp1: float, tp2: float, sl: float):
    """Insert a new signal into DB. Status starts as OPEN."""
    with _conn() as c:
        c.execute("""
            INSERT INTO signals (
                symbol, direction, entry_price, tp1, tp2, sl, opened_at, status,
                confidence, reason, entry_low, entry_high, entry_source, market_price,
                mtf_score, mtf_score_max
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis["symbol"], analysis["direction"], analysis["current_price"],
            tp1, tp2, sl, time_mod.time(),
            analysis.get("confidence", "?"), analysis.get("reason", ""),
            analysis.get("entry_low"), analysis.get("entry_high"),
            analysis.get("entry_source"), analysis.get("market_price"),
            analysis.get("mtf_score"), analysis.get("mtf_score"),
        ))


def get_open_signals() -> list:
    """Return all signals that still need monitoring (OPEN + TP1_PARTIAL)."""
    placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
    with _conn() as c:
        rows = c.execute(
            f"SELECT * FROM signals WHERE status IN ({placeholders})",
            ACTIVE_STATUSES,
        ).fetchall()
        return [dict(r) for r in rows]


def update_signal_status(signal_id: int, status: str, exit_price=None):
    """
    Update signal lifecycle.
    TP1_PARTIAL records TP1 but keeps signal active for TP2/BE monitoring.
    All other statuses close the signal.
    """
    now = time_mod.time()
    with _conn() as c:
        if status == "TP1_PARTIAL":
            c.execute("""
                UPDATE signals
                SET status = 'TP1_PARTIAL', tp1_hit_at = ?, tp1_exit_price = ?
                WHERE id = ? AND status = 'OPEN'
            """, (now, exit_price, signal_id))
        else:
            c.execute("""
                UPDATE signals SET status = ?, closed_at = ?, exit_price = ?
                WHERE id = ?
            """, (status, now, exit_price, signal_id))


def _status_to_r(status: str) -> float:
    """Approximate R for symbol-level blocking."""
    if status == "TP2_HIT":
        return 1.5   # 50% at 1R + 50% at 2R
    if status in ("BREAKEVEN", "TP1_EXPIRED", "TP1_HIT"):
        return 0.5
    if status == "SL_HIT":
        return -1.0
    return 0.0


def get_symbol_performance(symbol: str, lookback: int = None) -> dict:
    """Return recent closed-signal performance for one symbol."""
    lookback = lookback or AUTO_BLOCK_LOOKBACK_TRADES
    placeholders = ",".join("?" for _ in FINAL_STATUSES)
    with _conn() as c:
        rows = c.execute(
            f"SELECT status FROM signals WHERE symbol = ? AND status IN ({placeholders})"
            f" ORDER BY opened_at DESC LIMIT ?",
            [symbol, *FINAL_STATUSES, lookback],
        ).fetchall()

    statuses = [r["status"] for r in rows]
    rs = [_status_to_r(s) for s in statuses]
    gross_profit = sum(r for r in rs if r > 0)
    gross_loss   = abs(sum(r for r in rs if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    wins  = sum(1 for r in rs if r > 0)
    total = len(rs)
    win_rate = wins / total * 100 if total else 0.0

    return {
        "symbol":        symbol,
        "trades":        total,
        "wins":          wins,
        "win_rate":      round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "total_r":       round(sum(rs), 2),
    }


def get_recent_outcomes(symbol: str, limit: int = 8) -> list:
    """Recent final outcomes for one symbol — fuel for HEAVY coin memory."""
    placeholders = ",".join("?" for _ in FINAL_STATUSES)
    with _conn() as c:
        rows = c.execute(
            f"SELECT direction, status, entry_price, exit_price, confidence, mtf_score "
            f"FROM signals WHERE symbol = ? AND status IN ({placeholders}) "
            f"ORDER BY opened_at DESC LIMIT ?",
            [symbol, *FINAL_STATUSES, limit],
        ).fetchall()
    return [dict(r) for r in rows]


def set_symbol_block(symbol: str, days: int, reason: str, stats: dict = None) -> None:
    now   = time_mod.time()
    until = now + days * 86400
    with _conn() as c:
        c.execute("""
            INSERT INTO symbol_blocks (symbol, blocked_until, reason, created_at, stats_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                blocked_until = excluded.blocked_until,
                reason = excluded.reason,
                created_at = excluded.created_at,
                stats_json = excluded.stats_json
        """, (symbol, until, reason, now, json.dumps(stats or {}, ensure_ascii=False)))


def is_symbol_auto_blocked(symbol: str) -> bool:
    now = time_mod.time()
    with _conn() as c:
        row = c.execute(
            "SELECT blocked_until FROM symbol_blocks WHERE symbol = ?", (symbol,)
        ).fetchone()
        if not row:
            return False
        if float(row["blocked_until"]) <= now:
            c.execute("DELETE FROM symbol_blocks WHERE symbol = ?", (symbol,))
            return False
        return True


def get_active_symbol_blocks() -> list:
    now = time_mod.time()
    with _conn() as c:
        c.execute("DELETE FROM symbol_blocks WHERE blocked_until <= ?", (now,))
        rows = c.execute(
            "SELECT * FROM symbol_blocks WHERE blocked_until > ? ORDER BY blocked_until DESC",
            (now,),
        ).fetchall()
        return [dict(r) for r in rows]


def auto_block_bad_symbols() -> list:
    """Block symbols with consistently bad closed-signal stats. No API calls."""
    if not AUTO_BLOCK_ENABLED:
        return []

    placeholders = ",".join("?" for _ in FINAL_STATUSES)
    with _conn() as c:
        symbols = [
            r["symbol"] for r in c.execute(
                f"SELECT DISTINCT symbol FROM signals WHERE status IN ({placeholders})",
                FINAL_STATUSES,
            ).fetchall()
        ]

    blocked = []
    for symbol in symbols:
        if is_symbol_auto_blocked(symbol):
            continue
        perf = get_symbol_performance(symbol)
        if perf["trades"] < AUTO_BLOCK_MIN_TRADES:
            continue
        if perf["profit_factor"] <= AUTO_BLOCK_MAX_PROFIT_FACTOR and \
           perf["win_rate"] <= AUTO_BLOCK_MAX_WIN_RATE:
            reason = (
                f"Auto-block {AUTO_BLOCK_DAYS}d: "
                f"PF={perf['profit_factor']} WR={perf['win_rate']}% trades={perf['trades']}"
            )
            set_symbol_block(symbol, AUTO_BLOCK_DAYS, reason, perf)
            blocked.append({"symbol": symbol, "reason": reason})
    return blocked


def unblock_symbol(symbol: str) -> None:
    """Manually remove a symbol from the block list."""
    with _conn() as c:
        c.execute("DELETE FROM symbol_blocks WHERE symbol = ?", (symbol,))


# ── User tracking ────────────────────────────────────────────────────────────

def upsert_user(user_id: int, username: str = None,
                first_name: str = None, last_name: str = None) -> None:
    """Insert or update a user record on every bot interaction."""
    now = time_mod.time()
    with _conn() as c:
        c.execute("""
            INSERT INTO users (user_id, username, first_name, last_name,
                               first_seen, last_seen, message_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(user_id) DO UPDATE SET
                username      = COALESCE(excluded.username,    username),
                first_name    = COALESCE(excluded.first_name,  first_name),
                last_name     = COALESCE(excluded.last_name,   last_name),
                last_seen     = excluded.last_seen,
                message_count = message_count + 1
        """, (user_id, username, first_name, last_name, now, now))


def get_user_by_id(user_id: int) -> dict | None:
    """Return a single user record or None."""
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE user_id = ?",
                        (user_id,)).fetchone()
        return dict(row) if row else None


def get_all_users(limit: int = 100) -> list:
    """Return up to `limit` users sorted by most recent interaction."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM users ORDER BY last_seen DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Dynamic admin management ──────────────────────────────────────────────────

def add_dynamic_admin(user_id: int, username: str = None,
                      first_name: str = None, added_by: int = None) -> None:
    """Add (or update) a dynamic admin entry in DB."""
    now = time_mod.time()
    with _conn() as c:
        c.execute("""
            INSERT INTO admins (user_id, username, first_name, added_by, added_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username   = COALESCE(excluded.username,   username),
                first_name = COALESCE(excluded.first_name, first_name)
        """, (user_id, username, first_name, added_by, now))


def remove_dynamic_admin(user_id: int) -> None:
    """Remove a dynamic admin from DB."""
    with _conn() as c:
        c.execute("DELETE FROM admins WHERE user_id = ?", (user_id,))


def get_dynamic_admins() -> list:
    """Return all dynamic admins ordered by when they were added."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM admins ORDER BY added_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def is_dynamic_admin(user_id: int) -> bool:
    """True when user_id has an entry in the admins table."""
    with _conn() as c:
        return c.execute(
            "SELECT 1 FROM admins WHERE user_id = ?", (user_id,)
        ).fetchone() is not None


# ── Claude budget tracking ────────────────────────────────────────────────────

# Pricing per 1M tokens (USD) — update if Anthropic changes rates.
_CLAUDE_PRICES = {
    # model_prefix: (input, cache_write, cache_read, output)
    "claude-haiku":  (1.00, 1.25, 0.10, 5.00),
    "claude-sonnet": (3.00, 3.75, 0.30, 15.00),
}

def _model_price(model: str) -> tuple:
    for prefix, prices in _CLAUDE_PRICES.items():
        if prefix in model.lower():
            return prices
    return _CLAUDE_PRICES["claude-haiku"]   # safe default


def log_claude_call(tier: str, model: str, usage, ok: bool = True) -> float:
    """
    Record one Claude API call and return its cost in USD.
    `usage` is the anthropic Usage object (input_tokens, output_tokens,
    cache_creation_input_tokens, cache_read_input_tokens).
    """
    inp  = getattr(usage, "input_tokens", 0) or 0
    out  = getattr(usage, "output_tokens", 0) or 0
    cw   = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr   = getattr(usage, "cache_read_input_tokens", 0) or 0

    p_in, p_cw, p_cr, p_out = _model_price(model)
    cost = (inp * p_in + cw * p_cw + cr * p_cr + out * p_out) / 1_000_000

    with _conn() as c:
        c.execute("""
            INSERT INTO claude_usage (ts, tier, input_tok, output_tok,
                                      cache_write, cache_read, cost_usd, ok)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (time_mod.time(), tier, inp, out, cw, cr, round(cost, 6), int(ok)))
    return round(cost, 6)


def get_claude_spend_today() -> float:
    """Return total Claude USD spend since midnight UTC today."""
    import time as _t
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    with _conn() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(cost_usd),0) FROM claude_usage WHERE ts >= ?",
            (midnight,)
        ).fetchone()
    return float(row[0])


def get_claude_spend_stats() -> dict:
    """Return spend summary: today, this week, total."""
    import time as _t
    from datetime import datetime, timezone
    now_ts = _t.time()
    now    = datetime.now(timezone.utc)
    today  = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    week   = now_ts - 7 * 86400
    with _conn() as c:
        def _sum(since):
            r = c.execute(
                "SELECT COALESCE(SUM(cost_usd),0), COUNT(*) FROM claude_usage WHERE ts >= ?",
                (since,)
            ).fetchone()
            return round(float(r[0]), 4), int(r[1])
        today_usd, today_calls = _sum(today)
        week_usd,  week_calls  = _sum(week)
        total_usd, total_calls = _sum(0)
    return {
        "today_usd": today_usd, "today_calls": today_calls,
        "week_usd":  week_usd,  "week_calls":  week_calls,
        "total_usd": total_usd, "total_calls": total_calls,
    }


def get_symbols_performance(days: int = 30) -> list:
    """
    Per-symbol closed-signal performance over `days` days.
    Returns list of dicts sorted by total_r descending.
    """
    cutoff = time_mod.time() - days * 86400
    placeholders = ",".join("?" for _ in FINAL_STATUSES)
    with _conn() as c:
        rows = c.execute(
            f"SELECT symbol, status FROM signals "
            f"WHERE opened_at >= ? AND status IN ({placeholders})",
            [cutoff, *FINAL_STATUSES],
        ).fetchall()

    from collections import defaultdict
    by_sym: dict = defaultdict(list)
    for r in rows:
        by_sym[r["symbol"]].append(_status_to_r(r["status"]))

    results = []
    for sym, rs in by_sym.items():
        total   = len(rs)
        wins    = sum(1 for r in rs if r > 0)
        total_r = round(sum(rs), 2)
        results.append({
            "symbol":   sym,
            "trades":   total,
            "wins":     wins,
            "win_rate": round(wins / total * 100, 1) if total else 0.0,
            "total_r":  total_r,
        })

    results.sort(key=lambda x: x["total_r"], reverse=True)
    return results


def _status_r(status: str) -> float:
    """R value of a closed trade outcome (fixed R model, TP1=1.5R TP2=3.0R SL=1R)."""
    # TP2: 50% closed at TP1 (0.75R) + 50% at TP2 (1.5R) = 2.25R
    if status == "TP2_HIT":    return  2.25
    # TP1 only outcomes: 50% at TP1 = 0.75R
    if status in ("TP1_HIT", "BREAKEVEN", "TP1_EXPIRED"): return 0.75
    # Full SL before TP1
    if status == "SL_HIT":     return -1.00
    # Expired before any TP — no profit, small fee drag (treat as 0)
    return 0.0


def get_stats(days: int = 7, since_ts: float = None) -> dict:
    """Aggregate stats with R-value, direction breakdown and recent streak.

    `days`     — rolling window (last N×24h) when since_ts is None.
    `since_ts` — explicit epoch cutoff (e.g. Riga midnight for calendar 'today').
    """
    cutoff = since_ts if since_ts is not None else time_mod.time() - days * 86400
    with _conn() as c:
        rows = c.execute(
            "SELECT status, direction, opened_at FROM signals WHERE opened_at >= ?",
            (cutoff,)
        ).fetchall()
        # Last 7 closed signals for streak (independent of days filter)
        streak_rows = c.execute(
            f"SELECT status FROM signals "
            f"WHERE status IN ({','.join('?'*len(FINAL_STATUSES))}) "
            f"ORDER BY opened_at DESC LIMIT 7",
            FINAL_STATUSES,
        ).fetchall()

    rows = [dict(r) for r in rows]

    # ── Basic counts ──────────────────────────────────────────────────────────
    total       = len(rows)
    active_open = sum(1 for r in rows if r["status"] == "OPEN")
    active_tp1  = sum(1 for r in rows if r["status"] == "TP1_PARTIAL")
    closed      = sum(1 for r in rows if r["status"] in FINAL_STATUSES)
    tp1_hit     = sum(1 for r in rows if r["status"] in TP1_STATUSES)
    tp2_hit     = sum(1 for r in rows if r["status"] == "TP2_HIT")
    breakeven   = sum(1 for r in rows if r["status"] == "BREAKEVEN")
    sl_hit      = sum(1 for r in rows if r["status"] == "SL_HIT")
    expired     = sum(1 for r in rows if r["status"] == "EXPIRED")
    tp1_expired = sum(1 for r in rows if r["status"] == "TP1_EXPIRED")
    profitable  = sum(1 for r in rows if r["status"] in PROFIT_STATUSES)

    win_rate = (profitable / closed * 100) if closed else 0.0
    tp1_rate = (tp1_hit    / total  * 100) if total  else 0.0

    # ── Total R ───────────────────────────────────────────────────────────────
    total_r = sum(_status_r(r["status"]) for r in rows if r["status"] in FINAL_STATUSES)
    r_per_trade = (total_r / closed) if closed else 0.0

    # ── Direction breakdown ───────────────────────────────────────────────────
    dir_stats = {}
    for direction in ("LONG", "SHORT"):
        dr = [r for r in rows if r.get("direction") == direction]
        dr_closed = [r for r in dr if r["status"] in FINAL_STATUSES]
        dr_wins   = sum(1 for r in dr_closed if r["status"] in PROFIT_STATUSES)
        dr_r      = sum(_status_r(r["status"]) for r in dr_closed)
        dir_stats[direction] = {
            "total":    len(dr),
            "closed":   len(dr_closed),
            "wins":     dr_wins,
            "win_rate": round(dr_wins / len(dr_closed) * 100, 1) if dr_closed else 0.0,
            "total_r":  round(dr_r, 2),
        }

    # ── Recent streak (last 7 closed, newest first) ───────────────────────────
    streak = []
    for r in streak_rows:
        st = r["status"]
        if st == "TP2_HIT":
            streak.append("🏆")
        elif st in PROFIT_STATUSES:
            streak.append("✅")
        elif st == "SL_HIT":
            streak.append("❌")
        else:
            streak.append("➖")
    # Count current run (consecutive same outcome from newest)
    current_run = 1
    if len(streak) >= 2:
        for i in range(1, len(streak)):
            if streak[i] == streak[0]:
                current_run += 1
            else:
                break

    return {
        "days":             days,
        "total":            total,
        "closed":           closed,
        "open":             active_open,
        "tp1_partial_open": active_tp1,
        "tp1_hit":          tp1_hit,
        "tp1_rate":         round(tp1_rate, 1),
        "tp2_hit":          tp2_hit,
        "breakeven":        breakeven,
        "tp1_expired":      tp1_expired,
        "sl_hit":           sl_hit,
        "expired":          expired,
        "win_rate":         round(win_rate, 1),
        "total_r":          round(total_r, 2),
        "r_per_trade":      round(r_per_trade, 3),
        "long":             dir_stats.get("LONG",  {}),
        "short":            dir_stats.get("SHORT", {}),
        "streak":           streak,
        "current_run":      current_run,
    }
