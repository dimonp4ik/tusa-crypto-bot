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


def get_stats(days: int = 7) -> dict:
    """Aggregate honest lifecycle stats. Win rate calculated on final closed signals only."""
    cutoff = time_mod.time() - days * 86400
    with _conn() as c:
        rows = c.execute(
            "SELECT status FROM signals WHERE opened_at >= ?", (cutoff,)
        ).fetchall()

    statuses = [r["status"] for r in rows]
    total    = len(statuses)
    active_open = statuses.count("OPEN")
    active_tp1  = statuses.count("TP1_PARTIAL")
    closed  = sum(1 for s in statuses if s in FINAL_STATUSES)
    tp1_hit = sum(1 for s in statuses if s in TP1_STATUSES)
    tp2_hit = statuses.count("TP2_HIT")
    breakeven   = statuses.count("BREAKEVEN")
    sl_hit      = statuses.count("SL_HIT")
    expired     = statuses.count("EXPIRED")
    tp1_expired = statuses.count("TP1_EXPIRED")
    profitable  = sum(1 for s in statuses if s in PROFIT_STATUSES)

    win_rate = (profitable / closed * 100) if closed else 0.0
    tp1_rate = (tp1_hit / total * 100) if total else 0.0

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
    }
