"""
SQLite database for tracking signal performance.
Logs every signal sent, then checks open signals each scan to update outcomes.
"""

import sqlite3
import time as time_mod
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """Create tables if missing."""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                direction   TEXT NOT NULL,
                entry_price REAL NOT NULL,
                tp1         REAL NOT NULL,
                tp2         REAL NOT NULL,
                sl          REAL NOT NULL,
                opened_at   REAL NOT NULL,
                status      TEXT NOT NULL DEFAULT 'OPEN',
                closed_at   REAL,
                exit_price  REAL,
                confidence  TEXT,
                reason      TEXT
            )
        """)


def log_signal(analysis: dict, tp1: float, tp2: float, sl: float):
    """Insert a new signal into DB. Status starts as OPEN."""
    with _conn() as c:
        c.execute("""
            INSERT INTO signals (symbol, direction, entry_price, tp1, tp2, sl,
                                 opened_at, status, confidence, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
        """, (
            analysis["symbol"], analysis["direction"], analysis["current_price"],
            tp1, tp2, sl, time_mod.time(),
            analysis.get("confidence", "?"), analysis.get("reason", ""),
        ))


def get_open_signals() -> list:
    """Return all signals with status OPEN."""
    with _conn() as c:
        rows = c.execute("SELECT * FROM signals WHERE status = 'OPEN'").fetchall()
        return [dict(r) for r in rows]


def update_signal_status(signal_id: int, status: str, exit_price: float):
    """Mark a signal closed with given status (TP1_HIT, TP2_HIT, SL_HIT, EXPIRED)."""
    with _conn() as c:
        c.execute("""
            UPDATE signals SET status = ?, closed_at = ?, exit_price = ?
            WHERE id = ?
        """, (status, time_mod.time(), exit_price, signal_id))


def get_stats(days: int = 7) -> dict:
    """
    Aggregate stats over last N days.
    Returns dict with totals and win rate.
    """
    cutoff = time_mod.time() - days * 86400
    with _conn() as c:
        rows = c.execute(
            "SELECT status FROM signals WHERE opened_at >= ? AND status != 'OPEN'",
            (cutoff,)
        ).fetchall()

        total    = len(rows)
        tp1_hit  = sum(1 for r in rows if r["status"] in ("TP1_HIT", "TP2_HIT"))
        tp2_hit  = sum(1 for r in rows if r["status"] == "TP2_HIT")
        sl_hit   = sum(1 for r in rows if r["status"] == "SL_HIT")
        expired  = sum(1 for r in rows if r["status"] == "EXPIRED")

        win_rate = (tp1_hit / total * 100) if total else 0.0

        return {
            "days":     days,
            "total":    total,
            "tp1_hit":  tp1_hit,
            "tp2_hit":  tp2_hit,
            "sl_hit":   sl_hit,
            "expired":  expired,
            "win_rate": round(win_rate, 1),
        }
