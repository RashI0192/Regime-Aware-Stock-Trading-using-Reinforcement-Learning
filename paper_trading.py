"""Small SQLite paper-trading ledger; it never submits brokerage orders."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "models" / "paper_trading.sqlite3"

def _connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS paper_trades (
        id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, trade_date TEXT NOT NULL,
        price REAL NOT NULL, action TEXT NOT NULL, probability_up REAL,
        model_version TEXT NOT NULL, fee REAL NOT NULL DEFAULT 0.001)""")
    con.commit(); return con

def record_signal(symbol, price, action, probability_up, model_version, fee=0.001):
    with _connect() as con:
        cur = con.execute("INSERT INTO paper_trades(symbol,trade_date,price,action,probability_up,model_version,fee) VALUES (?,?,?,?,?,?,?)",
                          (symbol, datetime.now(timezone.utc).isoformat(), price, action, probability_up, model_version, fee))
        return cur.lastrowid

def recent_trades(limit=50):
    with _connect() as con:
        return [dict(row) for row in con.execute("SELECT * FROM paper_trades ORDER BY id DESC LIMIT ?", (limit,))]
