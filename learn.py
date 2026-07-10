import sqlite3
import os
import json
from datetime import datetime, timezone

_BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE, "database", "learn.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user TEXT,
            role TEXT,
            question TEXT NOT NULL,
            sql_generated TEXT,
            answer TEXT,
            had_dashboard INTEGER DEFAULT 0,
            row_count INTEGER DEFAULT 0,
            retrieved_sources TEXT,
            success INTEGER DEFAULT 1,
            feedback INTEGER DEFAULT NULL,
            feedback_text TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_query_log_question ON query_log(question)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_query_log_success ON query_log(success)
    """)
    conn.commit()
    conn.close()


def log_query(question, user, role, sql, answer, dashboard, retrieved_sources, success=True):
    conn = sqlite3.connect(DB_PATH)
    try:
        sources_json = json.dumps(retrieved_sources or [])
    except Exception:
        sources_json = json.dumps([])
    conn.execute(
        """INSERT INTO query_log
           (question, user, role, sql_generated, answer, had_dashboard, row_count, retrieved_sources, success, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            question,
            user,
            role,
            sql or "",
            answer or "",
            1 if dashboard else 0,
            dashboard.get("row_count", 0) if dashboard else 0,
            sources_json,
            1 if success else 0,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_failed_queries(limit=50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT question, sql_generated, answer FROM query_log WHERE success = 0 ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [{"question": r[0], "sql": r[1], "answer": r[2]} for r in rows]


def get_popular_queries(limit=20):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT question, sql_generated, COUNT(*) as cnt FROM query_log WHERE success = 1 GROUP BY question ORDER BY cnt DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [{"question": r[0], "sql": r[1], "count": r[2]} for r in rows]


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    failed = conn.execute("SELECT COUNT(*) FROM query_log WHERE success = 0").fetchone()[0]
    with_sql = conn.execute("SELECT COUNT(*) FROM query_log WHERE sql_generated != ''").fetchone()[0]
    with_dash = conn.execute("SELECT COUNT(*) FROM query_log WHERE had_dashboard = 1").fetchone()[0]
    conn.close()
    return {
        "total_queries": total,
        "failed_queries": failed,
        "sql_generated": with_sql,
        "dashboards_created": with_dash,
        "success_rate": round((total - failed) / total * 100, 1) if total else 0,
    }


def export_for_training(limit=200):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT question, sql_generated, answer FROM query_log WHERE success = 1 AND sql_generated != '' ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {"instruction": r[0], "sql": r[1], "response": r[2]}
        for r in rows
    ]


init_db()