import sqlite3

conn = sqlite3.connect("database/gold.db")

tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()

schema = ""

for table in tables:

    table = table[0]

    cols = conn.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()

    schema += f"\nTable: {table}\n"

    for c in cols:
        schema += f"- {c[1]}\n"

conn.close()