import sqlite3
conn = sqlite3.connect("database/gold.db")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", [t[0] for t in tables])

import pandas as pd
from dashboard_engine import build_dashboard, analyze_dataframe, generate_charts_matplotlib

for tname in [t[0] for t in tables]:
    try:
        df = pd.read_sql(f"SELECT * FROM {tname} LIMIT 20", conn)
        print(f"\n=== {tname} ===")
        print("Cols:", df.columns.tolist())
        analysis = analyze_dataframe(df)
        print("Numeric:", analysis["numeric_cols"][:3])
        charts = generate_charts_matplotlib(df, analysis)
        print("Charts:", len(charts))
        for c in charts[:3]:
            print(f"  {c['type']}: img_len={len(c['img'])}")
        break
    except Exception as e:
        print(f"  {tname}: {e}")