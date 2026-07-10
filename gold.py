import os
import sqlite3
import pandas as pd
from datetime import datetime, timezone

SILVER_DIR = "silver_layer"
GOLD_DIR = "gold_layer"
DB_PATH = "database/gold.db"


def load_silver(filename):
    path = os.path.join(SILVER_DIR, filename)
    df = pd.read_csv(path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def get_entity_resolution_map(hr_df, tickets_df, sales_df):
    cross_refs = []

    for _, row in hr_df.iterrows():
        cross_refs.append({
            "domain": "hr",
            "entity_type": "employee",
            "local_id": str(row["employeenumber"]),
            "name": f"{row.get('jobrole', '')} - Dept: {row.get('department', '')}",
        })

    for _, row in tickets_df.iterrows():
        cross_refs.append({
            "domain": "it_support",
            "entity_type": "ticket",
            "local_id": row["ticket_id"],
            "name": f"Issue: {row.get('issue_type', '')} - Priority: {row.get('priority', '')}",
        })

    for _, row in sales_df.iterrows():
        cross_refs.append({
            "domain": "sales",
            "entity_type": "order",
            "local_id": row["order_id"],
            "name": row.get("customer_name", ""),
        })

    return pd.DataFrame(cross_refs)


def create_master_view():
    conn = sqlite3.connect(DB_PATH)

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name != 'cross_reference'"
    ).fetchall()]

    union_parts = []
    for table in tables:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        col_list = ", ".join(cols)
        union_parts.append(f"SELECT '{table}' AS source_domain, {col_list} FROM {table}")

    if union_parts:
        conn.execute("DROP VIEW IF EXISTS master_data")
        union_sql = " UNION ALL ".join(union_parts)
        conn.execute(f"CREATE VIEW master_data AS {union_sql}")

    conn.commit()
    conn.close()


def create_gold():
    os.makedirs(GOLD_DIR, exist_ok=True)
    os.makedirs("database", exist_ok=True)

    print("--- Building Gold Layer ---")

    hr_df = load_silver("WA_Fn-UseC_-HR-Employee-Attrition_silver.csv")
    tickets_df = load_silver("synthetic_it_support_tickets_silver.csv")
    maintenance_df = load_silver("predictive_maintenance_silver.csv")
    sales_df = load_silver("train_silver.csv")
    emails_df = load_silver("emails_silver.csv")

    print(f"  HR: {len(hr_df)} rows")
    print(f"  Tickets: {len(tickets_df)} rows")
    print(f"  Maintenance: {len(maintenance_df)} rows")
    print(f"  Sales: {len(sales_df)} rows")
    print(f"  Emails: {len(emails_df)} rows")

    cross_refs = get_entity_resolution_map(hr_df, tickets_df, sales_df)

    conn = sqlite3.connect(DB_PATH)

    hr_df.to_sql("hr", conn, if_exists="replace", index=False)
    tickets_df.to_sql("tickets", conn, if_exists="replace", index=False)
    maintenance_df.to_sql("maintenance", conn, if_exists="replace", index=False)
    sales_df.to_sql("sales", conn, if_exists="replace", index=False)
    emails_df.to_sql("emails", conn, if_exists="replace", index=False)
    cross_refs.to_sql("cross_reference", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_hr_employee ON hr(employeenumber)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_id ON tickets(ticket_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_order ON sales(order_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_maintenance_udi ON maintenance(udi)")

    create_master_view()

    conn.close()

    master_path = os.path.join(GOLD_DIR, "master_unified_data.csv")
    merged = pd.concat([
        hr_df.assign(source_domain="hr"),
        tickets_df.assign(source_domain="it_support"),
        maintenance_df.assign(source_domain="maintenance"),
        sales_df.assign(source_domain="sales"),
        emails_df.assign(source_domain="emails"),
    ], ignore_index=True)
    merged.to_csv(master_path, index=False)

    print(f"  master_data view created with {len(merged)} total rows")
    print(f"  Cross-reference table: {len(cross_refs)} entities")
    print("--- Gold Layer Complete ---")


if __name__ == "__main__":
    create_gold()
