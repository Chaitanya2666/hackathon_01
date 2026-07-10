import functools
import sqlite3
import os
import re
import pandas as pd
from llm import ask_llm

_BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE, "database", "gold.db")

TABLE_DOMAINS = {
    "hr": "HR/employee data: attrition, department, salary, job roles, overtime, performance",
    "tickets": "IT support tickets: priority, status, resolution time, region, issue type, CSAT",
    "maintenance": "Machine maintenance: failure types, tool wear, temperature, rotational speed",
    "sales": "Sales orders: revenue, region, category, customer segment, shipping",
    "emails": "Email communications: messages, correspondence, subject, content",
}

DOMAIN_KEYWORDS = {
    "hr": ["attrition", "employee", "employees", "department", "salary", "income", "overtime",
           "job", "hire", "resign", "performance", "promotion", "hr", "workforce",
           "staff", "turnover", "pay", "monthlyincome", "tenure", "working years",
           "education", "gender", "marital", "jobrole", "job role"],
    "tickets": ["ticket", "tickets", "support", "issue", "bug", "priority", "resolution",
                "csat", "complaint", "helpdesk", "sla", "incident", "open ticket",
                "open tickets", "status", "it support", "unresolved", "reopen",
                "agent", "channel", "platform", "sentiment"],
    "maintenance": ["maintenance", "failure", "failures", "machine", "machines", "tool", "wear",
                    "temperature", "sensor", "equipment", "breakdown", "repair",
                    "rotational", "torque", "predictive", "defect"],
    "sales": ["sales", "revenue", "order", "orders", "customer", "customers", "category",
              "region", "shipping", "product", "products", "segment", "profit",
              "purchase", "discount", "ship", "delivery"],
    "emails": ["email", "emails", "communication", "correspondence", "message", "messages", "meeting",
               "discuss", "discussion", "update", "request", "approval", "escalation",
               "report", "inbox", "sent", "mail", "correspond"],
}

FALLBACK_QUERIES = {
    "emails": {
        "email_count":
            "SELECT COUNT(*) as total_emails FROM emails",
    },
    "hr": {
        "attrition_by_department":
            "SELECT department, COUNT(*) as total_employees, "
            "SUM(CASE WHEN attrition='Yes' THEN 1 ELSE 0 END) as attrition_count, "
            "ROUND(100.0 * SUM(CASE WHEN attrition='Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) as attrition_rate "
            "FROM hr GROUP BY department ORDER BY attrition_rate DESC",
        "overtime_by_department":
            "SELECT department, overtime, COUNT(*) as count FROM hr GROUP BY department, overtime",
        "salary_by_department":
            "SELECT department, ROUND(AVG(monthlyincome), 0) as avg_monthly_income, "
            "ROUND(AVG(yearsatcompany), 1) as avg_tenure_years, "
            "ROUND(AVG(totalworkingyears), 1) as avg_working_years "
            "FROM hr GROUP BY department",
        "attrition_summary":
            "SELECT attrition, COUNT(*) as count, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM hr), 1) as pct "
            "FROM hr GROUP BY attrition",
        "salary_by_jobrole":
            "SELECT jobrole, ROUND(AVG(monthlyincome), 0) as avg_monthly_income, "
            "SUM(CASE WHEN attrition='Yes' THEN 1 ELSE 0 END) as attrition_count, "
            "ROUND(100.0 * SUM(CASE WHEN attrition='Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) as attrition_rate "
            "FROM hr GROUP BY jobrole ORDER BY avg_monthly_income DESC",
    },
    "tickets": {
        "by_priority_status":
            "SELECT priority, status, COUNT(*) as count FROM tickets GROUP BY priority, status",
        "by_region":
            "SELECT region, COUNT(*) as ticket_count, "
            "ROUND(AVG(resolution_time_hours), 1) as avg_resolution_hours "
            "FROM tickets GROUP BY region",
        "resolution_by_priority":
            "SELECT priority, COUNT(*) as count, "
            "ROUND(AVG(resolution_time_hours), 1) as avg_resolution_hours, "
            "ROUND(AVG(csat_score), 1) as avg_csat "
            "FROM tickets GROUP BY priority",
        "open_tickets":
            "SELECT priority, COUNT(*) as open_count FROM tickets WHERE status='open' OR status='in_progress' "
            "GROUP BY priority ORDER BY open_count DESC",
        "by_issue_type":
            "SELECT issue_type, COUNT(*) as count, "
            "ROUND(AVG(resolution_time_hours), 1) as avg_resolution_hours "
            "FROM tickets GROUP BY issue_type ORDER BY count DESC LIMIT 10",
        "by_channel":
            "SELECT channel, COUNT(*) as count, ROUND(AVG(csat_score), 1) as avg_csat "
            "FROM tickets GROUP BY channel ORDER BY count DESC",
    },
    "maintenance": {
        "failure_by_type":
            "SELECT failure_type, COUNT(*) as count FROM maintenance WHERE target=1 GROUP BY failure_type ORDER BY count DESC",
        "failure_rate_by_machine":
            "SELECT type, COUNT(*) as total, SUM(target) as failures, "
            "ROUND(100.0 * SUM(target) / COUNT(*), 1) as failure_rate_pct "
            "FROM maintenance GROUP BY type",
        "failure_summary":
            "SELECT target, COUNT(*) as count FROM maintenance GROUP BY target",
        "avg_metrics_by_type":
            "SELECT type, ROUND(AVG(air_temperature_[k]), 1) as avg_air_temp, "
            "ROUND(AVG(rotational_speed_[rpm]), 0) as avg_rotational_speed, "
            "ROUND(AVG(torque_[nm]), 1) as avg_torque, "
            "ROUND(AVG(tool_wear_[min]), 0) as avg_tool_wear "
            "FROM maintenance GROUP BY type",
    },
    "sales": {
        "revenue_by_region":
            "SELECT region, SUM(sales) as total_revenue, COUNT(*) as order_count, "
            "ROUND(AVG(sales), 2) as avg_order_value "
            "FROM sales GROUP BY region ORDER BY total_revenue DESC",
        "revenue_by_category":
            "SELECT category, SUM(sales) as total_revenue, COUNT(*) as order_count "
            "FROM sales GROUP BY category ORDER BY total_revenue DESC",
        "revenue_by_segment":
            "SELECT segment, SUM(sales) as total_revenue, COUNT(*) as order_count "
            "FROM sales GROUP BY segment",
        "top_customers":
            "SELECT customer_name, SUM(sales) as total_spent, COUNT(*) as order_count "
            "FROM sales GROUP BY customer_name ORDER BY total_spent DESC LIMIT 10",
        "revenue_by_region_category":
            "SELECT region, category, SUM(sales) as total_revenue "
            "FROM sales GROUP BY region, category ORDER BY region, total_revenue DESC",
    },
}


def _normalize_tables(tables):
    if tables is None:
        return None
    if isinstance(tables, str):
        tables = [tables]
    normalized = tuple(sorted({t.strip() for t in tables if isinstance(t, str) and t.strip()}))
    return normalized or None


@functools.lru_cache(maxsize=32)
def _get_schema_cached(include_cross_ref: bool, tables: tuple[str, ...] | None = None) -> str:
    conn = get_conn()
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%_fts'"
    rows = []
    if tables:
        placeholders = ",".join("?" for _ in tables)
        rows = conn.execute(f"{query} AND name IN ({placeholders})", tables).fetchall()
    else:
        rows = conn.execute(query).fetchall()

    schema_lines = []
    selected_names = [row[0] for row in rows]
    for table in rows:
        tname = table[0]
        cols = conn.execute(f"PRAGMA table_info([{tname}])").fetchall()
        col_defs = "\n".join(f"  - {c[1]} ({c[2]})" for c in cols)
        schema_lines.append(f"Table: {tname}\n{col_defs}")

    if include_cross_ref and "cross_reference" not in selected_names:
        cols = conn.execute("PRAGMA table_info([cross_reference])").fetchall()
        if cols:
            col_defs = "\n".join(f"  - {c[1]} ({c[2]})" for c in cols)
            schema_lines.append(f"Table: cross_reference\n{col_defs}")

    conn.close()
    return "\n\n".join(schema_lines)


def get_schema(include_cross_ref=True, tables=None):
    tables = _normalize_tables(tables)
    return _get_schema_cached(bool(include_cross_ref), tables)


@functools.lru_cache(maxsize=32)
def _get_cross_reference_summary_cached(domains: tuple[str, ...] | None = None) -> str:
    conn = get_conn()
    query = "SELECT domain, entity_type, COUNT(*) as count FROM cross_reference"
    params = ()
    if domains:
        placeholders = ",".join("?" for _ in domains)
        query = f"{query} WHERE domain IN ({placeholders})"
        params = domains
    query = f"{query} GROUP BY domain, entity_type"
    summary = conn.execute(query, params).fetchall()

    samples = {}
    for domain, etype, _ in summary:
        sample_query = "SELECT local_id, name FROM cross_reference WHERE domain=? AND entity_type=? LIMIT 3"
        rows = conn.execute(sample_query, (domain, etype)).fetchall()
        samples[f"{domain}/{etype}"] = rows

    conn.close()
    lines = ["cross_reference table — maps entity IDs to readable names across domains:"]
    for domain, etype, count in summary:
        sample_str = "; ".join(f"{sid}: {sn}" for sid, sn in samples.get(f"{domain}/{etype}", []))
        lines.append(f"  - {domain}.{etype}: {count} entries (e.g. {sample_str})")
    return "\n".join(lines)


def get_cross_reference_summary(domains=None):
    if domains is None:
        domains = None
    else:
        domains = tuple(sorted(set(domains)))
    return _get_cross_reference_summary_cached(domains)


def get_table_schema(table_name):
    conn = get_conn()
    cols = conn.execute(f"PRAGMA table_info([{table_name}])").fetchall()
    conn.close()
    return "\n".join(f"  - {c[1]} ({c[2]})" for c in cols)


def detect_relevant_domains(question):
    q = question.lower()
    relevant = []
    score_map = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            score_map[domain] = score
            relevant.append(domain)
    relevant.sort(key=lambda d: score_map[d], reverse=True)
    return relevant


def run_fallback_queries(domains):
    all_results = []
    all_errors = []
    for domain in domains:
        if domain not in FALLBACK_QUERIES:
            continue
        for qname, sql in FALLBACK_QUERIES[domain].items():
            df, err = execute_sql(sql)
            if err:
                all_errors.append(f"[{domain}/{qname}] {err}")
            elif df is not None and not df.empty:
                all_results.append((f"-- [{domain}] {qname}", df))
    return all_results, all_errors


TABLE_NAMES = set(TABLE_DOMAINS.keys())


def _is_trivial_query(sql):
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        return True
    if " FROM " not in stripped:
        return True
    if "CANNOT_ANSWER" in stripped or "NO_DATA" in stripped or "NOT_AVAILABLE" in stripped:
        return True
    has_real_table = any(tbl in stripped.lower() for tbl in TABLE_NAMES)
    if not has_real_table:
        return True
    return False


@functools.lru_cache(maxsize=128)
def generate_sql(question):
    relevant_domains = detect_relevant_domains(question)
    tables = relevant_domains if relevant_domains else None
    schema_text = get_schema(include_cross_ref=True, tables=tables)
    tables_info = "\n".join(f"  - {tbl}: {desc}" for tbl, desc in TABLE_DOMAINS.items())
    cross_ref_info = get_cross_reference_summary(domains=relevant_domains if relevant_domains else None)

    prompt = f"""You are an expert SQLite SQL generator for an enterprise database.

Available Tables:
{tables_info}

Relevant Schema (tables most likely needed for this question):
{schema_text}

{cross_ref_info}

TASK: For the question below, generate SQL queries that ANSWER the question.
You can JOIN tables using cross_reference to link entities across domains.
Use aggregation (GROUP BY/COUNT/AVG/SUM) for summaries.

RULES:
- Return queries separated by semicolons (;)
- Use JOINs to connect data across tables via cross_reference when relevant
- Every query MUST have a FROM clause referencing a real table
- NEVER return placeholder queries like SELECT '...'
- Use ONLY columns that exist in the schema
- Use SQLite-compatible syntax
- Focus on answering the specific question with cross-domain insights

EXAMPLES:
- "Show employees by department" -> SELECT department, COUNT(*) as count FROM hr GROUP BY department
- "How many high priority tickets?" -> SELECT priority, COUNT(*) as count FROM tickets WHERE priority = 'high' AND status = 'open'
- "Total sales by region" -> SELECT region, SUM(sales) as total_revenue FROM sales GROUP BY region

User Question: {question}"""

    response = ask_llm([
        {"role": "system", "content": "You generate SQLite queries. You can JOIN across tables using the cross_reference table to link entities. Always read from real tables. Never return placeholder queries."},
        {"role": "user", "content": prompt}
    ])

    sql = response.replace("```sql", "").replace("```", "").replace("```SQL", "").strip()

    if sql.upper().startswith("CANNOT_ANSWER") or sql.upper() == "NOT_RELEVANT":
        return None, "LLM could not generate queries."

    statements = [s.strip() for s in sql.replace("\n", " ").split(";") if s.strip()]
    real_queries = [s for s in statements if not _is_trivial_query(s)]

    if not real_queries:
        return None, "LLM returned only trivial queries."

    return "; ".join(real_queries), None


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=268435456")
    return conn


def _has_limit(sql: str) -> bool:
    return bool(re.search(r"\bLIMIT\b", sql, flags=re.IGNORECASE))


def _limit_query(sql: str, default_limit: int = 1000) -> str:
    cleaned = sql.strip().rstrip(";")
    if _has_limit(cleaned):
        return cleaned
    if cleaned.lower().startswith("with "):
        return f"{cleaned} LIMIT {default_limit}"
    return f"SELECT * FROM ({cleaned}) AS _subquery LIMIT {default_limit}"


def execute_sql(sql):
    conn = get_conn()
    try:
        query = _limit_query(sql)
        df = pd.read_sql_query(query, conn)
        return df, None
    except Exception as e:
        return None, f"SQL Execution Error: {e}"
    finally:
        conn.close()


def clean_sql_text(text):
    text = text.replace("```sql", "").replace("```", "").replace("```SQL", "")
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("SELECT") or line.upper().startswith("WITH") or \
           line.upper().startswith("--") or line == "" or \
           line.upper().startswith("ORDER") or line.upper().startswith("GROUP") or \
           line.upper().startswith("HAVING") or line.upper().startswith("LIMIT") or \
           line.upper().startswith("UNION") or line.upper().startswith("FROM") or \
           line.upper().startswith("WHERE") or line.upper().startswith("INNER") or \
           line.upper().startswith("LEFT") or line.upper().startswith("RIGHT") or \
           line.upper().startswith("CROSS") or line.upper().startswith("JOIN") or \
           line.upper().startswith("NATURAL"):
            lines.append(line)
    return " ".join(lines)


def split_and_execute(sql_text):
    if not sql_text or sql_text.strip() == "":
        return [], ["No SQL to execute."]

    cleaned = clean_sql_text(sql_text)

    parts = []
    current = ""
    depth = 0
    for ch in cleaned:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
        elif ch == ';' and depth == 0:
            if current.strip():
                parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())

    statements = [p for p in parts if p and not _is_trivial_query(p)]

    results = []
    errors = []
    seen = set()
    for stmt in statements:
        stmt = stmt.strip()
        if not stmt:
            continue
        stmt_lower = stmt.lower()
        if stmt_lower in seen or stmt_lower.startswith("not_relevant"):
            continue
        seen.add(stmt_lower)
        df, err = execute_sql(stmt)
        if err:
            errors.append(f"Query failed: {stmt[:80]}... -> {err}")
        elif df is not None and not df.empty:
            results.append((stmt, df))
    return results, errors


def query_all_domains(question):
    relevant = detect_relevant_domains(question)
    all_domains = relevant if relevant else list(TABLE_DOMAINS.keys())
    fallback_results, fallback_errors = run_fallback_queries(all_domains)

    llm_results = []
    llm_errors = []
    sql_generated, err = generate_sql(question)
    if not err:
        llm_results, llm_errors = split_and_execute(sql_generated)

    seen_queries = set()
    merged_results = []

    for label, df in fallback_results:
        seen_queries.add(label)
        merged_results.append((label, df))

    for stmt, df in llm_results:
        key = stmt[:100]
        if key not in seen_queries:
            seen_queries.add(key)
            merged_results.append((stmt, df))

    all_errors = fallback_errors + llm_errors
    return merged_results, all_errors, sql_generated
