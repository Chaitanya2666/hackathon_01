import sqlite3
import os
import re
from llm import ask_llm

_BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE, "database", "gold.db")
FTS_EMAILS_TABLE = "emails_fts"
FTS_TICKETS_TABLE = "tickets_fts"
MAX_CHUNKS = 8
_RAG_CONN_CACHE = None


def _get_rag_conn():
    global _RAG_CONN_CACHE
    if _RAG_CONN_CACHE is None:
        _RAG_CONN_CACHE = sqlite3.connect(DB_PATH, timeout=10)
        _RAG_CONN_CACHE.execute("PRAGMA temp_store=MEMORY")
        _RAG_CONN_CACHE.execute("PRAGMA mmap_size=268435456")
    return _RAG_CONN_CACHE

DOMAIN_KEYWORDS = {
    "hr": ["employee", "employees", "attrition", "salary", "department", "job", "hire", "resign", "overtime",
           "performance", "promotion", "income", "pay", "hr", "workforce", "staff", "personnel",
           "tenure", "turnover", "monthlyincome", "jobrole", "education", "gender"],
    "tickets": ["ticket", "tickets", "support", "issue", "bug", "priority", "resolution", "csat",
                "complaint", "helpdesk", "sla", "incident", "service", "open", "status",
                "unresolved", "reopen", "agent", "channel"],
    "maintenance": ["maintenance", "failure", "failures", "machine", "machines", "tool", "wear", "temperature",
                    "sensor", "equipment", "breakdown", "repair", "inspection", "rotational", "torque"],
    "sales": ["sales", "revenue", "order", "orders", "customer", "customers", "category", "region", "shipping",
              "product", "products", "segment", "profit", "purchase", "delivery", "ship"],
    "emails": ["email", "emails", "communication", "correspondence", "message", "messages", "sent", "meeting",
               "discuss", "discussion", "update", "request", "approval", "escalation", "report",
               "inbox", "mail", "correspond"],
}


def _chunk_text(text, max_words=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


def build_fts_index():
    conn = sqlite3.connect(DB_PATH, timeout=30)

    try:
        conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_EMAILS_TABLE} USING fts5(file, message, tokenize='porter unicode61')")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TICKETS_TABLE} USING fts5(ticket_id, issue_type, initial_message, resolution_summary, tokenize='porter unicode61')")
    except sqlite3.OperationalError:
        pass

    existing = conn.execute("SELECT COUNT(*) FROM emails_fts").fetchone()[0]
    if existing == 0:
        emails = conn.execute("SELECT file, message FROM emails LIMIT 10000").fetchall()
        for file, msg in emails:
            chunks = _chunk_text(msg)
            for i, chunk in enumerate(chunks):
                try:
                    conn.execute(
                        f"INSERT INTO {FTS_EMAILS_TABLE} (file, message) VALUES (?, ?)",
                        (f"{file}#chunk{i}", chunk)
                    )
                except sqlite3.OperationalError:
                    pass
        conn.commit()
        print(f"  Indexed {len(emails)} emails as {conn.execute('SELECT COUNT(*) FROM emails_fts').fetchone()[0]} chunks")

    existing_t = conn.execute(f"SELECT COUNT(*) FROM {FTS_TICKETS_TABLE}").fetchone()[0]
    if existing_t == 0:
        tickets = conn.execute(
            "SELECT ticket_id, issue_type, initial_message, resolution_summary FROM tickets"
        ).fetchall()
        for t in tickets:
            try:
                conn.execute(
                    f"INSERT INTO {FTS_TICKETS_TABLE} (ticket_id, issue_type, initial_message, resolution_summary) VALUES (?, ?, ?, ?)",
                    t
                )
            except sqlite3.OperationalError:
                pass
        conn.commit()
        print(f"  Indexed {len(tickets)} tickets")

    conn.close()


STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "can", "could", "shall", "should", "may", "might", "to", "of",
    "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "as", "until", "while", "it", "its", "what",
    "which", "who", "whom", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "him", "her", "his", "they", "them", "their", "please", "tell",
    "show", "list", "find", "get", "give", "need", "want", "like",
    "analyze", "analyse", "across", "between", "both", "each",
    "also", "then", "than", "very", "just",
}


def _extract_fts_terms(question):
    terms = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]{2,}\b', question.lower())
    terms = [t for t in terms if t not in STOP_WORDS and len(t) > 2]

    for domain, keywords in DOMAIN_KEYWORDS.items():
        q_lower = question.lower()
        matched = [kw for kw in keywords if kw in q_lower]
        if matched:
            terms.extend(matched)

    return list(set(terms))


def _fts_search_emails(conn, terms, max_chunks):
    results = []
    if not terms:
        return results
    try:
        fts_query = " OR ".join(terms)
        rows = conn.execute(
            f"SELECT file as doc_id, message as content FROM {FTS_EMAILS_TABLE} WHERE message MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, max_chunks)
        ).fetchall()
        for row in rows:
            results.append({"source": "email", "doc_id": row[0], "content": row[1]})
    except sqlite3.OperationalError:
        pass
    return results


def _fts_search_tickets(conn, terms, max_chunks):
    results = []
    if not terms:
        return results
    try:
        fts_query = " OR ".join(terms)
        for content_col in ["resolution_summary", "initial_message", "issue_type"]:
            try:
                rows = conn.execute(
                    f"SELECT ticket_id as doc_id, {content_col} as content FROM {FTS_TICKETS_TABLE} WHERE {content_col} MATCH ? ORDER BY rank LIMIT ?",
                    (fts_query, max_chunks)
                ).fetchall()
                for row in rows:
                    if row[1]:
                        results.append({"source": "ticket", "doc_id": row[0], "content": row[1]})
            except sqlite3.OperationalError:
                pass
    except sqlite3.OperationalError:
        pass
    return results


def _like_search_emails(conn, terms, max_chunks):
    results = []
    if not terms:
        return results
    for term in terms[:5]:
        try:
            pattern = f"%{term}%"
            rows = conn.execute(
                "SELECT file, message FROM emails WHERE message LIKE ? LIMIT ?",
                (pattern, max_chunks)
            ).fetchall()
            for row in rows:
                results.append({"source": "email", "doc_id": row[0], "content": row[1][:1000]})
        except sqlite3.OperationalError:
            pass
        if len(results) >= max_chunks:
            break
    return results


def _like_search_tickets(conn, terms, max_chunks):
    results = []
    if not terms:
        return results
    for term in terms[:5]:
        try:
            pattern = f"%{term}%"
            rows = conn.execute(
                "SELECT ticket_id, resolution_summary FROM tickets WHERE resolution_summary LIKE ? OR initial_message LIKE ? LIMIT ?",
                (pattern, pattern, max_chunks)
            ).fetchall()
            for row in rows:
                if row[1]:
                    results.append({"source": "ticket", "doc_id": row[0], "content": row[1]})
        except sqlite3.OperationalError:
            pass
        if len(results) >= max_chunks:
            break
    return results


def _like_search_hr(conn, terms, max_chunks):
    results = []
    if not terms:
        return results
    search_cols = ["department", "jobrole", "educationfield", "maritalstatus", "gender", "businesstravel", "overtime"]
    for term in terms[:5]:
        for col in search_cols:
            try:
                pattern = f"%{term}%"
                rows = conn.execute(
                    f"SELECT employeenumber, {col} FROM hr WHERE {col} LIKE ? LIMIT ?",
                    (pattern, max_chunks)
                ).fetchall()
                for row in rows:
                    results.append({"source": "hr", "doc_id": f"emp_{row[0]}", "content": f"{col}: {row[1]}"})
            except sqlite3.OperationalError:
                pass
        if len(results) >= max_chunks:
            break
    return results


def _like_search_maintenance(conn, terms, max_chunks):
    results = []
    if not terms:
        return results
    search_cols = ["type", "failure_type", "product_id"]
    for term in terms[:5]:
        for col in search_cols:
            try:
                pattern = f"%{term}%"
                rows = conn.execute(
                    f"SELECT udi, {col} FROM maintenance WHERE {col} LIKE ? LIMIT ?",
                    (pattern, max_chunks)
                ).fetchall()
                for row in rows:
                    results.append({"source": "maintenance", "doc_id": f"machine_{row[0]}", "content": f"{col}: {row[1]}"})
            except sqlite3.OperationalError:
                pass
        if len(results) >= max_chunks:
            break
    return results


def _like_search_sales(conn, terms, max_chunks):
    results = []
    if not terms:
        return results
    search_cols = ["category", "region", "segment", "product_name", "customer_name", "city", "state", "ship_mode"]
    for term in terms[:5]:
        for col in search_cols:
            try:
                pattern = f"%{term}%"
                rows = conn.execute(
                    f"SELECT order_id, {col} FROM sales WHERE {col} LIKE ? LIMIT ?",
                    (pattern, max_chunks)
                ).fetchall()
                for row in rows:
                    results.append({"source": "sales", "doc_id": row[0], "content": f"{col}: {row[1]}"})
            except sqlite3.OperationalError:
                pass
        if len(results) >= max_chunks:
            break
    return results


def retrieve_context(question, max_chunks=MAX_CHUNKS):
    conn = _get_rag_conn()
    results = []
    terms = _extract_fts_terms(question)

    if terms:
        results.extend(_fts_search_emails(conn, terms, max_chunks))
        results.extend(_fts_search_tickets(conn, terms, max_chunks))

    if len(results) < max_chunks:
        # Only LIKE-search small tables (HR=1.4k, maintenance=10k, sales=9.8k)
        # Emails (517k) and tickets (100k) use FTS which is already fast
        results.extend(_like_search_hr(conn, terms, max_chunks))
        results.extend(_like_search_maintenance(conn, terms, max_chunks))
        results.extend(_like_search_sales(conn, terms, max_chunks))

    seen = set()
    unique_results = []
    for r in results:
        key = (r["source"], r["doc_id"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    return unique_results[:max_chunks]


def is_document_query(question):
    doc_keywords = [
        "email", "emails", "mail", "message", "ticket", "tickets",
        "support", "issue", "complaint", "resolution", "resolved",
        "said", "mentioned", "wrote", "discuss", "conversation",
        "correspondence", "communication", "escalation", "sent",
    ]
    q = question.lower()
    return any(kw in q for kw in doc_keywords)


def generate_answer(question, context_chunks, sql_result=None):
    if not context_chunks and sql_result is None:
        return None

    if context_chunks:
        context_text = "\n\n".join(
            f"[{c['source']} #{c['doc_id']}]: {c['content'][:600]}"
            for c in context_chunks
        )
        prompt = f"""
You are an enterprise AI assistant with access to internal documents and database.

User Question: {question}

Relevant Documents:
{context_text}

{f'Database Results:\n{sql_result}\n' if sql_result is not None else ''}

Answer the question based on ALL available information above.
Cite which source(s) the information came from.
If the documents don't contain the answer, say so.
"""
    else:
        prompt = f"""
You are an enterprise AI assistant.

User Question: {question}

Database Results: {sql_result}

Give a clear answer based on the database results.
"""
    return ask_llm([
        {"role": "system", "content": "You are an expert enterprise AI assistant."},
        {"role": "user", "content": prompt}
    ])
