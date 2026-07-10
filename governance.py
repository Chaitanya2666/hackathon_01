import sqlite3
import os
import re
import hashlib
from datetime import datetime, timezone
from functools import wraps

_BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE, "database", "governance.db")
PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "name": r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\b',
}


class PIIMasker:
    def __init__(self, mask_char="***"):
        self.mask_char = mask_char
        self.patterns = {k: re.compile(v) for k, v in PII_PATTERNS.items()}

    def mask(self, text):
        if not isinstance(text, str):
            return text
        masked = text
        for ptype, pattern in self.patterns.items():
            masked = pattern.sub(f"[{ptype}_{self.mask_char}]", masked)
        return masked

    def mask_dataframe(self, df, sensitive_columns=None):
        if sensitive_columns is None:
            sensitive_columns = ["customer_name", "customer_email", "ssn",
                                 "employeename", "employeenumber"]
        df = df.copy()
        for col in df.columns:
            col_lower = col.lower()
            if any(s in col_lower for s in ["name", "email", "phone", "ssn", "address"]):
                if col in df.columns and df[col].dtype == object:
                    df[col] = df[col].apply(lambda x: self.mask(str(x)) if pd.notna(x) else x)
        if "employeenumber" in df.columns:
            df["employeenumber"] = df["employeenumber"].apply(
                lambda x: f"EMP_{hashlib.md5(str(x).encode()).hexdigest()[:8]}" if pd.notna(x) else x
            )
        return df

    def mask_sql_result(self, df):
        sensitive_cols = ["employeenumber", "employee_number", "customer_name",
                          "customer_id", "ssn", "phone", "email"]
        for col in df.columns:
            if any(s in col.lower() for s in sensitive_cols):
                df[col] = df[col].apply(
                    lambda x: f"[REDACTED]" if pd.notna(x) else x
                )
        return df


class AuditTrail:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user TEXT,
                role TEXT,
                action TEXT,
                query TEXT,
                status TEXT,
                ip_address TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                table_name TEXT,
                row_count INTEGER,
                null_counts TEXT,
                duplicate_count INTEGER,
                status TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_query(self, user, role, action, query, status="success", ip="127.0.0.1"):
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO audit_log (timestamp, user, role, action, query, status, ip_address) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), user, role, action, query, status, ip)
        )
        conn.commit()
        conn.close()

    def log_quality(self, table_name, row_count, null_counts, duplicate_count, status):
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO data_quality (timestamp, table_name, row_count, null_counts, duplicate_count, status) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), table_name, row_count, str(null_counts), duplicate_count, status)
        )
        conn.commit()
        conn.close()


class RBAC:
    ROLES = {
        "admin": {"query", "report", "manage", "audit"},
        "analyst": {"query", "report"},
        "viewer": {"query"},
    }

    def __init__(self):
        self.audit = AuditTrail()

    def check_access(self, user, role, action):
        if role not in self.ROLES:
            return False
        return action in self.ROLES[role]

    def require(self, action):
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                user = kwargs.get("user", "anonymous")
                role = kwargs.get("role", "viewer")
                if not self.check_access(user, role, action):
                    self.audit.log_query(user, role, action, "BLOCKED - insufficient permissions")
                    raise PermissionError(f"User '{user}' with role '{role}' cannot perform '{action}'")
                return f(*args, **kwargs)
            return wrapper
        return decorator


class DataQualityMonitor:
    def __init__(self):
        self.audit = AuditTrail()

    def check_table(self, conn, table_name):
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 10000", conn)

        null_counts = df.isnull().sum().to_dict()
        null_counts = {k: int(v) for k, v in null_counts.items() if v > 0}

        duplicate_count = int(df.duplicated().sum())

        status = "pass"
        if null_counts:
            status = "warning"
        if duplicate_count > 0:
            status = "degraded"

        self.audit.log_quality(table_name, len(df), null_counts, duplicate_count, status)
        return {"table": table_name, "rows": len(df), "nulls": null_counts, "duplicates": duplicate_count, "status": status}


import pandas as pd

masker = PIIMasker()
audit = AuditTrail()
rbac = RBAC()
quality = DataQualityMonitor()
