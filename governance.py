import os
import re
import sqlite3
import hashlib
import threading
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
DB_PATH = DATABASE_DIR / "governance.db"

# Create database directory automatically if it does not exist
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# Used only to prevent simultaneous write operations
_DB_WRITE_LOCK = threading.RLock()


def get_db_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """
    Create a fresh SQLite connection for the current thread.

    Never store this connection globally or reuse it between threads.
    """
    connection = sqlite3.connect(
        str(db_path),
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    # Wait instead of immediately failing when database is temporarily locked
    connection.execute("PRAGMA busy_timeout = 30000")

    return connection


# ============================================================
# PII MASKING
# ============================================================

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "name": r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\b",
}


class PIIMasker:
    def __init__(self, mask_char: str = "***") -> None:
        self.mask_char = mask_char
        self.patterns = {
            pattern_type: re.compile(pattern)
            for pattern_type, pattern in PII_PATTERNS.items()
        }

    def mask(self, text: Any) -> Any:
        if not isinstance(text, str):
            return text

        masked_text = text

        for pattern_type, pattern in self.patterns.items():
            masked_text = pattern.sub(
                f"[{pattern_type}_{self.mask_char}]",
                masked_text,
            )

        return masked_text

    def mask_dataframe(
        self,
        df: pd.DataFrame,
        sensitive_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        masked_df = df.copy()

        if sensitive_columns is None:
            sensitive_columns = [
                "customer_name",
                "customer_email",
                "ssn",
                "employeename",
                "employeenumber",
                "employee_name",
                "employee_number",
                "phone",
                "address",
            ]

        sensitive_terms = {
            column.lower()
            for column in sensitive_columns
        }

        for column in masked_df.columns:
            column_lower = column.lower()

            should_mask = any(
                sensitive_term in column_lower
                for sensitive_term in sensitive_terms
            )

            if should_mask and (
                pd.api.types.is_object_dtype(masked_df[column])
                or pd.api.types.is_string_dtype(masked_df[column])
            ):
                non_null_mask = masked_df[column].notna()
                series = masked_df[column].astype(str)
                for pattern_type, pattern in self.patterns.items():
                    series = series.str.replace(
                        pattern,
                        f"[{pattern_type}_{self.mask_char}]",
                        regex=True,
                    )
                masked_df.loc[non_null_mask, column] = series[non_null_mask]

        employee_number_columns = [
            "employeenumber",
            "employee_number",
        ]

        for column in employee_number_columns:
            if column in masked_df.columns:
                non_null_mask = masked_df[column].notna()
                masked_df.loc[non_null_mask, column] = (
                    masked_df.loc[non_null_mask, column]
                    .astype(str)
                    .apply(
                        lambda value: f"EMP_{hashlib.sha256(value.encode()).hexdigest()[:8]}"
                    )
                )

        return masked_df

    def mask_sql_result(self, df: pd.DataFrame) -> pd.DataFrame:
        masked_df = df.copy()

        sensitive_terms = [
            "employeenumber",
            "employee_number",
            "employee_name",
            "customer_name",
            "customer_id",
            "ssn",
            "phone",
            "email",
            "address",
        ]

        for column in masked_df.columns:
            column_lower = column.lower()

            if any(term in column_lower for term in sensitive_terms):
                masked_df[column] = masked_df[column].astype(object)
                non_null_mask = masked_df[column].notna()
                masked_df.loc[non_null_mask, column] = "[REDACTED]"

        return masked_df


# ============================================================
# AUDIT TRAIL
# ============================================================

class AuditTrail:
    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """
        A new connection is created for every operation.

        This prevents:
        SQLite objects created in a thread can only be used in that same thread.
        """
        return get_db_connection(self.db_path)

    def _init_db(self) -> None:
        with _DB_WRITE_LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        user TEXT,
                        role TEXT,
                        action TEXT,
                        query TEXT,
                        status TEXT,
                        ip_address TEXT
                    )
                    """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS data_quality (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        table_name TEXT,
                        row_count INTEGER,
                        null_counts TEXT,
                        duplicate_count INTEGER,
                        status TEXT
                    )
                    """
                )

                conn.commit()

    def log_query(
        self,
        user: str,
        role: str,
        action: str,
        query: str,
        status: str = "success",
        ip: str = "127.0.0.1",
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()

        with _DB_WRITE_LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_log (
                        timestamp,
                        user,
                        role,
                        action,
                        query,
                        status,
                        ip_address
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        user,
                        role,
                        action,
                        query,
                        status,
                        ip,
                    ),
                )

                conn.commit()

    def log_quality(
        self,
        table_name: str,
        row_count: int,
        null_counts: dict[str, int],
        duplicate_count: int,
        status: str,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()

        with _DB_WRITE_LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO data_quality (
                        timestamp,
                        table_name,
                        row_count,
                        null_counts,
                        duplicate_count,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        table_name,
                        row_count,
                        str(null_counts),
                        duplicate_count,
                        status,
                    ),
                )

                conn.commit()

    def get_audit_logs(self, limit: int = 100) -> pd.DataFrame:
        safe_limit = max(1, min(int(limit), 10_000))

        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT *
                FROM audit_log
                ORDER BY id DESC
                LIMIT ?
                """,
                conn,
                params=(safe_limit,),
            )

    def get_quality_logs(self, limit: int = 100) -> pd.DataFrame:
        safe_limit = max(1, min(int(limit), 10_000))

        with self._connect() as conn:
            return pd.read_sql_query(
                """
                SELECT *
                FROM data_quality
                ORDER BY id DESC
                LIMIT ?
                """,
                conn,
                params=(safe_limit,),
            )


# ============================================================
# ROLE-BASED ACCESS CONTROL
# ============================================================

class RBAC:
    ROLES = {
        "admin": {"query", "report", "manage", "audit"},
        "analyst": {"query", "report"},
        "viewer": {"query"},
    }

    def __init__(self, audit_trail: Optional[AuditTrail] = None) -> None:
        self.audit = audit_trail or AuditTrail()

    def check_access(
        self,
        user: str,
        role: str,
        action: str,
    ) -> bool:
        del user  # Reserved for future user-specific access rules

        permissions = self.ROLES.get(role)

        if permissions is None:
            return False

        return action in permissions

    def require(self, action: str):
        def decorator(function):
            @wraps(function)
            def wrapper(*args, **kwargs):
                user = kwargs.get("user", "anonymous")
                role = kwargs.get("role", "viewer")

                if not self.check_access(user, role, action):
                    self.audit.log_query(
                        user=user,
                        role=role,
                        action=action,
                        query="BLOCKED - insufficient permissions",
                        status="blocked",
                    )

                    raise PermissionError(
                        f"User '{user}' with role '{role}' "
                        f"cannot perform '{action}'"
                    )

                return function(*args, **kwargs)

            return wrapper

        return decorator


# ============================================================
# DATA QUALITY MONITOR
# ============================================================

class DataQualityMonitor:
    def __init__(self, audit_trail: Optional[AuditTrail] = None) -> None:
        self.audit = audit_trail or AuditTrail()

    @staticmethod
    def _validate_table_name(table_name: str) -> str:
        """
        Table names cannot be passed as SQLite query parameters,
        so validate them before adding them to SQL.
        """
        if not isinstance(table_name, str):
            raise TypeError("Table name must be a string.")

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            raise ValueError(
                f"Invalid table name: {table_name!r}"
            )

        return table_name

    def check_table(
        self,
        table_name: str,
        database_path: Path | str,
        limit: int = 10_000,
    ) -> dict[str, Any]:
        """
        Check a table by opening a fresh connection in the current thread.

        Important:
        Do not pass an existing sqlite3.Connection object here.
        Pass the database path instead.

        Example:
            quality.check_table(
                table_name="employees",
                database_path="database/gold.db"
            )
        """
        safe_table_name = self._validate_table_name(table_name)
        safe_limit = max(1, min(int(limit), 1_000_000))

        connection = get_db_connection(database_path)

        try:
            query = (
                f'SELECT * FROM "{safe_table_name}" '
                f"LIMIT {safe_limit}"
            )

            df = pd.read_sql_query(query, connection)

        finally:
            connection.close()

        null_counts_series = df.isnull().sum()

        null_counts = {
            column: int(count)
            for column, count in null_counts_series.items()
            if count > 0
        }

        duplicate_count = int(df.duplicated().sum())

        if duplicate_count > 0:
            status = "degraded"
        elif null_counts:
            status = "warning"
        else:
            status = "pass"

        result = {
            "table": safe_table_name,
            "rows": int(len(df)),
            "nulls": null_counts,
            "duplicates": duplicate_count,
            "status": status,
        }

        self.audit.log_quality(
            table_name=safe_table_name,
            row_count=len(df),
            null_counts=null_counts,
            duplicate_count=duplicate_count,
            status=status,
        )

        return result


# ============================================================
# SHARED SERVICES
# ============================================================

# These objects do not store any SQLite connection.
# Every database operation creates a connection in its current thread.

masker = PIIMasker()

audit = AuditTrail()

rbac = RBAC(
    audit_trail=audit
)

quality = DataQualityMonitor(
    audit_trail=audit
)