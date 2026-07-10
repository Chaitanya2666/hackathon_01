import os
import time
import json
import hashlib
import pandas as pd
import requests
from datetime import datetime, timezone
from abc import ABC, abstractmethod


class AuditLogger:
    def __init__(self, db_path="database/governance.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline TEXT,
                source TEXT,
                status TEXT,
                rows_ingested INTEGER,
                rows_rejected INTEGER,
                error TEXT,
                started_at TEXT,
                finished_at TEXT,
                checksum TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log(self, pipeline, source, status, rows_ingested=0, rows_rejected=0, error=None, checksum=None):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO ingestion_audit (pipeline, source, status, rows_ingested, rows_rejected, error, started_at, finished_at, checksum) VALUES (?,?,?,?,?,?,?,?,?)",
            (pipeline, source, status, rows_ingested, rows_rejected, error,
             datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat(), checksum)
        )
        conn.commit()
        conn.close()


class ValidationRule:
    def __init__(self, name, func, severity="error"):
        self.name = name
        self.func = func
        self.severity = severity

    def validate(self, df):
        passed, message = self.func(df)
        return {"rule": self.name, "passed": passed, "severity": self.severity, "message": message}


class BaseConnector(ABC):
    def __init__(self, source_name):
        self.source_name = source_name
        self.validation_rules = []

    def add_rule(self, rule):
        self.validation_rules.append(rule)

    def validate(self, df):
        results = []
        for rule in self.validation_rules:
            results.append(rule.validate(df))
        errors = [r for r in results if not r["passed"] and r["severity"] == "error"]
        warnings = [r for r in results if not r["passed"] and r["severity"] == "warning"]
        return results, errors, warnings

    @abstractmethod
    def extract(self, **kwargs):
        pass


class FileConnector(BaseConnector):
    def __init__(self, source_name, file_path, file_format="csv"):
        super().__init__(source_name)
        self.file_path = file_path
        self.file_format = file_format

    def _compute_checksum(self):
        hasher = hashlib.md5()
        with open(self.file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def extract(self, **kwargs):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Source file not found: {self.file_path}")

        checksum = self._compute_checksum()

        if self.file_format == "csv":
            df = pd.read_csv(self.file_path, **kwargs)
        elif self.file_format == "json":
            df = pd.read_json(self.file_path, **kwargs)
        elif self.file_format == "parquet":
            df = pd.read_parquet(self.file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {self.file_format}")

        return df, checksum


class ApiConnector(BaseConnector):
    def __init__(self, source_name, api_url, api_key=None, headers=None):
        super().__init__(source_name)
        self.api_url = api_url
        self.api_key = api_key
        self.custom_headers = headers or {}
        self.timeout = 60
        self.max_retries = 3

    def extract(self, **kwargs):
        headers = self.custom_headers.copy()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(self.api_url, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    return pd.DataFrame(data), hashlib.md5(str(data).encode()).hexdigest()
                if isinstance(data, dict) and "data" in data:
                    return pd.DataFrame(data["data"]), hashlib.md5(str(data).encode()).hexdigest()
                return pd.DataFrame([data]), hashlib.md5(str(data).encode()).hexdigest()
            except requests.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        raise ConnectionError(f"API extraction failed after {self.max_retries} retries: {last_error}")


class IngestionPipeline:
    def __init__(self, name="default"):
        self.name = name
        self.connectors = []
        self.audit = AuditLogger()
        self.bronze_dir = "bronze_layer"

    def add_connector(self, connector):
        self.connectors.append(connector)

    def run(self):
        os.makedirs(self.bronze_dir, exist_ok=True)

        for connector in self.connectors:
            print(f"\n--- Ingesting: {connector.source_name} ---")
            try:
                df, checksum = connector.extract()

                results, errors, warnings = connector.validate(df)

                for w in warnings:
                    print(f"  WARNING [{w['rule']}]: {w['message']}")

                if errors:
                    for e in errors:
                        print(f"  ERROR [{e['rule']}]: {e['message']}")
                    self.audit.log(self.name, connector.source_name, "failed",
                                   error="; ".join(e["message"] for e in errors))
                    continue

                filename = connector.source_name.lower().replace(" ", "_") + ".csv"
                filepath = os.path.join(self.bronze_dir, filename)
                df.to_csv(filepath, index=False)

                self.audit.log(self.name, connector.source_name, "success",
                               rows_ingested=len(df), checksum=checksum)
                print(f"  Ingested {len(df)} rows -> {filepath}")

            except Exception as ex:
                print(f"  FAILED: {ex}")
                self.audit.log(self.name, connector.source_name, "failed", error=str(ex))


def default_validation_rules():
    return [
        ValidationRule("not_empty", lambda df: (len(df) > 0, "Dataset is empty")),
        ValidationRule("has_columns", lambda df: (len(df.columns) > 0, "No columns found")),
    ]


def build_pipeline():
    pipeline = IngestionPipeline("enterprise_ingestion")

    pipeline.add_connector(FileConnector("hr_employees", "bronze_layer/WA_Fn-UseC_-HR-Employee-Attrition.csv"))
    pipeline.add_connector(FileConnector("it_tickets", "bronze_layer/synthetic_it_support_tickets.csv"))
    pipeline.add_connector(FileConnector("iot_maintenance", "bronze_layer/predictive_maintenance.csv"))
    pipeline.add_connector(FileConnector("sales_orders", "bronze_layer/train.csv"))

    for connector in pipeline.connectors:
        for rule in default_validation_rules():
            connector.add_rule(rule)

    return pipeline


if __name__ == "__main__":
    p = build_pipeline()
    p.run()
    print("\nIngestion complete.")
