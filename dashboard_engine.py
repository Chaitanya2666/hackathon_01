import pandas as pd
import json
import numpy as np
import traceback
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from io import BytesIO
import base64
from llm import ask_llm

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#e0e0e0",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": "#e8e8e8",
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


def _to_img(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.4)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def _to_native(val):
    if isinstance(val, (np.generic,)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def analyze_dataframe(df):
    df = df.copy()
    numeric_cols, categorical_cols, date_cols = [], [], []
    for col in df.columns:
        try:
            pd.to_datetime(df[col], infer_datetime_format=True)
            date_cols.append(col)
            continue
        except Exception:
            pass
        converted = pd.to_numeric(df[col], errors="coerce")
        if not converted.isna().all():
            df[col] = converted.fillna(0).astype(float)
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_cols": date_cols,
        "row_count": _to_native(len(df)),
        "all_cols": df.columns.tolist(),
    }


def compute_kpis(df, analysis):
    kpis = []
    for col in analysis["numeric_cols"][:4]:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        is_count = any(kw in col.lower() for kw in ["count", "sum", "total", "number"])
        val = series.sum() if is_count else series.mean()
        kpis.append({
            "label": f"Total {col}" if is_count else f"Avg {col}",
            "value": _to_native(round(val, 2)),
        })
    kpis.append({"label": "Records", "value": _to_native(len(df))})
    return kpis


def generate_charts_matplotlib(df, analysis):
    charts = []
    num = analysis["numeric_cols"]
    cat = analysis["categorical_cols"]
    GREEN = "#10a37f"
    colors = ["#10a37f", "#ef4444", "#f59e0b", "#6366f1", "#a78bfa", "#fb923c", "#2dd4bf", "#f472b6", "#8e8ea0"]

    if not num and not cat:
        return charts

    for col in num[:3]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 2:
            continue
        try:
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            data = series.head(30)
            ax.bar(range(len(data)), data.values, color=GREEN, width=0.6, edgecolor="white", linewidth=0.5)
            ax.set_title(col[:30], fontweight=600, pad=10)
            ax.set_xlabel("Index", labelpad=6)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            fig.tight_layout()
            charts.append({"type": "bar", "title": f"{col} Distribution", "img": _to_img(fig)})
        except Exception:
            pass
        if len(charts) >= 6:
            return charts

    for col in num[:3]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 2:
            continue
        try:
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            data = series.head(40)
            ax.plot(data.values, color=GREEN, linewidth=2.5, marker="o", markersize=4)
            ax.fill_between(range(len(data)), data.values, alpha=0.08, color=GREEN)
            ax.set_title(f"{col[:25]} Trend", fontweight=600, pad=10)
            ax.set_xlabel("Index", fontsize=13, labelpad=6)
            fig.tight_layout()
            charts.append({"type": "line", "title": f"{col[:25]} Trend", "img": _to_img(fig)})
        except Exception:
            pass
        if len(charts) >= 6:
            return charts

    for col in cat[:3]:
        try:
            counts = df[col].value_counts().head(6)
            if len(counts) < 2:
                continue
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            ax.pie(
                counts.values, labels=None, autopct="%1.0f%%",
                colors=colors[:len(counts)], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            )
            ax.set_title(col[:30], fontweight=600, pad=10)
            ax.legend(
                [f"{l}" for l in counts.index],
                loc="lower center", bbox_to_anchor=(0.5, -0.38),
                ncol=min(3, len(counts)), fontsize=10,
            )
            fig.tight_layout()
            charts.append({"type": "pie", "title": f"{col[:25]} Breakdown", "img": _to_img(fig)})
        except Exception:
            pass
        if len(charts) >= 6:
            return charts

    for col in cat[:3]:
        try:
            counts = df[col].value_counts().head(6)
            if len(counts) < 2:
                continue
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            ax.barh(counts.index[::-1], counts.values[::-1], color=GREEN, height=0.6, edgecolor="white")
            ax.set_title(f"{col[:25]} (Horizontal)", fontweight=600, pad=10)
            ax.set_xlabel("Count", fontsize=13, labelpad=6)
            for i, v in enumerate(counts.values[::-1]):
                ax.text(v + max(counts.values) * 0.015, i, str(v), va="center", fontsize=11, color="#555")
            fig.tight_layout()
            charts.append({"type": "horizontalBar", "title": f"{col[:25]} Breakdown", "img": _to_img(fig)})
        except Exception:
            pass
        if len(charts) >= 6:
            return charts

    for col in num[:2]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 5:
            continue
        try:
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            ax.hist(series.values, bins=12, color=GREEN, edgecolor="white", linewidth=0.5, alpha=0.8)
            ax.set_title(f"{col[:25]} Histogram", fontweight=600, pad=10)
            ax.set_xlabel(col[:20], fontsize=13, labelpad=6)
            ax.set_ylabel("Frequency", fontsize=13, labelpad=6)
            fig.tight_layout()
            charts.append({"type": "histogram", "title": f"{col[:25]} Histogram", "img": _to_img(fig)})
        except Exception:
            pass
        if len(charts) >= 6:
            return charts

    return charts[:8]


def generate_insights(question, df, sql):
    try:
        analysis = analyze_dataframe(df)
        sample = df.head(10).to_string(index=False)
        prompt = f"""Analyze this data. Return valid JSON only.
Query: {question}
SQL: {sql}
Rows: {_to_native(len(df))}
Cols: {analysis['all_cols']}
Numeric: {analysis['numeric_cols']}
Categorical: {analysis['categorical_cols']}
Sample:
{sample}
Return JSON:
- "summary": 2 sentence summary
- "pain_points": list of 2-3 issues
- "insights": list of 2-3 observations
- "recommendations": list of 1-2 actions
ONLY valid JSON."""
        raw = ask_llm([
            {"role": "system", "content": "You are a data analyst. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ])
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        for key in result:
            if isinstance(result[key], list):
                result[key] = [_to_native(item) for item in result[key]]
            else:
                result[key] = _to_native(result[key])
        return result
    except Exception:
        return {
            "summary": f"Found {_to_native(len(df))} records across {_to_native(len(df.columns))} columns.",
            "pain_points": [],
            "insights": [f"{_to_native(len(df))} rows analyzed"],
            "recommendations": [],
        }


def build_dashboard(question, df, sql):
    try:
        if df is None or df.empty:
            return None
        if len(df) > 1000:
            df = df.head(1000)
        analysis = analyze_dataframe(df)
        kpis = compute_kpis(df, analysis)
        charts = generate_charts_matplotlib(df, analysis)
        insights = generate_insights(question, df, sql)
        return {
            "kpis": kpis,
            "charts": charts,
            "insights": insights,
            "columns": analysis["all_cols"],
            "row_count": analysis["row_count"],
        }
    except Exception:
        return None
