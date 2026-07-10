import os
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent / "streamlit_demo" / "sample_hr.csv"

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def summarize_data(df: pd.DataFrame) -> str:
    return f"Loaded {len(df)} rows and {len(df.columns)} columns from the HR sample dataset."


def query_hr_data(question: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    q = question.lower()
    sql_preview = "SELECT * FROM sample_hr"

    if "attrition" in q and "department" in q:
        sql_preview = "SELECT Department, Attrition, COUNT(*) FROM sample_hr GROUP BY Department, Attrition"
        result = (
            df.groupby(["Department", "Attrition"], dropna=False)
            .size()
            .reset_index(name="Count")
            .sort_values(["Department", "Attrition"], ascending=[True, False])
        )
        return sql_preview, result

    if "attrition" in q:
        sql_preview = "SELECT Attrition, COUNT(*) FROM sample_hr GROUP BY Attrition"
        result = (
            df.groupby("Attrition", dropna=False)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )
        return sql_preview, result

    if "monthly income" in q or "income" in q or "salary" in q:
        sql_preview = "SELECT Department, ROUND(AVG(MonthlyIncome), 0) AS AvgMonthlyIncome FROM sample_hr GROUP BY Department"
        result = (
            df.groupby("Department", dropna=False)["MonthlyIncome"]
            .mean()
            .round(0)
            .reset_index(name="AvgMonthlyIncome")
            .sort_values("AvgMonthlyIncome", ascending=False)
        )
        return sql_preview, result

    if "gender" in q and "count" in q:
        sql_preview = "SELECT Gender, COUNT(*) FROM sample_hr GROUP BY Gender"
        result = (
            df.groupby("Gender", dropna=False)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )
        return sql_preview, result

    if "work life" in q or "work-life" in q or "work life balance" in q:
        sql_preview = "SELECT WorkLifeBalance, ROUND(AVG(MonthlyIncome),0) AS AvgMonthlyIncome, COUNT(*) AS Count FROM sample_hr GROUP BY WorkLifeBalance"
        result = (
            df.groupby("WorkLifeBalance", dropna=False)["MonthlyIncome"]
            .mean()
            .round(0)
            .reset_index(name="AvgMonthlyIncome")
        )
        result["Count"] = df.groupby("WorkLifeBalance").size().values
        return sql_preview, result

    if "over time" in q or "overtime" in q:
        sql_preview = "SELECT OverTime, COUNT(*) FROM sample_hr GROUP BY OverTime"
        result = (
            df.groupby("OverTime", dropna=False)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )
        return sql_preview, result

    sql_preview = "SELECT * FROM sample_hr LIMIT 20"
    return sql_preview, df.head(20)


def main():
    st.set_page_config(page_title="HR Data Demo", layout="wide")
    st.title("HR Data Demo with Streamlit")
    st.markdown(
        "This demo uses a small HR sample dataset so you can deploy it easily and still answer real questions."
    )

    df = load_data(DATA_PATH)
    st.info(summarize_data(df))

    examples = [
        "Show attrition counts by department",
        "What is the average monthly income by department?",
        "How many employees overtime?",
        "Count employees by gender",
        "Show the first 20 rows"
    ]

    question = st.text_input("Ask a question about HR data:", value=examples[0])

    if st.button("Run Query"):
        sql_preview, result_df = query_hr_data(question, df)
        st.subheader("Generated SQL Preview")
        st.code(sql_preview, language="sql")
        st.subheader("Query Result")
        st.dataframe(result_df)

        if len(result_df) > 0:
            st.subheader("Result Summary")
            st.write(result_df.describe(include="all"))

    with st.expander("Example questions"):
        for text in examples:
            st.write(f"- {text}")


if __name__ == "__main__":
    main()
