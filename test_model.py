from llm import ask_llm

answer = ask_llm([
    {
        "role": "user",
        "content": "Say Hello"
    }
])

print("\nAI Response:")
print(answer)
import sqlite3
import pandas as pd

conn = sqlite3.connect("database/gold.db")

df = pd.read_sql("SELECT * FROM master_data LIMIT 5", conn)

print("\nColumns:")
print(df.columns.tolist())

print("\nData:")
print(df.head())

conn.close()
import pandas as pd

print("Emails")
print(pd.read_csv("silver_layer/emails_silver.csv").columns.tolist())

print()

print("Tickets")
print(pd.read_csv("silver_layer/synthetic_it_support_tickets_silver.csv").columns.tolist())

print()

print("Maintenance")
print(pd.read_csv("silver_layer/predictive_maintenance_silver.csv").columns.tolist())