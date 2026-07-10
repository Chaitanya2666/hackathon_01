from llm import ask_llm
from nl2sql import generate_sql, execute_sql
from rag_engine import generate_answer
import traceback

print("=" * 60)
print(" Enterprise NL2SQL + RAG Assistant ")
print("=" * 60)

DB_KEYWORDS = [
    "employee",
    "employees",
    "department",
    "ticket",
    "tickets",
    "device",
    "devices",
    "asset",
    "assets",
    "finance",
    "salary",
    "maintenance",
    "sensor",
    "machine",
    "iot",
    "email",
    "count",
    "total",
    "how many",
    "average",
    "avg",
    "maximum",
    "minimum",
    "highest",
    "lowest",
    "top",
    "list",
    "show",
    "display",
    "find",
    "sales",
    "revenue",
    "cost"
]


def is_database_query(question):
    q = question.lower()

    for keyword in DB_KEYWORDS:
        if keyword in q:
            return True

    return False


while True:

    question = input("\nAsk a question (type 'exit' to quit): ").strip()

    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    if question == "":
        continue

    try:

        # -----------------------------
        # General Conversation
        # -----------------------------
        if not is_database_query(question):

            answer = ask_llm([
                {
                    "role": "system",
                    "content": """You are a friendly enterprise AI assistant.
Answer normally.
Do NOT generate SQL.
"""
                },
                {
                    "role": "user",
                    "content": question
                }
            ])

            print("\nAI Response:")
            print("-" * 60)
            print(answer)
            continue

        # -----------------------------
        # Database Query
        # -----------------------------
        sql = generate_sql(question)

        print("\nGenerated SQL:")
        print("-" * 60)
        print(sql)

        result = execute_sql(sql)

        print("\nQuery Result:")
        print("-" * 60)
        print(result)

        answer = generate_answer(question, result)

        print("\nAI Answer:")
        print("-" * 60)
        print(answer)

    except Exception:
        print("\nAn error occurred:\n")
        traceback.print_exc()