import os
import pandas as pd
from gold import create_gold
from rag_engine import build_fts_index

if __name__ == "__main__":
    print("=== Rebuilding Enterprise Database ===\n")

    if os.path.exists("database/gold.db"):
        os.remove("database/gold.db")
        print("Removed old gold.db")

    create_gold()

    print("\n--- Building RAG FTS Index ---")
    build_fts_index()

    print("\n=== Database Ready ===")
