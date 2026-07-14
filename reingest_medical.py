# ============================================================
# RE-INGEST MEDICAL DB — improved chunking (Q&A kept together)
# Sybrid Internship 2026 | Muhammad Ashir Siddiqui
# ============================================================
# Rebuilds the medical vector store using the improved
# chunk_documents() (larger chunks so each answer travels with
# its question). Fixes the "sources shown but bot refuses"
# problem, which was caused by questions and answers landing in
# separate chunks.
#
# Writes to a NEW folder: db/medquad_chromadb_v2
# Your old db/medquad_chromadb is left untouched as a backup.
#
# Run (in venv):
#   python reingest_medical.py
#
# Takes ~20–40 min (embedding ~15k Q&A pairs). Let it finish.
# ============================================================

import os
import pandas as pd
from medquad_pipeline import (
    clean_dataset,
    load_documents,
    chunk_documents,
    embed_and_store,
    MEDQUAD_DB_PATH,
)

if __name__ == "__main__":
    print("=" * 60)
    print("RE-INGESTING MEDICAL DB (proper chunking, chunk_size=2000)")
    print(f"Target folder: {MEDQUAD_DB_PATH}")
    print("=" * 60)

    # Point this at your RAW original MedQuAD CSV.
    # Common names: 'medquad.csv' or 'docs/medquad.csv'. Adjust if different.
    RAW_CSV = "docs/medquad.csv"
    CLEANED_CSV = "docs/medquad_cleaned.csv"

    # 1. Load data. Prefer cleaning the raw file; fall back to the cleaned CSV.
    if os.path.exists(RAW_CSV):
        print(f"\nCleaning raw file: {RAW_CSV}")
        df = clean_dataset(RAW_CSV)
    else:
        print(f"\nRaw file not found at {RAW_CSV}; using cleaned CSV instead.")
        df = pd.read_csv(CLEANED_CSV)
    print(f"Rows: {len(df)}")

    # 2. Build documents (Question + Answer together)
    documents = load_documents(df)

    # 3. Chunk with the improved logic (chunk_size=2000, keeps Q&A together)
    chunks = chunk_documents(documents)

    # 4. Embed + store into the v2 folder
    vector_store = embed_and_store(chunks)

    print("\n" + "=" * 60)
    print("DONE. New DB built at:", MEDQUAD_DB_PATH)
    print("Total chunks:", vector_store._collection.count())
    print("=" * 60)
    print("\nNext: run the app (medquad_pipeline already points to v2).")
    print("Test 'What is osteoporosis?' — it should now answer.")
