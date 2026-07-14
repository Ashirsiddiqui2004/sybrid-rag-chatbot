# ============================================================
# LATENCY PROFILER — RAG Pipeline Timing
# Sybrid Internship 2026 | Muhammad Ashir Siddiqui
# ============================================================
# Measures how long each stage of the pipeline takes, so we
# know where the bot spends its time. Runs an English query
# (no translation) and an Urdu query (with translation) so we
# can see exactly how much the translation layer costs.
#
# Stages timed:
#   1. Embedding model load (one-time startup cost)
#   2. Vector store connect
#   3. Retrieval (search)
#   4. LLM answer call
#   5. Translation calls (Urdu only)
#
# Run:  (venv) PS D:\sybrid_rag> python profile_latency.py
# ============================================================

import time
from medquad_pipeline import (
    load_vector_store,
    vector_search,
    ask_llm,
    translate_to_english,
    translate_to_urdu,
    detect_full_language,
)


def timed(label, fn, *args, **kwargs):
    """Runs fn, prints how long it took, returns its result."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"  {label:<32} {elapsed:6.2f} s")
    return result, elapsed


def profile_english(vector_store, question, k=8):
    """Profile a plain English query — no translation involved."""
    print(f"\n{'='*55}")
    print(f"ENGLISH QUERY: {question}")
    print("=" * 55)
    total = 0.0

    chunks, t = timed("1. Retrieval (vector search)", vector_search, vector_store, question, k=k)
    total += t

    # Build context (negligible, but included for honesty)
    start = time.perf_counter()
    context = ""
    for i, c in enumerate(chunks):
        context += f"Document {i+1}:\nContent: {c.page_content}\n\n"
    prompt = f"""Answer ONLY from the documents.
Documents:
{context}
Question: {question}
Answer:"""
    total += time.perf_counter() - start

    _, t = timed("2. LLM answer call", ask_llm, prompt)
    total += t

    print("-" * 55)
    print(f"  {'TOTAL (English)':<32} {total:6.2f} s")
    return total


def profile_urdu(vector_store, question, k=8):
    """Profile an Urdu query — includes BOTH translation calls."""
    print(f"\n{'='*55}")
    print(f"URDU QUERY: {question}")
    print("=" * 55)
    total = 0.0

    # Translation layer 1: Urdu question -> English
    search_query, t = timed("1. Translate question -> English", translate_to_english, question)
    total += t

    chunks, t = timed("2. Retrieval (vector search)", vector_search, vector_store, search_query, k=k)
    total += t

    start = time.perf_counter()
    context = ""
    for i, c in enumerate(chunks):
        context += f"Document {i+1}:\nContent: {c.page_content}\n\n"
    prompt = f"""Answer ONLY from the documents.
Documents:
{context}
Question: {search_query}
Answer:"""
    total += time.perf_counter() - start

    answer, t = timed("3. LLM answer call", ask_llm, prompt)
    total += t

    # Translation layer 2: English answer -> Urdu
    _, t = timed("4. Translate answer -> Urdu", translate_to_urdu, answer)
    total += t

    print("-" * 55)
    print(f"  {'TOTAL (Urdu)':<32} {total:6.2f} s")
    return total


if __name__ == "__main__":
    print("=" * 55)
    print("RAG PIPELINE LATENCY PROFILE")
    print("=" * 55)

    # One-time startup cost: embedding model load + DB connect
    vector_store, load_time = timed("Startup: load embeddings + connect DB",
                                    load_vector_store)

    # Profile an English query (no translation)
    eng_total = profile_english(vector_store, "What is tuberculosis?", k=8)

    # Profile an Urdu query (with translation — the suspected bottleneck)
    urdu_total = profile_urdu(vector_store, "ذیابیطس کیا ہے؟", k=8)

    # Summary
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  English query total : {eng_total:6.2f} s")
    print(f"  Urdu query total    : {urdu_total:6.2f} s")
    print(f"  Translation overhead : {urdu_total - eng_total:6.2f} s "
          f"({(urdu_total/eng_total - 1)*100:.0f}% slower)")
    print("\n  Interpretation: Urdu makes 3 LLM calls (translate in,")
    print("  answer, translate out) vs 1 for English. The extra two")
    print("  translation calls are the main latency cost.")