# Dual-Domain Bilingual RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from two separate knowledge bases — **medical Q&A** and **ecommerce products** — in **English, Urdu, and Roman Urdu**. Every answer is grounded in retrieved documents, not the language model's training data.

Built during the Sybrid internship (2026) by Muhammad Ashir Siddiqui.

---

## What it does

Ask a question in the chat interface and the bot:

1. Detects the language (English / Urdu script / Roman Urdu)
2. Retrieves the most relevant chunks from a vector database
3. Generates an answer **using only those chunks** — and refuses when the documents don't contain the answer
4. Replies in the same language you asked in, and cites its sources

Two domains, switchable from the sidebar:

- **Medical** — 14,979 cleaned Q&A pairs from the NIH MedQuAD dataset
- **Ecommerce** — 12,625 cleaned product listings from a Flipkart catalog

---

## Key features

- **Grounded answers, no hallucination.** The bot answers strictly from retrieved documents. If the answer isn't there, it says so instead of inventing one.
- **Trilingual.** English, Urdu script, and Roman Urdu — with automatic detection and same-language replies.
- **Two domains, one app.** Medical and ecommerce knowledge bases kept in separate vector stores, switchable live.
- **Conversation memory + follow-up handling.** Vague follow-ups like "how much does it cost?" are rewritten into standalone queries using chat history.
- **Three retrieval methods.** Vector search, MMR (diverse results), and score-threshold (honest "no match" for out-of-scope questions).
- **Source transparency.** Every answer shows its sources, and a sidebar toggle reveals the exact retrieved chunks for verification.
- **PDF upload.** Add your own documents to either knowledge base at runtime.

---

## Groundedness — how we verify answers come from the data

A core concern for any RAG system is: *how do you know the bot is using the documents and not just its own training knowledge?* This project demonstrates groundedness three ways:

1. **Show retrieved chunks.** A sidebar toggle displays the exact document chunks passed to the model, so any answer can be traced back to its source text.
2. **Manual verification against the source data.** Because sources are cited by name, any answer can be checked against the original CSV row. This is especially convincing for rare entries (e.g. the disease *CADASIL*) whose specific wording could only come from the dataset, not general training knowledge.
3. **Honest refusal.** Out-of-scope questions ("What is the capital of France?") are refused, even though the model knows the answer — proving it only speaks from retrieved documents.

---

## Tech stack

| Component | Choice |
|-----------|--------|
| Language model | Qwen3-32B via Groq (cloud) or Llama 3.2 via Ollama (local) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (supports Urdu + English) |
| Vector database | ChromaDB (one store per domain) |
| Retrieval | LangChain (vector / MMR / score-threshold) |
| Interface | Streamlit |
| Datasets | NIH MedQuAD (medical), Flipkart products (ecommerce) |

---

## Project structure

```
sybrid_rag/
├── app.py                  # Streamlit app — the main interface (both domains)
├── medquad_pipeline.py     # Medical RAG pipeline (clean → chunk → embed → retrieve → answer)
├── flipkart_pipeline.py    # Ecommerce RAG pipeline (same flow, product-specific)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed) — holds GROQ_API_KEY
├── docs/                   # Cleaned datasets (medquad_cleaned.csv, flipkart_cleaned.csv)
└── db/                     # ChromaDB vector stores
    ├── medquad_chromadb/
    └── flipkart_chromadb/
```

Each pipeline file is self-contained: it can be run on its own in the terminal (interactive chat) or imported by `app.py`.

---

## How to run

### 1. Prerequisites
- Python 3.13
- A Groq API key (free tier works) — put it in a `.env` file:
  ```
  GROQ_API_KEY=your_key_here
  ```

### 2. Set up the environment
```powershell
cd sybrid_rag
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app
```powershell
streamlit run app.py
```
The app opens in your browser. Pick a domain in the sidebar and start asking questions.

### 4. (Optional) Run a pipeline directly in the terminal
```powershell
python medquad_pipeline.py      # interactive medical chat
python flipkart_pipeline.py     # interactive ecommerce chat
```

---

## The RAG pipeline (how it works internally)

1. **Clean** — remove empty rows, fix whitespace, drop duplicates, fill missing fields
2. **Load** — convert each cleaned row into a document with searchable text + metadata
3. **Chunk** — split into ~500-character pieces with 50-char overlap
4. **Embed & store** — convert chunks to vectors with a multilingual model, store in ChromaDB
5. **Retrieve** — for each question, fetch the top relevant chunks
6. **Generate** — send question + retrieved chunks to the LLM with a strict "answer only from these documents" instruction
7. **Respond** — return the grounded answer with sources, translated back to the user's language if needed

Ingestion (steps 1–4) is run once. Normal use only runs steps 5–7 against the saved vector store.

---

## Bilingual support (how the languages are handled)

- **English** → straight through the pipeline
- **Urdu script** → translated to English for retrieval → answer generated in English → translated back to Urdu script
- **Roman Urdu** → detected via common keywords → translated to English for retrieval → answer generated in Roman Urdu (product names, prices, and medical terms kept in English, as is natural)

The dataset is English-only, so all retrieval happens in English regardless of input language. Translation happens at the edges.

---

## Known limitations

- **Roman Urdu generation** is good but not perfect — Roman Urdu has no standard spelling, so wording can vary. A reliable fallback is to answer Roman Urdu questions in plain English.
- **Answer quality depends on retrieval.** Strict grounding means the bot sometimes says "I don't have information on that" for questions that are technically answerable but where retrieval missed the right chunk. The fix is better retrieval (higher K, improved chunking), not loosening the grounding rule.
- **Translation for very long answers** can occasionally read awkwardly, since marketing/medical text doesn't always translate cleanly.

---

## Datasets & credits

- **MedQuAD** — Medical Question Answering Dataset (National Institutes of Health)
- **Flipkart Products** — ecommerce product listings (Kaggle / PromptCloud)

Built by **Muhammad Ashir Siddiqui** — Sybrid Internship, 2026.
GitHub: [github.com/Ashirsiddiqui2004](https://github.com/Ashirsiddiqui2004)