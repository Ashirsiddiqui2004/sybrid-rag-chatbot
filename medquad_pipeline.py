# ============================================================
# MEDQUAD PIPELINE — Medical Q&A RAG Chatbot
# Sybrid Internship 2026 | Muhammad Ashir Siddiqui
# ============================================================
# Dataset: MedQuAD — 16,412 medical Q&A pairs from NIH
#
# Key Features:
# - Full bilingual support (Urdu script + Roman Urdu + English)
# - 3 retrieval methods (vector, MMR, score threshold)
# - 2 LLM options (Groq cloud + Ollama local)
# - Conversation memory with query reformulation
# - Interactive while loop for live user input
#
# CHANGELOG (latest fixes):
# - FIX 1: reformulate_query() now strips punctuation before
#   matching vague words ("it?" was not matching "it")
# - FIX 2: chunk_documents() prepends the question to every
#   chunk so answer-chunks are retrievable by the question
#   (requires re-ingestion to take effect)
# - FIX 3: default k raised from 5 to 8 for better coverage
# ============================================================


# ============================================================
# SECTION 1 - IMPORTS
# ============================================================
import pandas as pd
import os
import string
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq


# ============================================================
# SECTION 2 - LOAD ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ChromaDB location for the medical vector store.
# v2 = rebuilt with proper chunking (chunk_size=2000, Q&A kept together).
# Old folder stays as backup until you confirm v2 works, then delete it.
MEDQUAD_DB_PATH = "D:/sybrid_rag/db/medquad_chromadb_v2"


# ============================================================
# SECTION 3 - DATA CLEANING
# ============================================================
def clean_dataset(filepath):
    """
    Loads and cleans the raw MedQuAD dataset.
    Cleaning steps:
    1. Remove rows with missing answers
    2. Fill missing focus_area with 'General Medicine'
    3. Clean whitespace from question and answer columns
    4. Remove duplicate questions
    Returns cleaned dataframe.
    """
    print("=" * 60)
    print("MEDQUAD DATASET - CLEANING REPORT")
    print("=" * 60)

    print("\nLoading raw dataset...")
    df = pd.read_csv(filepath)

    # ---- BEFORE CLEANING ----
    print("\n" + "-" * 60)
    print("BEFORE CLEANING")
    print("-" * 60)
    print(f"Total rows:          {df.shape[0]}")
    print(f"Total columns:       {df.shape[1]}")
    print(f"Columns:             {df.columns.tolist()}")
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nDuplicate questions: {df.duplicated(subset=['question']).sum()}")
    print(f"\nAnswer length stats (characters):")
    print(df['answer'].dropna().str.len().describe())
    print(f"\nUnique focus areas:  {df['focus_area'].nunique()}")
    print(f"Top 5 focus areas:")
    print(df['focus_area'].value_counts().head(5))

    # ---- CLEANING ----
    print("\n" + "-" * 60)
    print("CLEANING STEPS")
    print("-" * 60)

    # Always work on a copy — never modify original data
    df_clean = df.copy()

    # Step 1 — Remove rows with missing answers
    # A Q&A pair with no answer is useless for the chatbot
    before = df_clean.shape[0]
    df_clean = df_clean.dropna(subset=['answer'])
    after = df_clean.shape[0]
    print(f"\n✓ Step 1 - Removed missing answers:     {before - after} rows removed ({before} → {after})")

    # Step 2 — Fill missing focus_area with 'General Medicine'
    # Keep valid Q&A pairs instead of dropping them
    before_nulls = df_clean['focus_area'].isnull().sum()
    df_clean['focus_area'] = df_clean['focus_area'].fillna('General Medicine')
    print(f"✓ Step 2 - Filled missing focus_area:   {before_nulls} filled with 'General Medicine'")

    # Step 3 — Clean question column
    # Extra whitespace causes inconsistent embeddings
    df_clean['question'] = df_clean['question'].str.strip()
    df_clean['question'] = df_clean['question'].str.replace(r'\s+', ' ', regex=True)
    print(f"✓ Step 3 - Cleaned question column:     Removed extra whitespace")

    # Step 4 — Clean answer column
    # Medical answers from web often have inconsistent spacing
    df_clean['answer'] = df_clean['answer'].str.strip()
    df_clean['answer'] = df_clean['answer'].str.replace(r'\s+', ' ', regex=True)
    print(f"✓ Step 4 - Cleaned answer column:       Removed extra whitespace")

    # Step 5 — Remove duplicate questions
    # Duplicates waste storage and return same info multiple times
    before = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates(subset=['question'])
    after = df_clean.shape[0]
    print(f"✓ Step 5 - Removed duplicate questions: {before - after} removed ({before} → {after})")

    # ---- AFTER CLEANING ----
    print("\n" + "-" * 60)
    print("AFTER CLEANING")
    print("-" * 60)
    print(f"Total rows:          {df_clean.shape[0]}")
    print(f"\nMissing values:")
    print(df_clean.isnull().sum())
    print(f"\nDuplicate questions: {df_clean.duplicated(subset=['question']).sum()}")

    # ---- SUMMARY ----
    print("\n" + "-" * 60)
    print("SUMMARY - BEFORE vs AFTER")
    print("-" * 60)
    print(f"{'Metric':<30} {'Before':>10} {'After':>10}")
    print("-" * 50)
    print(f"{'Total rows':<30} {df.shape[0]:>10} {df_clean.shape[0]:>10}")
    print(f"{'Missing answers':<30} {df['answer'].isnull().sum():>10} {df_clean['answer'].isnull().sum():>10}")
    print(f"{'Missing focus_area':<30} {df['focus_area'].isnull().sum():>10} {0:>10}")
    print(f"{'Duplicate questions':<30} {df.duplicated(subset=['question']).sum():>10} {df_clean.duplicated(subset=['question']).sum():>10}")

    # Save cleaned dataset — never overwrite original
    df_clean.to_csv("docs/medquad_cleaned.csv", index=False)
    print("\n✓ Cleaned dataset saved to docs/medquad_cleaned.csv")
    print("=" * 60)

    return df_clean


# ============================================================
# SECTION 4 - LOAD DOCUMENTS
# ============================================================
def load_documents(df):
    """
    Converts each cleaned CSV row into a LangChain Document.
    page_content = Question + Answer combined (both searchable)
    metadata     = domain tag, source, focus_area, question
    """
    print("\n" + "=" * 60)
    print("STEP - LOADING DOCUMENTS")
    print("=" * 60)

    documents = []

    for _, row in df.iterrows():
        # Combine question and answer — both searchable
        content = f"Question: {row['question']}\nAnswer: {row['answer']}"

        # Metadata stored alongside vector — NOT embedded
        # Used for citation and domain filtering
        metadata = {
            "domain": "medical",
            "source": row['source'],
            "focus_area": row['focus_area'],
            "question": row['question']
        }

        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    print(f"Total documents created: {len(documents)}")
    print(f"Sample preview: {documents[0].page_content[:200]}...")
    return documents


# ============================================================
# SECTION 5 - CHUNK DOCUMENTS
# ============================================================
def chunk_documents(documents):
    """
    Splits long documents into smaller chunks.

    FIX 2 (v2) — Keep questions and answers TOGETHER:
    The original 500-char chunking split "Question: X?" and "Answer: Y"
    into SEPARATE chunks. Retrieval then matched the user's question to
    the question-only chunk (which has no answer), so the bot refused
    even though the answer existed in another chunk.

    Two changes fix this:
    1. Larger chunk_size (1200) so most complete Q&A pairs fit in ONE
       chunk — the answer travels with its question.
    2. For any answer long enough to still overflow into extra chunks,
       we prepend BOTH the question AND an "Answer:" marker so every
       piece is retrievable by the question and is recognised as answer
       content, not an orphaned question.
    """
    print("\n" + "=" * 60)
    print("STEP - CHUNKING DOCUMENTS")
    print("=" * 60)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,    # data-driven: at 2000, ~84% of MedQuAD Q&A pairs
                            # fit in ONE chunk (vs only 22% at 500), so the
                            # answer stays with its question instead of being
                            # orphaned in a separate chunk
        chunk_overlap=200,  # generous overlap so long answers that DO split
                            # keep continuity across pieces
        separators=["\n\n", ". ", " ", ""]  # no bare "\n" — prevents a clean
                            # split between the Question line and Answer line
    )

    chunks = splitter.split_documents(documents)

    # For any chunk that still lost its question (long-answer overflow),
    # prepend the question so it's retrievable, and ensure it reads as answer
    # content rather than an orphaned question.
    for chunk in chunks:
        q = chunk.metadata.get("question", "")
        text = chunk.page_content
        if q and "Question:" not in text:
            # This is an overflow chunk — attach the question context
            chunk.page_content = f"Question: {q}\nAnswer (continued): {text}"

    print(f"Total chunks created: {len(chunks)}")
    print(f"Sample chunk: {chunks[0].page_content[:200]}...")
    return chunks


# ============================================================
# SECTION 6 - EMBED AND STORE IN CHROMADB
# ============================================================
def embed_and_store(chunks):
    """
    Converts chunks into vectors and stores in ChromaDB on D drive.
    Uses multilingual model supporting Urdu + English.
    Processes in batches of 50 to avoid RAM overflow.
    """
    print("\n" + "=" * 60)
    print("STEP - EMBEDDING AND STORING IN CHROMADB")
    print("=" * 60)

    # Load multilingual embedding model — free, supports Urdu + English
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Embedding chunks and storing in ChromaDB...")
    batch_size = 50  # Small batches to avoid RAM overflow
    vector_store = None
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]

        if vector_store is None:
            # First batch — create new ChromaDB collection
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embedding_model,
                persist_directory=MEDQUAD_DB_PATH,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            # Add to existing collection
            vector_store.add_documents(batch)

        done = min(i + batch_size, total)
        print(f"  Processed {done}/{total} chunks... ({done/total*100:.1f}%)")

    print(f"\nTotal chunks stored: {vector_store._collection.count()}")
    return vector_store


# ============================================================
# SECTION 7 - LOAD EXISTING VECTOR STORE
# ============================================================
def load_vector_store():
    """
    Reconnects to existing MedQuAD ChromaDB on D drive.
    Use this on second run onwards — no re-embedding needed.
    Must use same embedding model used during ingestion.
    """
    print("\nConnecting to existing MedQuAD ChromaDB...")

    # Must use SAME model as during ingestion
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vector_store = Chroma(
        persist_directory=MEDQUAD_DB_PATH,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Connected! Total chunks: {vector_store._collection.count()}")
    return vector_store


# ============================================================
# SECTION 8 - RETRIEVAL METHODS
# ============================================================
def vector_search(vector_store, question, k=8):
    """
    Basic semantic similarity search.
    Returns the top-K most similar chunks.

    With the improved chunking (chunk_size=2000, most Q&A pairs kept whole),
    retrieved chunks contain both the question and its answer, so this simple
    search reliably returns answer-bearing content.
    """
    results = vector_store.similarity_search(question, k=k)
    return results


def mmr_search(vector_store, question, k=3, fetch_k=15):
    """
    Maximum Marginal Relevance search.
    Fetches 15 chunks then selects 3 most diverse + relevant.
    Prevents returning multiple near-identical chunks.
    Best for topics where many chunks overlap in content.
    """
    results = vector_store.max_marginal_relevance_search(
        question, k=k, fetch_k=fetch_k
    )
    return results


def score_threshold_search(vector_store, question, k=8, threshold=0.3):
    """
    Similarity search with minimum score threshold of 0.3.
    Returns empty list if nothing qualifies.
    Prevents hallucination on completely out-of-scope questions.
    Best for production honesty.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": threshold, "k": k}
    )
    results = retriever.invoke(question)
    return results


# ============================================================
# SECTION 9 - GROQ LLM (Cloud)
# ============================================================
# API Key = authentication (proves identity to Groq servers)
# Model   = which AI to use (Qwen3-32B in our case)
# Different purposes: access vs tool selection
client = Groq(api_key=groq_api_key)


def ask_llm(prompt, model="qwen/qwen3-32b"):
    """
    Sends prompt to Groq LLM and returns clean response.
    Strips Qwen3 <think>...</think> reasoning block automatically.
    Users only need the final clean answer.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    # Strip Qwen3 thinking block — keep only final answer
    if "<think>" in answer and "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()

    return answer


# ============================================================
# SECTION 10 - OLLAMA LLM (Local)
# ============================================================
def ask_ollama(prompt, model="llama3.2"):
    """
    Sends prompt to locally running Ollama model.
    No internet or API key needed — runs 100% on your machine.
    Medical data never leaves your machine (privacy benefit).
    Must run 'ollama serve' in separate terminal first.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False  # Get complete response at once
    }
    response = requests.post(url, json=payload)
    result = response.json()
    return result["message"]["content"]


# ============================================================
# SECTION 11 - LANGUAGE DETECTION
# ============================================================
def detect_language(text):
    """
    Basic Urdu script detection using Unicode range U+0600-U+06FF.
    More than 2 Urdu characters → Urdu. Otherwise → English.
    Used internally — use detect_full_language() for complete detection.
    """
    urdu_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return 'urdu' if urdu_chars > 2 else 'english'


# ============================================================
# SECTION 12 - ROMAN URDU DETECTION
# ============================================================
def detect_roman_urdu(text):
    """
    Detects Roman Urdu — Urdu words written in English letters.
    Example: "diabetes kya hai" instead of "ذیابیطس کیا ہے؟"

    Why needed:
    Pakistanis commonly type Urdu in Roman script without
    an Urdu keyboard. Unicode detector treats this as English
    — missing the intent that user wants a Urdu-style response.

    How it works:
    Checks for common Roman Urdu words that don't appear in
    normal English. If 2+ found → Roman Urdu detected.

    Returns 'roman_urdu' or 'english'
    """
    # Common Roman Urdu words that don't appear in English
    roman_urdu_keywords = [
        'kya', 'hai', 'hain', 'kaise', 'kyun', 'kahan',
        'kaun', 'kab', 'mujhe', 'mera', 'meri', 'mere',
        'aap', 'tum', 'woh', 'yeh', 'iska', 'uska',
        'batao', 'bolo', 'samjhao', 'dijiye', 'chahiye',
        'nahi', 'nahin', 'hoga', 'hogi', 'tha', 'thi',
        'kar', 'karo', 'karna', 'ho', 'hona', 'wala',
        'ke', 'ki', 'ka', 'se', 'mein', 'par', 'tak',
        'aur', 'ya', 'lekin', 'phir', 'ab', 'jab',
        'ilaj', 'bimari', 'dawa', 'mareez', 'bemar',
        'sehat', 'takleef', 'dard'
    ]

    text_lower = text.lower()
    # Strip punctuation from each word so "hai?" matches "hai", "kya," matches
    # "kya", etc. Without this, a trailing question mark on the last word (very
    # common) drops the match count below the threshold and Roman Urdu gets
    # misdetected as English.
    words = [w.strip(string.punctuation) for w in text_lower.split()]

    # Count how many Roman Urdu keywords appear
    matches = sum(1 for word in words if word in roman_urdu_keywords)

    return 'roman_urdu' if matches >= 2 else 'english'


def detect_full_language(text):
    """
    Complete language detector handling 3 cases:
    1. Urdu script (ذیابیطس کیا ہے؟) → 'urdu'
    2. Roman Urdu (diabetes kya hai) → 'roman_urdu'
    3. English (What is diabetes?) → 'english'

    Priority: Urdu script check first, then Roman Urdu, then English.
    """
    # First check for Urdu script characters
    urdu_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    if urdu_chars > 2:
        return 'urdu'

    # Then check for Roman Urdu keywords
    if detect_roman_urdu(text) == 'roman_urdu':
        return 'roman_urdu'

    # Otherwise English
    return 'english'


# ============================================================
# SECTION 13 - TRANSLATION FUNCTIONS
# ============================================================
def translate_to_english(text):
    """
    Translates Urdu (script or Roman) query to English
    before ChromaDB search.

    Why needed:
    MedQuAD dataset is entirely in English. Searching with
    Urdu text gives poor retrieval — even multilingual models
    don't match Urdu queries to English chunks perfectly.
    Translation ensures ChromaDB finds correct medical chunks.
    This is Layer 1 of our bilingual support strategy.

    Example: "ذیابیطس کیا ہے؟" → "What is diabetes?"
    """
    prompt = f"""Translate this Urdu text to English.
Return ONLY the English translation — nothing else.
Keep medical terms accurate.

Urdu text: {text}
English translation:"""

    return ask_llm(prompt)


def translate_to_urdu(text):
    """
    Translates English answer back to Urdu script for Urdu users.

    Why needed:
    LLM generates English answer from English chunks.
    We translate the complete final answer to Urdu.
    This is Layer 2 of our bilingual support strategy.

    Example: "Diabetes is a condition..." → "ذیابیطس ایک حالت ہے..."
    """
    prompt = f"""Translate this English medical text to Urdu.
Return ONLY the Urdu translation — nothing else.
Keep medical terms accurate and use proper Urdu script.

English text: {text}
Urdu translation:"""

    return ask_llm(prompt)


def translate_to_roman_urdu(text):
    """
    Converts an English answer into Roman Urdu (Urdu written in the
    Latin/English alphabet), the way Pakistanis type Urdu casually.

    Why this exists as a separate step:
    Asking the LLM to *generate* directly in Roman Urdu is unreliable —
    with English documents and English instructions it drifts back to
    English. Generating in English first, then translating the finished
    answer to Roman Urdu, is far more consistent (same trick that makes
    the Urdu-script path reliable).

    Medical terms and numbers stay in English, which is natural.

    Example: "Tuberculosis spreads through the air when..."
          -> "Tuberculosis hawa ke zariye phailta hai jab..."
    """
    # Use Llama 3.3 70B for this step instead of Qwen3-32B. Qwen drifts to
    # Hindi/Devanagari when asked for Roman Urdu; Llama handles South Asian
    # romanization more reliably. Everything else in the app still uses Qwen.
    ROMAN_URDU_MODEL = "llama-3.3-70b-versatile"

    prompt = f"""Convert the following English text into ROMAN URDU.

STRICT OUTPUT RULES:
- Use ONLY English/Latin letters (a-z). Every single character must be a Latin letter, digit, or punctuation.
- ABSOLUTELY NO Devanagari/Hindi script (like हवा, फैलता). NO Urdu script (like ہوا). If you output any non-Latin character, that is WRONG.
- Roman Urdu = Urdu language spelled with English letters, the way Pakistanis text.
- Keep medical terms, disease names, and numbers in English.
- Return ONLY the Roman Urdu line, nothing else.

Example 1:
English: "Tuberculosis spreads through the air when an infected person coughs or sneezes."
Roman Urdu: Tuberculosis hawa ke zariye phailta hai jab koi infected insaan khaansta ya cheenkta hai.

Example 2:
English: "Diabetes is a disease where blood sugar becomes high."
Roman Urdu: Diabetes aik bimari hai jis mein blood sugar barh jati hai.

Now convert this:
English: {text}
Roman Urdu:"""

    result = ask_llm(prompt, model=ROMAN_URDU_MODEL)
    print(f"[DEBUG-RU] input(len={len(text)}): {text[:80]!r}")
    print(f"[DEBUG-RU] llama output: {result[:120]!r}")

    # NOTE: previously there was a retry here that re-called the model if the
    # output contained non-Latin script. It's removed because (a) Llama 3.3
    # reliably returns Latin-script Roman Urdu, and (b) the check could not
    # distinguish Roman Urdu from English (both Latin), so a spurious retry
    # risked replacing good Roman Urdu. Keep it simple: one clean call.
    return result


# ============================================================
# SECTION 14 - QUERY REFORMULATION
# ============================================================
def reformulate_query(question, conversation_history):
    """
    Rewrites vague follow-up questions into standalone queries
    that ChromaDB can search effectively.

    Problem:
    "What causes it?" doesn't tell ChromaDB what "it" is.
    ChromaDB searches literally and finds random irrelevant chunks.

    Solution:
    Use the most recent conversation turn to rewrite the question.
    "What causes it?" + "Glaucoma" history → "What causes Glaucoma?"

    FIX 1 — Punctuation stripping:
    "What causes it?" splits into ['what', 'causes', 'it?'] and
    'it?' != 'it', so the vague-word check silently failed.
    We now strip punctuation from each word before comparing.

    Why only last 2 messages (1 turn)?
    Taking more history adds noise and can confuse reformulation.
    The immediately previous question/answer is almost always
    enough context to resolve vague pronouns.

    Only runs if vague pronouns detected AND history exists.
    Returns original question unchanged if no reformulation needed.
    """
    # Vague pronouns that signal question needs context
    vague_words = ['it', 'its', 'they', 'their', 'this',
                   'that', 'these', 'those', 'the condition',
                   'the disease', 'the same']

    question_lower = question.lower()

    # FIX 1 — strip punctuation so "it?" matches "it"
    words = [w.strip(string.punctuation) for w in question_lower.split()]
    needs_reformulation = any(word in words for word in vague_words)

    # No reformulation needed
    if not needs_reformulation or not conversation_history:
        return question

    # Take only the LAST 2 messages (1 turn) for context
    # More history adds noise — last turn is enough
    recent_history = conversation_history[-2:]
    history_text = ""
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Limit to 300 chars to keep prompt focused
        history_text += f"{role}: {msg['content'][:300]}\n"

    reformulation_prompt = f"""Given this recent conversation:
{history_text}

Rewrite this follow-up question as a clear standalone medical search query.
Remove vague pronouns like "it", "its", "they", "this" etc.
Replace them with the actual medical subject from the conversation.
Return ONLY the rewritten question — nothing else. No explanation.

Follow-up question: {question}
Rewritten question:"""

    rewritten = ask_llm(reformulation_prompt)

    # Clean up any quotes or extra whitespace
    rewritten = rewritten.strip().strip('"').strip("'")

    print(f"[Reformulated: {rewritten}]")

    return rewritten


# ============================================================
# SECTION 15 - FULL RAG PIPELINE
# ============================================================
def rag_pipeline(vector_store, question, search_type="vector",
                 k=8, use_ollama=False):
    """
    Complete end-to-end Medical RAG pipeline.
    Supports English, Urdu script, and Roman Urdu.

    Bilingual flow for Urdu/Roman Urdu questions:
    1. Detect language
    2. Translate query to English (Layer 1)
    3. Search ChromaDB with English query
    4. Generate English answer from English chunks
    5. Translate back to Urdu script if needed (Layer 2)
    6. Return grounded answer with source citation
    """
    print(f"\nQuestion: {question}")
    print(f"Search: {search_type} | K: {k} | LLM: {'Ollama' if use_ollama else 'Groq'}")
    print("-" * 60)

    # Step 1 - Detect language (3 cases)
    language = detect_full_language(question)
    print(f"Language: {language}")

    # Step 2 - Translate to English for retrieval if needed
    # Dataset is English-only — must search in English
    if language in ['urdu', 'roman_urdu']:
        search_query = translate_to_english(question)
        print(f"Translated query: {search_query}")
    else:
        search_query = question

    # Step 3 - Retrieve relevant chunks
    if search_type == "vector":
        chunks = vector_search(vector_store, search_query, k=k)
    elif search_type == "mmr":
        chunks = mmr_search(vector_store, search_query, k=k)
    elif search_type == "threshold":
        chunks = score_threshold_search(vector_store, search_query, k=k)
    else:
        chunks = vector_search(vector_store, search_query, k=k)

    # Step 4 - Handle no results (threshold search only)
    if not chunks:
        print("No relevant chunks found above threshold.")
        if language == 'urdu':
            return "میرے دستاویزات میں اس بارے میں معلومات نہیں ہے۔"
        elif language == 'roman_urdu':
            return "Is baare mein mujhe information nahi mili documents mein."
        return "I don't have information on that in my medical documents."

    # Step 5 - Build context from retrieved chunks
    context = ""
    sources = []

    for i, chunk in enumerate(chunks):
        context += f"Document {i+1}:\n"
        context += f"Topic: {chunk.metadata.get('focus_area', 'Unknown')}\n"
        context += f"Content: {chunk.page_content}\n\n"

        source = chunk.metadata.get('focus_area', 'Unknown')
        if source not in sources:
            sources.append(source)

    # Step 6 - Language-specific instruction for LLM
    if language == 'urdu':
        language_instruction = "Answer in Urdu script only. Do not use English."
    elif language == 'roman_urdu':
        language_instruction = ("Answer in Roman Urdu (Urdu written in the English/Latin alphabet), "
                                "matching the style the user used. Do NOT use Urdu script, Hindi, or "
                                "Devanagari. Example style: 'Diabetes aik aisi bimari hai jis mein khoon "
                                "mein sugar barh jati hai.' Keep medical terms in English where natural.")
    else:
        language_instruction = "Answer in English only."

    # Step 7 - Build complete prompt
    prompt = f"""You are a helpful medical assistant.
{language_instruction}
Use the information from the documents below to answer as best you can.
The answer must come from the retrieved documents — do not use general knowledge.
Even if documents don't directly answer, use related info to help.
Only say 'I dont have information on that' if completely unrelated.
At the end mention the medical topic it relates to.

Documents:
{context}

Question: {search_query}

Answer:"""

    # Step 8 - Get answer from selected LLM
    if use_ollama:
        answer = ask_ollama(prompt)
    else:
        answer = ask_llm(prompt)

    # Step 9 - Translate to Urdu script if question was in Urdu script
    # Roman Urdu gets English answer (handled in prompt instruction)
    if language == 'urdu':
        print("Translating answer to Urdu...")
        answer = translate_to_urdu(answer)

    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
    print("-" * 60)

    return answer


# ============================================================
# SECTION 16 - CONVERSATION MEMORY
# ============================================================
# Global list storing full conversation as message dictionaries
# Format: [{"role": "user"/"assistant", "content": "..."}]
# Grows with each turn and passed to LLM every time
conversation_history = []


def chat_with_memory(vector_store, question, search_type="vector", k=8):
    """
    RAG pipeline with conversation memory, query reformulation,
    and full bilingual support (Urdu script + Roman Urdu + English).

    Three systems working together:
    1. Memory — LLM sees full conversation history
    2. Query reformulation — retriever gets clear search query
    3. Translation — Urdu questions get Urdu answers
    """
    global conversation_history

    print(f"\nUser: {question}")
    print("-" * 60)

    # Step 1 - Detect language
    language = detect_full_language(question)
    print(f"Language: {language}")

    # Step 2 - Reformulate vague follow-up questions first
    # Works on original language before translation
    search_query = reformulate_query(question, conversation_history)

    # Step 3 - Translate to English if Urdu or Roman Urdu
    # After reformulation — ensures reformulation and translation
    # both work correctly in sequence
    if language in ['urdu', 'roman_urdu']:
        search_query = translate_to_english(search_query)
        print(f"Translated search query: {search_query}")

    # Step 4 - Retrieve chunks using English search query
    if search_type == "vector":
        chunks = vector_search(vector_store, search_query, k=k)
    elif search_type == "mmr":
        chunks = mmr_search(vector_store, search_query, k=k)
    elif search_type == "threshold":
        chunks = score_threshold_search(vector_store, search_query, k=k)
    else:
        chunks = vector_search(vector_store, search_query, k=k)

    # Step 5 - Build context from retrieved chunks
    context = ""
    sources = []

    if chunks:
        for i, chunk in enumerate(chunks):
            context += f"Document {i+1}:\n"
            context += f"Topic: {chunk.metadata.get('focus_area', 'Unknown')}\n"
            context += f"Content: {chunk.page_content}\n\n"

            source = chunk.metadata.get('focus_area', 'Unknown')
            if source not in sources:
                sources.append(source)

    # Step 6 - Language instruction
    if language == 'urdu':
        language_instruction = "Answer in Urdu script only. Do not use English."
    elif language == 'roman_urdu':
        language_instruction = ("Answer in Roman Urdu (Urdu written in the English/Latin alphabet), "
                                "matching the style the user used. Do NOT use Urdu script, Hindi, or "
                                "Devanagari. Keep medical terms in English where natural.")
    else:
        language_instruction = "Answer in English only."

    # Step 7 - System prompt
    system_prompt = f"""You are a helpful medical assistant.
{language_instruction}
Use the provided documents AND conversation history to answer.
The answer must come from retrieved documents — not general knowledge.
Even if documents don't directly answer, use related info to help.
Only say 'I dont have information on that' if completely unrelated.
At the end mention the medical topic."""

    # Step 8 - Build messages with full conversation history
    # Format: [system, user_1, assistant_1, ..., current_user]
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)

    current_message = f"""Retrieved Medical Documents:
{context}

Current Question: {question}"""
    messages.append({"role": "user", "content": current_message})

    # Step 9 - Get answer from LLM
    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=messages
    )

    answer = response.choices[0].message.content

    if "<think>" in answer and "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()

    # Step 10 - Translate to Urdu script if needed
    if language == 'urdu':
        print("Translating answer to Urdu...")
        answer = translate_to_urdu(answer)

    # Step 11 - Store original question in history (not translated)
    # Keeps history natural for future reformulation
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    print(f"Assistant: {answer}")
    if sources:
        print(f"Sources: {sources}")
    print(f"Memory: {len(conversation_history)//2} turns stored")
    print("-" * 60)

    return answer


def clear_memory():
    """Resets conversation history for a fresh session."""
    global conversation_history
    conversation_history = []
    print("Conversation memory cleared!")


# ============================================================
# SECTION 17 - INTERACTIVE WHILE LOOP
# ============================================================
def interactive_chat(vector_store, search_type="vector", k=8):
    """
    Interactive chat loop that takes live user input.

    Replaces hardcoded test questions with real conversational
    interface in terminal. Supports English, Urdu script,
    and Roman Urdu questions.

    Commands:
    - Type any question → get answer
    - 'clear' → clear conversation memory
    - 'switch vector' → use vector search
    - 'switch mmr' → use MMR search
    - 'switch threshold' → use score threshold search
    - 'quit' or 'exit' → stop the loop
    """
    global conversation_history
    conversation_history = []

    print("\n" + "=" * 60)
    print("MEDQUAD MEDICAL RAG CHATBOT — INTERACTIVE MODE")
    print("=" * 60)
    print("Ask medical questions in English, Urdu, or Roman Urdu")
    print("Commands: 'clear' | 'switch vector/mmr/threshold' | 'quit'")
    print(f"Settings: search={search_type} | k={k}")
    print("=" * 60)

    while True:
        print()
        user_input = input("You: ").strip()

        # Skip empty input
        if not user_input:
            continue

        # ---- HANDLE COMMANDS ----
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        elif user_input.lower() == 'clear':
            clear_memory()
            print("Conversation cleared. Starting fresh.")
            continue

        elif user_input.lower() == 'switch vector':
            search_type = 'vector'
            print("Switched to: vector search")
            continue

        elif user_input.lower() == 'switch mmr':
            search_type = 'mmr'
            print("Switched to: MMR search")
            continue

        elif user_input.lower() == 'switch threshold':
            search_type = 'threshold'
            print("Switched to: score threshold search")
            continue

        # ---- DETECT LANGUAGE ----
        # Handles 3 cases: Urdu script, Roman Urdu, English
        language = detect_full_language(user_input)
        print(f"[Language: {language}]")

        # ---- QUERY REFORMULATION ----
        # Rewrites vague follow-ups using last conversation turn
        # "What causes it?" → "What causes Glaucoma?"
        search_query = reformulate_query(user_input, conversation_history)
        print(f"[Search query: {search_query}]")

        # ---- TRANSLATE TO ENGLISH FOR SEARCH ----
        # Dataset is English-only — must search in English
        # Roman Urdu also translated since it contains Urdu words
        if language in ['urdu', 'roman_urdu']:
            search_query = translate_to_english(search_query)
            print(f"[Translated: {search_query}]")

        # ---- RETRIEVE CHUNKS ----
        if search_type == "vector":
            chunks = vector_search(vector_store, search_query, k=k)
        elif search_type == "mmr":
            chunks = mmr_search(vector_store, search_query, k=k)
        elif search_type == "threshold":
            chunks = score_threshold_search(vector_store, search_query, k=k)
        else:
            chunks = vector_search(vector_store, search_query, k=k)

        # ---- HANDLE NO RESULTS ----
        if not chunks:
            if language == 'urdu':
                answer = "میرے دستاویزات میں اس بارے میں معلومات نہیں ہے۔"
            elif language == 'roman_urdu':
                answer = "Is baare mein mujhe information nahi mili documents mein."
            else:
                answer = "I don't have information on that in my medical documents."

            print(f"\nBot: {answer}")
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": answer})
            continue

        # ---- BUILD CONTEXT ----
        context = ""
        sources = []

        for i, chunk in enumerate(chunks):
            context += f"Document {i+1}:\n"
            context += f"Topic: {chunk.metadata.get('focus_area', 'Unknown')}\n"
            context += f"Content: {chunk.page_content}\n\n"

            source = chunk.metadata.get('focus_area', 'Unknown')
            if source not in sources:
                sources.append(source)

        # ---- LANGUAGE INSTRUCTION FOR LLM ----
        if language == 'urdu':
            language_instruction = "Answer in Urdu script only. Do not use English."
        elif language == 'roman_urdu':
            language_instruction = ("Answer in Roman Urdu (Urdu written in the English/Latin alphabet), "
                                    "matching the style the user used. Do NOT use Urdu script, Hindi, or "
                                    "Devanagari. Keep medical terms in English where natural.")
        else:
            language_instruction = "Answer in English only."

        # ---- BUILD SYSTEM PROMPT ----
        system_prompt = f"""You are a helpful medical assistant.
{language_instruction}
Use the provided documents AND conversation history to answer.
The answer must come from retrieved documents — not general knowledge.
Even if documents don't directly answer, use related info to help.
Only say 'I dont have information on that' if completely unrelated.
At the end mention the medical topic."""

        # ---- BUILD MESSAGES WITH CONVERSATION HISTORY ----
        # Includes full history so LLM understands follow-up context
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)

        current_message = f"""Retrieved Medical Documents:
{context}

Current Question: {user_input}"""
        messages.append({"role": "user", "content": current_message})

        # ---- GET ANSWER FROM LLM ----
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages
        )

        answer = response.choices[0].message.content

        if "<think>" in answer and "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()

        # ---- TRANSLATE TO URDU SCRIPT IF NEEDED ----
        # Roman Urdu gets English answer (handled in prompt)
        # Only Urdu script questions get Urdu script answers
        if language == 'urdu':
            print("[Translating to Urdu...]")
            answer = translate_to_urdu(answer)

        # ---- STORE IN CONVERSATION HISTORY ----
        # Store ORIGINAL question (not reformulated/translated)
        # Keeps history natural for future reformulation
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": answer})

        # ---- DISPLAY ANSWER ----
        print(f"\nBot: {answer}")
        print(f"Sources: {', '.join(sources)}")
        print(f"Memory: {len(conversation_history)//2} turns")
        print("-" * 60)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # ---- INGESTION (DONE — keep commented) ----
    # NOTE: chunk_documents() now prepends the question to every
    # chunk (FIX 2). To apply this to your data, uncomment these
    # 4 lines, change persist_directory in embed_and_store() to a
    # NEW folder (e.g. db/medquad_chromadb_v2), run once, then
    # update load_vector_store() to point to the new folder.
    # df_cleaned = clean_dataset("docs/medquad.csv")
    # documents = load_documents(df_cleaned)
    # chunks = chunk_documents(documents)
    # vector_store = embed_and_store(chunks)

    # ---- LOAD EXISTING DB ----
    # Connects to saved ChromaDB — no re-embedding needed
    vector_store = load_vector_store()

    # ---- START INTERACTIVE CHAT ----
    # User types questions live — no hardcoded questions
    # Supports English, Urdu script, and Roman Urdu
    interactive_chat(vector_store, search_type="vector", k=8)
