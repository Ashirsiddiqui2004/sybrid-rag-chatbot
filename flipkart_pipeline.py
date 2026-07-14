# ============================================================
# FLIPKART PIPELINE — Ecommerce Product RAG Chatbot
# Sybrid Internship 2026 | Muhammad Ashir Siddiqui
# ============================================================
# This file is a complete self-contained RAG pipeline for the
# Flipkart ecommerce products dataset. It covers everything
# from raw data cleaning to LLM-powered answer generation.
#
# Dataset: Flipkart Products — 20,000 product listings
# Source: https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products
#
# Pipeline Flow:
# 1. Load and clean the raw Flipkart CSV
# 2. Convert cleaned rows into LangChain documents
# 3. Chunk documents into 500-character pieces
# 4. Embed chunks using multilingual model → store in ChromaDB
# 5. Retrieve relevant chunks using 3 methods
# 6. Send retrieved chunks + question to LLM → get answer
#
# Key Features:
# - Full bilingual support (Urdu + English)
# - 3 retrieval methods (vector, MMR, score threshold)
# - 2 LLM options (Groq cloud + Ollama local)
# - Conversation memory with query reformulation
#
# CHANGELOG (latest fixes — mirror of medquad_pipeline):
# - FIX 1: reformulate_query() strips punctuation before matching
#   vague words ("it?" now matches "it")
# - FIX 2: chunk_documents() prepends product name to every chunk
#   so all parts of a product are retrievable (needs re-ingestion)
# - FIX 3: default k raised from 5 to 8
# - FIX 4: robust <think> stripping (handles unclosed tag)
# ============================================================


# ============================================================
# SECTION 1 - IMPORTS
# ============================================================
import pandas as pd      # For loading and cleaning the CSV dataset
import os                # For reading environment variables and file paths
import string            # For stripping punctuation in reformulation
import requests          # For sending HTTP requests to local Ollama server
from dotenv import load_dotenv                          # Loads API keys from .env file securely
from langchain_core.documents import Document           # LangChain Document object (text + metadata)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For chunking long documents
from langchain_huggingface import HuggingFaceEmbeddings # Free multilingual embedding model
from langchain_chroma import Chroma                     # ChromaDB vector store integration
from groq import Groq                                   # Groq API client for LLM inference


# ============================================================
# SECTION 2 - LOAD ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# ============================================================
# SECTION 3 - DATA CLEANING
# ============================================================
def clean_dataset(filepath):
    """
    Loads and cleans the raw Flipkart ecommerce dataset.

    Cleaning steps:
    1. Remove rows with missing description
    2. Remove rows with missing price
    3. Fill missing brand with 'Unknown Brand'
    4. Extract clean category from product_category_tree
    5. Clean description and product_name text
    6. Remove duplicate products
    """
    print("=" * 60)
    print("FLIPKART DATASET - CLEANING REPORT")
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
    print(f"\nDuplicate products:  {df.duplicated(subset=['product_name']).sum()}")
    print(f"\nDescription length stats (characters):")
    print(df['description'].dropna().str.len().describe())

    # ---- CLEANING ----
    print("\n" + "-" * 60)
    print("CLEANING STEPS")
    print("-" * 60)

    df_clean = df.copy()

    # Step 1 — Remove rows with missing description
    before = df_clean.shape[0]
    df_clean = df_clean.dropna(subset=['description'])
    after = df_clean.shape[0]
    print(f"\n✓ Step 1 - Removed missing descriptions: {before - after} rows removed ({before} → {after})")

    # Step 2 — Remove rows with missing price
    before = df_clean.shape[0]
    df_clean = df_clean.dropna(subset=['retail_price', 'discounted_price'])
    after = df_clean.shape[0]
    print(f"✓ Step 2 - Removed missing prices:        {before - after} rows removed ({before} → {after})")

    # Step 3 — Fill missing brand with 'Unknown Brand'
    before_nulls = df_clean['brand'].isnull().sum()
    df_clean['brand'] = df_clean['brand'].fillna('Unknown Brand')
    print(f"✓ Step 3 - Filled missing brands:         {before_nulls} filled with 'Unknown Brand'")

    # Step 4 — Extract clean category from product_category_tree
    # Raw format: ["Clothing >> Women's Clothing >> Shorts"]
    # We extract just the top-level category (before first >>)
    def extract_category(cat_string):
        try:
            clean = cat_string.strip('[]"').split('>>')[0].strip()
            return clean
        except:
            return "Unknown Category"

    df_clean['category'] = df_clean['product_category_tree'].apply(extract_category)
    print(f"✓ Step 4 - Extracted clean category from product_category_tree")
    print(f"           Unique categories found: {df_clean['category'].nunique()}")

    # Step 5 — Clean description column
    df_clean['description'] = df_clean['description'].astype(str).str.strip()
    df_clean['description'] = df_clean['description'].str.replace(r'\s+', ' ', regex=True)
    print(f"✓ Step 5 - Cleaned description column")

    # Step 6 — Clean product_name column
    df_clean['product_name'] = df_clean['product_name'].astype(str).str.strip()
    print(f"✓ Step 6 - Cleaned product_name column")

    # Step 7 — Remove duplicate product names
    before = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates(subset=['product_name'])
    after = df_clean.shape[0]
    print(f"✓ Step 7 - Removed duplicate products:   {before - after} removed ({before} → {after})")

    # ---- AFTER CLEANING ----
    print("\n" + "-" * 60)
    print("AFTER CLEANING")
    print("-" * 60)
    print(f"Total rows:          {df_clean.shape[0]}")
    print(f"\nMissing values:")
    print(df_clean[['product_name', 'description',
                     'retail_price', 'brand', 'category']].isnull().sum())
    print(f"\nUnique categories:   {df_clean['category'].nunique()}")
    print(f"Top 10 categories:")
    print(df_clean['category'].value_counts().head(10))

    # ---- SUMMARY ----
    print("\n" + "-" * 60)
    print("SUMMARY - BEFORE vs AFTER")
    print("-" * 60)
    print(f"{'Metric':<30} {'Before':>10} {'After':>10}")
    print("-" * 50)
    print(f"{'Total rows':<30} {df.shape[0]:>10} {df_clean.shape[0]:>10}")
    print(f"{'Missing descriptions':<30} {df['description'].isnull().sum():>10} {df_clean['description'].isnull().sum():>10}")
    print(f"{'Missing prices':<30} {df['retail_price'].isnull().sum():>10} {df_clean['retail_price'].isnull().sum():>10}")
    print(f"{'Missing brands':<30} {df['brand'].isnull().sum():>10} {0:>10}")
    print(f"{'Duplicate products':<30} {df.duplicated(subset=['product_name']).sum():>10} {df_clean.duplicated(subset=['product_name']).sum():>10}")

    df_clean.to_csv("docs/flipkart_cleaned.csv", index=False)
    print("\n✓ Cleaned dataset saved to docs/flipkart_cleaned.csv")
    print("=" * 60)

    return df_clean


# ============================================================
# SECTION 4 - LOAD DOCUMENTS
# ============================================================
def load_documents(df):
    """
    Converts each cleaned Flipkart row into a LangChain Document.

    page_content = product name + brand + category + price + description
    metadata     = domain tag, source, category, brand, price

    Including price in page_content helps price-related queries
    like "Show me products under Rs 500" retrieve correctly.
    """
    print("\n" + "=" * 60)
    print("STEP - LOADING DOCUMENTS")
    print("=" * 60)

    documents = []

    for _, row in df.iterrows():
        # Rich content combining all searchable product attributes
        content = (
            f"Product: {row['product_name']}\n"
            f"Brand: {row['brand']}\n"
            f"Category: {row['category']}\n"
            f"Price: Rs.{row['discounted_price']} "
            f"(Original: Rs.{row['retail_price']})\n"
            f"Description: {row['description']}"
        )

        # Metadata stored alongside vector — NOT embedded
        metadata = {
            "domain": "ecommerce",
            "source": "Flipkart",
            "product_name": str(row['product_name']),
            "brand": str(row['brand']),
            "category": str(row['category']),
            "price": float(row['discounted_price']),
            "retail_price": float(row['retail_price'])
        }

        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    print(f"Total documents created: {len(documents)}")
    print(f"Sample preview:\n{documents[0].page_content[:300]}...")
    return documents


# ============================================================
# SECTION 5 - CHUNK DOCUMENTS
# ============================================================
def chunk_documents(documents):
    """
    Splits long product documents into smaller chunks.

    FIX 2 — Product name prepended to every chunk:
    Long descriptions split into multiple chunks. Without this,
    only the first chunk names the product, so later chunks
    (with specs/features) rank poorly for "price of X" queries.
    Prepending the product name keeps every chunk findable.

    NOTE: Only affects NEW ingestion. To apply to existing data,
    re-run ingestion with a fresh persist_directory.
    """
    print("\n" + "=" * 60)
    print("STEP - CHUNKING DOCUMENTS")
    print("=" * 60)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # FIX 2 — prepend product name to chunks that lost it in splitting
    for chunk in chunks:
        name = chunk.metadata.get("product_name", "")
        if name and not chunk.page_content.startswith("Product:"):
            chunk.page_content = f"Product: {name}\n{chunk.page_content}"

    print(f"Total chunks created: {len(chunks)}")
    print(f"Sample chunk:\n{chunks[0].page_content[:300]}...")
    return chunks


# ============================================================
# SECTION 6 - EMBED AND STORE IN CHROMADB
# ============================================================
def embed_and_store(chunks):
    """
    Converts product chunks into vectors and stores in ChromaDB.
    Separate ChromaDB from MedQuAD (db/flipkart_chromadb).
    Processes in batches of 50 to avoid RAM overflow.
    """
    print("\n" + "=" * 60)
    print("STEP - EMBEDDING AND STORING IN CHROMADB")
    print("=" * 60)

    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Embedding chunks and storing in Flipkart ChromaDB...")
    batch_size = 50
    vector_store = None
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embedding_model,
                persist_directory="D:/sybrid_rag/db/flipkart_chromadb",
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
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
    Reconnects to existing Flipkart ChromaDB on D drive.
    Use this on second run onwards — no re-embedding needed.
    Must use same embedding model used during ingestion.
    """
    print("\nConnecting to existing Flipkart ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vector_store = Chroma(
        persist_directory="D:/sybrid_rag/db/flipkart_chromadb",
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
    Always returns K results. Good for product browsing queries.
    k=8 default gives broader product coverage.
    """
    results = vector_store.similarity_search(question, k=k)
    return results


def mmr_search(vector_store, question, k=3, fetch_k=15):
    """
    Maximum Marginal Relevance search.
    Returns K diverse products from pool of fetch_k=15.
    Prevents returning multiple versions of same product.
    Best for: "Show me clothing options" — want variety.
    """
    results = vector_store.max_marginal_relevance_search(
        question, k=k, fetch_k=fetch_k
    )
    return results


def score_threshold_search(vector_store, question, k=8, threshold=0.3):
    """
    Similarity search with minimum score threshold of 0.3.
    Returns empty list if nothing qualifies — prevents showing
    completely unrelated products. Best for specific queries.
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
client = Groq(api_key=groq_api_key)


def _strip_think(answer):
    """
    Removes Qwen3 <think>...</think> reasoning block.
    FIX 4 — handles BOTH closed and unclosed <think> tags
    (model can hit token limit mid-thought and never close it,
    which would leak the raw monologue to the user).
    """
    if "</think>" in answer:
        return answer.split("</think>")[-1].strip()
    elif "<think>" in answer:
        # Opened but never closed — drop everything after <think>
        cleaned = answer.split("<think>")[0].strip()
        return cleaned
    return answer


def ask_llm(prompt, model="qwen/qwen3-32b"):
    """
    Sends prompt to Groq LLM and returns clean response.
    Strips Qwen3 <think>...</think> block automatically.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    return _strip_think(answer)


# ============================================================
# SECTION 10 - OLLAMA LLM (Local)
# ============================================================
def ask_ollama(prompt, model="llama3.2"):
    """
    Sends prompt to locally running Ollama model.
    No internet or API key needed. Runs 100% locally.
    Must run 'ollama serve' in separate terminal first.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    response = requests.post(url, json=payload)
    result = response.json()
    return result["message"]["content"]


# ============================================================
# SECTION 11 - LANGUAGE DETECTION
# ============================================================
def detect_language(text):
    """
    Detects Urdu vs English based on Unicode range U+0600-U+06FF.
    More than 2 Urdu characters → Urdu. Otherwise → English.

    NOTE: Prefer detect_full_language() below — it also catches
    Roman Urdu. This basic one is kept for backward compatibility.
    """
    urdu_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return 'urdu' if urdu_chars > 2 else 'english'


def detect_roman_urdu(text):
    """
    Detects Roman Urdu — Urdu words written in English letters.
    Example: "shoes dikhao" or "cycling shorts kitne ke hain".

    Why needed:
    Pakistanis commonly type Urdu in Roman script without an Urdu
    keyboard. The Unicode detector treats this as English and
    misses the intent that the user wants an Urdu-style response.

    If 2+ common Roman Urdu words are found → Roman Urdu detected.
    Returns 'roman_urdu' or 'english'.
    """
    # Common Roman Urdu words that don't appear in normal English.
    # Includes shopping/ecommerce-flavoured words (keemat, kitne,
    # dikhao, sasta, mehnga) alongside general Urdu grammar words.
    roman_urdu_keywords = [
        'kya', 'hai', 'hain', 'kaise', 'kyun', 'kahan',
        'kaun', 'kab', 'kitne', 'kitna', 'kitni',
        'mujhe', 'mera', 'meri', 'mere',
        'aap', 'tum', 'woh', 'yeh', 'iska', 'uska', 'inki', 'unki',
        'batao', 'bolo', 'samjhao', 'dikhao', 'dikhaen', 'dijiye', 'chahiye',
        'nahi', 'nahin', 'hoga', 'hogi', 'tha', 'thi',
        'kar', 'karo', 'karna', 'ho', 'hona', 'wala', 'wali',
        'ke', 'ki', 'ka', 'se', 'mein', 'par', 'tak',
        'aur', 'ya', 'lekin', 'phir', 'ab', 'jab',
        'keemat', 'qeemat', 'daam', 'sasta', 'sasti', 'mehnga', 'mehngi',
        'kapre', 'kapde', 'joote', 'jootay', 'cheez'
    ]

    text_lower = text.lower()
    # Strip punctuation so "hai?" matches "hai" (a trailing ? is very common
    # and would otherwise drop the match count and misdetect as English).
    words = [w.strip(string.punctuation) for w in text_lower.split()]
    matches = sum(1 for word in words if word in roman_urdu_keywords)

    return 'roman_urdu' if matches >= 2 else 'english'


def detect_full_language(text):
    """
    Complete language detector handling 3 cases:
    1. Urdu script (جوتے دکھائیں) → 'urdu'
    2. Roman Urdu (shoes dikhao / joote kitne ke hain) → 'roman_urdu'
    3. English (Show me shoes) → 'english'

    Priority: Urdu script first, then Roman Urdu, then English.
    Mirrors detect_full_language in medquad_pipeline.
    """
    urdu_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    if urdu_chars > 2:
        return 'urdu'

    if detect_roman_urdu(text) == 'roman_urdu':
        return 'roman_urdu'

    return 'english'


# ============================================================
# SECTION 12 - TRANSLATION FUNCTIONS
# ============================================================
def translate_to_english(text):
    """
    Translates Urdu product query to English before ChromaDB search.
    Flipkart dataset is English-only, so Urdu queries must be
    translated for retrieval to work. Layer 1 of bilingual support.

    Example: "جوتے دکھائیں" → "Show me shoes"
    """
    prompt = f"""Translate this Urdu text to English.
Return ONLY the English translation — nothing else.
Keep product and brand names accurate.

Urdu text: {text}
English translation:"""

    return ask_llm(prompt)


def translate_to_urdu(text):
    """
    Translates English product answer back to Urdu for Urdu users.
    Layer 2 of bilingual support.

    Example: "Nike shoes at Rs.2999..." → "نائیکی جوتے Rs.2999 میں..."
    """
    prompt = f"""Translate this English product/ecommerce text to Urdu.
Return ONLY the Urdu translation — nothing else.
Keep product names, brand names and prices as they are.
Use proper Urdu script.

English text: {text}
Urdu translation:"""

    return ask_llm(prompt)


# ============================================================
# SECTION 13 - FULL RAG PIPELINE
# ============================================================
def rag_pipeline(vector_store, question, search_type="vector",
                 k=8, use_ollama=False):
    """
    Complete end-to-end Ecommerce RAG pipeline with bilingual support.

    Bilingual flow for Urdu questions:
    1. Detect language → Urdu
    2. Translate query to English (Layer 1)
    3. Search ChromaDB with English query
    4. LLM generates English answer from English product data
    5. Translate answer back to Urdu (Layer 2)
    6. User receives clean Urdu answer with product details
    """
    print(f"\nQuestion: {question}")
    print(f"Search: {search_type} | K: {k} | LLM: {'Ollama' if use_ollama else 'Groq'}")
    print("-" * 60)

    # Step 1 - Detect language (Urdu script / Roman Urdu / English)
    language = detect_full_language(question)
    print(f"Language: {language}")

    # Step 2 - Translate Urdu/Roman Urdu query to English for retrieval
    if language in ['urdu', 'roman_urdu']:
        search_query = translate_to_english(question)
        print(f"Translated query: {search_query}")
    else:
        search_query = question

    # Step 3 - Retrieve relevant product chunks using English query
    if search_type == "vector":
        chunks = vector_search(vector_store, search_query, k=k)
    elif search_type == "mmr":
        chunks = mmr_search(vector_store, search_query, k=k)
    elif search_type == "threshold":
        chunks = score_threshold_search(vector_store, search_query, k=k)
    else:
        chunks = vector_search(vector_store, search_query, k=k)

    # Step 4 - Handle no results
    if not chunks:
        print("No relevant products found above threshold.")
        if language == 'urdu':
            return "میرے کیٹالاگ میں اس پروڈکٹ کے بارے میں معلومات نہیں ہے۔"
        elif language == 'roman_urdu':
            return "Is product ke baare mein mujhe catalog mein information nahi mili."
        return "I don't have information about that product in my catalog."

    # Step 5 - Build context from retrieved product chunks
    context = ""
    sources = []

    for i, chunk in enumerate(chunks):
        context += f"Product {i+1}:\n"
        context += f"Name: {chunk.metadata.get('product_name', 'Unknown')}\n"
        context += f"Brand: {chunk.metadata.get('brand', 'Unknown')}\n"
        context += f"Category: {chunk.metadata.get('category', 'Unknown')}\n"
        context += f"Price: Rs.{chunk.metadata.get('price', 'N/A')}\n"
        context += f"Description: {chunk.page_content}\n\n"

        source = chunk.metadata.get('category', 'Unknown')
        if source not in sources:
            sources.append(source)

    # Step 6 - Language-aware instruction for the LLM
    if language == 'urdu':
        lang_line = "Answer in Urdu script only."
    elif language == 'roman_urdu':
        lang_line = ("Answer in Roman Urdu (Urdu written in the English/Latin alphabet), "
                     "matching the style the user used. Do NOT use Urdu script, Hindi, or "
                     "Devanagari. Keep product names, brands and prices in English.")
    else:
        lang_line = "Answer in English only."

    # Build ecommerce prompt (grounding-enforced)
    prompt = f"""You are a Flipkart ecommerce assistant that answers ONLY from the products below.
{lang_line}
Use ONLY the product information below. Do NOT invent products, prices, brands, or specifications.
Include exact product names, brands, and prices from the products when relevant.
If no product below matches the question, reply: "I don't have information about that product in my catalog."
Do not mention product numbers (e.g. "Product 3") in your answer.
At the end mention the product category.

Products:
{context}

Question: {search_query}

Answer:"""

    # Step 7 - Get answer from LLM
    if use_ollama:
        answer = ask_ollama(prompt)
    else:
        answer = ask_llm(prompt)

    # Step 8 - Translate answer to Urdu script only if question was
    # in Urdu script. Roman Urdu already gets a simple-English answer.
    if language == 'urdu':
        print("Translating answer to Urdu...")
        answer = translate_to_urdu(answer)

    print(f"Answer: {answer}")
    print(f"Categories: {sources}")
    print("-" * 60)

    return answer


# ============================================================
# SECTION 14 - QUERY REFORMULATION
# ============================================================
def reformulate_query(question, conversation_history):
    """
    Rewrites vague follow-up product questions into standalone queries.

    Problem: "How much does it cost?" doesn't tell ChromaDB
    which product "it" refers to — retrieves random products.

    Solution: Rewrite using conversation history.
    "How much does it cost?" + cycling shorts history
    → "What is the price of cycling shorts?"

    FIX 1 — Punctuation stripping:
    "What brand is it?" splits into [...,'it?'] and 'it?' != 'it',
    so the vague-word check silently failed. We now strip
    punctuation from each word before comparing.

    Only runs if vague pronouns detected AND history exists.
    """
    vague_words = ['it', 'its', 'they', 'their', 'this',
                   'that', 'these', 'those', 'the product',
                   'the item', 'same']

    question_lower = question.lower()

    # FIX 1 — strip punctuation so "it?" matches "it"
    words = [w.strip(string.punctuation) for w in question_lower.split()]
    needs_reformulation = any(word in words for word in vague_words)

    if not needs_reformulation or not conversation_history:
        return question

    recent_history = conversation_history[-4:]
    history_text = ""
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content'][:200]}\n"

    reformulation_prompt = f"""Given this ecommerce conversation:
{history_text}

Rewrite this follow-up question as a clear standalone product search query.
Remove pronouns like "it", "its", "they", "this" etc.
Replace with the actual product name or category from the conversation.
Return ONLY the rewritten question — nothing else.

Follow-up question: {question}
Rewritten question:"""

    rewritten = ask_llm(reformulation_prompt)
    rewritten = rewritten.strip().strip('"').strip("'")

    print(f"Original:     {question}")
    print(f"Reformulated: {rewritten}")

    return rewritten


# ============================================================
# SECTION 15 - CONVERSATION MEMORY
# ============================================================
conversation_history = []


def chat_with_memory(vector_store, question, search_type="vector", k=8):
    """
    Ecommerce RAG with conversation memory, query reformulation,
    AND full bilingual support (Urdu + English).
    """
    global conversation_history

    print(f"\nUser: {question}")
    print("-" * 60)

    # Step 1 - Detect language (Urdu script / Roman Urdu / English)
    language = detect_full_language(question)

    # Step 2 - Reformulate vague follow-up questions
    search_query = reformulate_query(question, conversation_history)

    # Step 2B - Translate to English if Urdu or Roman Urdu
    if language in ['urdu', 'roman_urdu']:
        search_query = translate_to_english(search_query)
        print(f"Translated search query: {search_query}")

    # Step 3 - Retrieve product chunks using English query
    if search_type == "vector":
        chunks = vector_search(vector_store, search_query, k=k)
    elif search_type == "mmr":
        chunks = mmr_search(vector_store, search_query, k=k)
    elif search_type == "threshold":
        chunks = score_threshold_search(vector_store, search_query, k=k)
    else:
        chunks = vector_search(vector_store, search_query, k=k)

    # Step 4 - Build context from retrieved product chunks
    context = ""
    sources = []

    if chunks:
        for i, chunk in enumerate(chunks):
            context += f"Product {i+1}:\n"
            context += f"Name: {chunk.metadata.get('product_name', 'Unknown')}\n"
            context += f"Brand: {chunk.metadata.get('brand', 'Unknown')}\n"
            context += f"Category: {chunk.metadata.get('category', 'Unknown')}\n"
            context += f"Price: Rs.{chunk.metadata.get('price', 'N/A')}\n"
            context += f"Description: {chunk.page_content}\n\n"

            source = chunk.metadata.get('category', 'Unknown')
            if source not in sources:
                sources.append(source)

    # Step 5 - Language-aware system prompt (grounding-enforced)
    if language == 'urdu':
        lang_line = "Answer in Urdu script only."
    elif language == 'roman_urdu':
        lang_line = ("Answer in Roman Urdu (Urdu written in the English/Latin alphabet), "
                     "matching the style the user used. Do NOT use Urdu script, Hindi, or "
                     "Devanagari. Keep product names, brands and prices in English.")
    else:
        lang_line = "Answer in English only."

    system_prompt = f"""You are a Flipkart ecommerce assistant that answers ONLY from the provided products.
{lang_line}
Use ONLY the product information AND conversation history to answer.
Include exact product names, brands and prices when relevant.
If no product matches the question, reply: "I don't have information about that product in my catalog."
Do not invent products or prices. Do not mention product numbers in your answer.
At the end mention the product category."""

    # Step 6 - Build messages with full conversation history
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)

    current_message = f"""Retrieved Products:
{context}

Current Question: {question}"""
    messages.append({"role": "user", "content": current_message})

    # Step 7 - Get answer from LLM
    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=messages
    )

    answer = _strip_think(response.choices[0].message.content)

    # Step 8 - Translate to Urdu if question was in Urdu
    if language == 'urdu':
        print("Translating answer to Urdu...")
        answer = translate_to_urdu(answer)

    # Step 9 - Store original question in history (not translated)
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    print(f"Assistant: {answer}")
    if sources:
        print(f"Categories: {sources}")
    print(f"Memory: {len(conversation_history)//2} turns stored")
    print("-" * 60)

    return answer


def clear_memory():
    """Resets conversation history for a fresh session."""
    global conversation_history
    conversation_history = []
    print("Conversation memory cleared!")


# ============================================================
# SECTION 16 - INTERACTIVE WHILE LOOP
# ============================================================
def interactive_chat(vector_store, search_type="vector", k=8):
    """
    Interactive chat loop that takes live user input for the
    Flipkart ecommerce bot. Mirrors interactive_chat in
    medquad_pipeline for consistency across both domains.

    Supports English, Urdu script, and Roman Urdu questions.

    Commands:
    - Type any product question → get answer
    - 'clear' → clear conversation memory
    - 'switch vector' → use vector search
    - 'switch mmr' → use MMR search
    - 'switch threshold' → use score threshold search
    - 'quit' or 'exit' → stop the loop
    """
    global conversation_history
    conversation_history = []

    print("\n" + "=" * 60)
    print("FLIPKART ECOMMERCE RAG CHATBOT — INTERACTIVE MODE")
    print("=" * 60)
    print("Ask product questions in English, Urdu, or Roman Urdu")
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

        # ---- DETECT LANGUAGE (Urdu script / Roman Urdu / English) ----
        language = detect_full_language(user_input)
        print(f"[Language: {language}]")

        # ---- QUERY REFORMULATION ----
        # Rewrites vague follow-ups using conversation history
        # "How much does it cost?" → "Price of cycling shorts?"
        search_query = reformulate_query(user_input, conversation_history)

        # ---- TRANSLATE TO ENGLISH FOR SEARCH ----
        # Dataset is English-only — must search in English
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
                answer = "میرے کیٹالاگ میں اس پروڈکٹ کے بارے میں معلومات نہیں ہے۔"
            elif language == 'roman_urdu':
                answer = "Is product ke baare mein mujhe catalog mein information nahi mili."
            else:
                answer = "I don't have information about that product in my catalog."

            print(f"\nBot: {answer}")
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": answer})
            continue

        # ---- BUILD CONTEXT ----
        context = ""
        sources = []

        for i, chunk in enumerate(chunks):
            context += f"Product {i+1}:\n"
            context += f"Name: {chunk.metadata.get('product_name', 'Unknown')}\n"
            context += f"Brand: {chunk.metadata.get('brand', 'Unknown')}\n"
            context += f"Category: {chunk.metadata.get('category', 'Unknown')}\n"
            context += f"Price: Rs.{chunk.metadata.get('price', 'N/A')}\n"
            context += f"Description: {chunk.page_content}\n\n"

            source = chunk.metadata.get('category', 'Unknown')
            if source not in sources:
                sources.append(source)

        # ---- LANGUAGE INSTRUCTION FOR LLM ----
        if language == 'urdu':
            lang_line = "Answer in Urdu script only."
        elif language == 'roman_urdu':
            lang_line = ("Answer in Roman Urdu (Urdu written in the English/Latin alphabet), "
                         "matching the style the user used. Do NOT use Urdu script, Hindi, or "
                         "Devanagari. Keep product names, brands and prices in English.")
        else:
            lang_line = "Answer in English only."

        # ---- BUILD SYSTEM PROMPT (grounding-enforced) ----
        system_prompt = f"""You are a Flipkart ecommerce assistant that answers ONLY from the provided products.
{lang_line}
Use ONLY the product information AND conversation history to answer.
Include exact product names, brands and prices when relevant.
If no product matches the question, reply: "I don't have information about that product in my catalog."
Do not invent products or prices. Do not mention product numbers in your answer.
At the end mention the product category."""

        # ---- BUILD MESSAGES WITH CONVERSATION HISTORY ----
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)

        current_message = f"""Retrieved Products:
{context}

Current Question: {user_input}"""
        messages.append({"role": "user", "content": current_message})

        # ---- GET ANSWER FROM LLM ----
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages
        )

        answer = _strip_think(response.choices[0].message.content)

        # Safety net — empty or echoed answer → honest refusal
        if not answer or answer.strip().lower() == user_input.strip().lower():
            answer = "I don't have information about that product in my catalog."

        # ---- TRANSLATE TO URDU SCRIPT IF NEEDED ----
        if language == 'urdu':
            print("[Translating to Urdu...]")
            answer = translate_to_urdu(answer)

        # ---- STORE IN CONVERSATION HISTORY ----
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": answer})

        # ---- DISPLAY ANSWER ----
        print(f"\nBot: {answer}")
        print(f"Categories: {', '.join(sources)}")
        print(f"Memory: {len(conversation_history)//2} turns")
        print("-" * 60)


# ============================================================
# MAIN - Run the Flipkart pipeline
# ============================================================
if __name__ == "__main__":

    # ---- INGESTION (DONE — keep commented) ----
    # NOTE: chunk_documents() now prepends the product name (FIX 2).
    # To apply to your data, uncomment these 4 lines, point
    # embed_and_store() at a NEW folder (e.g. flipkart_chromadb_v2),
    # run once, then update load_vector_store() to the new folder.
    # df_cleaned = clean_dataset("docs/flipkart_com-ecommerce_sample.csv")
    # documents = load_documents(df_cleaned)
    # chunks = chunk_documents(documents)
    # vector_store = embed_and_store(chunks)

    # ---- LOAD EXISTING DB ----
    # Connects to saved ChromaDB — no re-embedding needed
    vector_store = load_vector_store()

    # ---- START INTERACTIVE CHAT ----
    # User types product questions live — no hardcoded questions
    # Supports English, Urdu script, and Roman Urdu
    interactive_chat(vector_store, search_type="vector", k=8)
