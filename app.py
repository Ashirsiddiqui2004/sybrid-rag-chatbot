# ============================================================
# STREAMLIT APP — Dual Domain RAG Chatbot
# Sybrid Internship 2026 | Muhammad Ashir Siddiqui
# ============================================================
# Single chat interface supporting two domains:
# - Medical: MedQuAD Q&A dataset (NIH)
# - Ecommerce: Flipkart Products dataset
#
# Features:
# - Switch between domains from sidebar
# - Urdu + Roman Urdu + English bilingual support
# - 3 retrieval methods (vector, MMR, threshold)
# - K value slider
# - Groq cloud or Ollama local LLM
# - Conversation memory per domain
# - PDF file upload
# - Source citations
#
# CHANGELOG (grounding fixes):
# - FIX 1: System prompts now FORBID training-data fallback.
#   The bot must answer ONLY from retrieved documents and say
#   "I don't have information on that" when they don't contain
#   the answer. Removed "answer as best you can" language.
# - FIX 2: Imports detect_full_language (Urdu + Roman Urdu),
#   not detect_language (Urdu-script only). Roman Urdu now
#   handled correctly.
# - FIX 3: K slider defaults to 8 (matches tuned pipeline).
# - FIX 4: "Do not mention document numbers" added so answers
#   read naturally.
# ============================================================

import streamlit as st
import os
import tempfile
import sys

# Add project folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from both pipelines
# FIX 2 — import detect_full_language (handles Roman Urdu too),
# and keep detect_roman_urdu available if needed
from medquad_pipeline import (
    load_vector_store as load_medical_vs,
    detect_full_language,
    translate_to_english,
    translate_to_urdu,
    translate_to_roman_urdu,
    vector_search as medical_vector_search,
    mmr_search as medical_mmr_search,
    score_threshold_search as medical_threshold_search,
    ask_llm,
    ask_ollama,
    client,
    reformulate_query as medical_reformulate
)

from flipkart_pipeline import (
    load_vector_store as load_ecommerce_vs,
    vector_search as ecommerce_vector_search,
    mmr_search as ecommerce_mmr_search,
    score_threshold_search as ecommerce_threshold_search,
    reformulate_query as ecommerce_reformulate
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="RAG Chatbot — Medical & Ecommerce",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# LOAD BOTH VECTOR STORES (cached)
# ============================================================
@st.cache_resource
def get_medical_store():
    """Load MedQuAD ChromaDB — cached so loads only once."""
    return load_medical_vs()

@st.cache_resource
def get_ecommerce_store():
    """Load Flipkart ChromaDB — cached so loads only once."""
    return load_ecommerce_vs()

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
# Separate message history per domain so switching domains
# doesn't mix up medical and ecommerce conversations
if "medical_messages" not in st.session_state:
    st.session_state.medical_messages = []

if "ecommerce_messages" not in st.session_state:
    st.session_state.ecommerce_messages = []

# Separate LLM history per domain for accurate memory
if "medical_llm_history" not in st.session_state:
    st.session_state.medical_llm_history = []

if "ecommerce_llm_history" not in st.session_state:
    st.session_state.ecommerce_llm_history = []

# Track uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    # Domain selector — the main switch between Medical and Ecommerce
    domain = st.radio(
        "🌐 Select Domain",
        options=["Medical (MedQuAD)", "Ecommerce (Flipkart)"],
        help="Medical: answers health questions from NIH dataset\nEcommerce: answers product questions from Flipkart"
    )

    st.markdown("---")

    # Retrieval method
    search_type = st.selectbox(
        "Retrieval Method",
        options=["vector", "mmr", "threshold"],
        help="vector: basic semantic search | mmr: diverse results | threshold: filtered search"
    )

    # K value slider — FIX 3: default 8 to match tuned pipeline
    k_value = st.slider(
        "Number of Chunks (K)",
        min_value=1,
        max_value=20,
        value=8,
        help="How many document chunks to retrieve per question. "
             "Higher K reduces over-refusals (pulls in answer chunks that "
             "rank below question chunks) but uses more tokens per query."
    )

    # LLM selector
    llm_choice = st.radio(
        "LLM Model",
        options=["Groq (Qwen3-32B)", "Ollama (Llama3.2 Local)"],
        help="Groq: fast cloud model | Ollama: offline local model"
    )

    # Show retrieved chunks toggle — for grounding verification
    show_chunks = st.checkbox(
        "🔍 Show retrieved chunks",
        value=False,
        help="Displays the exact document chunks given to the LLM — useful for verifying answers are grounded"
    )

    st.markdown("---")

    # Clear conversation button — clears only current domain
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        if "Medical" in domain:
            st.session_state.medical_messages = []
            st.session_state.medical_llm_history = []
        else:
            st.session_state.ecommerce_messages = []
            st.session_state.ecommerce_llm_history = []
        st.rerun()

    st.markdown("---")

    # Dataset info — changes based on selected domain
    if "Medical" in domain:
        st.markdown("### 📊 Dataset Info")
        st.markdown("**MedQuAD** — Medical Q&A")
        st.markdown("- 14,979 cleaned Q&A pairs")
        st.markdown("- 18,899 chunks in ChromaDB")
        st.markdown("- Source: NIH")
    else:
        st.markdown("### 📊 Dataset Info")
        st.markdown("**Flipkart** — Ecommerce Products")
        st.markdown("- 12,625 cleaned products")
        st.markdown("- 63,472 chunks in ChromaDB")
        st.markdown("- Source: Flipkart.com")

    st.markdown("---")
    st.markdown("### 🌐 Language Support")
    st.markdown("Ask in **Urdu**, **Roman Urdu**, or **English**")
    st.markdown("Bot replies in same language")

    st.markdown("---")

    # PDF Upload section
    st.markdown("### 📄 Upload Your Document")
    st.markdown("Upload a PDF to add to knowledge base")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="PDF will be added to the currently selected domain"
    )

    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner("Processing PDF..."):

                # Save to temp file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Load and chunk the PDF
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", " ", ""]
                )
                new_chunks = splitter.split_documents(pages)

                # Add to the currently selected domain's vector store
                if "Medical" in domain:
                    vs = get_medical_store()
                else:
                    vs = get_ecommerce_store()

                vs.add_documents(new_chunks)
                os.remove(tmp_path)

                st.session_state.uploaded_files.append(uploaded_file.name)

            st.success(f"✅ Added {len(new_chunks)} chunks from {uploaded_file.name}")
        else:
            st.info(f"✅ {uploaded_file.name} already loaded")

    if st.session_state.uploaded_files:
        st.markdown("**Uploaded files:**")
        for fname in st.session_state.uploaded_files:
            st.caption(f"📄 {fname}")

# ============================================================
# MAIN CHAT AREA
# ============================================================
# Title and description change based on selected domain
if "Medical" in domain:
    st.title("🏥 Medical RAG Chatbot")
    st.markdown("Ask any **medical question** in Urdu or English. Answers grounded in NIH medical documents.")
    active_messages = st.session_state.medical_messages
    active_llm_history = st.session_state.medical_llm_history
    vector_store = get_medical_store()
    input_placeholder = "Ask a medical question in Urdu or English..."
else:
    st.title("🛒 Ecommerce RAG Chatbot")
    st.markdown("Ask any **product question** in Urdu or English. Answers grounded in Flipkart product catalog.")
    active_messages = st.session_state.ecommerce_messages
    active_llm_history = st.session_state.ecommerce_llm_history
    vector_store = get_ecommerce_store()
    input_placeholder = "Ask about products, prices, brands in Urdu or English..."

st.markdown("---")

# Display conversation history
for message in active_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.caption(f"📚 Sources: {', '.join(message['sources'])}")

# ============================================================
# HANDLE USER INPUT
# ============================================================
if prompt := st.chat_input(input_placeholder):

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    active_messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching documents and generating answer..."):

        # Step 1 - Detect language (FIX 2: full detector, Roman Urdu aware)
        language = detect_full_language(prompt)

        # Step 2 - Reformulate vague follow-up questions
        if "Medical" in domain:
            search_query = medical_reformulate(prompt, active_llm_history)
        else:
            search_query = ecommerce_reformulate(prompt, active_llm_history)

        # Step 3 - Translate to English if Urdu or Roman Urdu
        # (dataset is English — must search in English)
        if language in ['urdu', 'roman_urdu']:
            search_query = translate_to_english(search_query)

        # Step 4 - Retrieve chunks from correct domain
        if "Medical" in domain:
            if search_type == "vector":
                chunks = medical_vector_search(vector_store, search_query, k=k_value)
            elif search_type == "mmr":
                chunks = medical_mmr_search(vector_store, search_query, k=k_value)
            else:
                chunks = medical_threshold_search(vector_store, search_query, k=k_value)
        else:
            if search_type == "vector":
                chunks = ecommerce_vector_search(vector_store, search_query, k=k_value)
            elif search_type == "mmr":
                chunks = ecommerce_mmr_search(vector_store, search_query, k=k_value)
            else:
                chunks = ecommerce_threshold_search(vector_store, search_query, k=k_value)

        # Step 5 - Handle no results
        if not chunks:
            if language == 'urdu':
                answer = "میرے دستاویزات میں اس بارے میں معلومات نہیں ہے۔"
            elif language == 'roman_urdu':
                answer = "Is baare mein mujhe information nahi mili documents mein."
            else:
                answer = "I don't have information on that in my documents."
            sources = []
        else:
            # Step 6 - Build context
            context = ""
            sources = []

            for i, chunk in enumerate(chunks):
                if "Medical" in domain:
                    context += f"Document {i+1}:\n"
                    context += f"Topic: {chunk.metadata.get('focus_area', 'Unknown')}\n"
                    context += f"Content: {chunk.page_content}\n\n"
                    source = chunk.metadata.get('focus_area', 'Unknown')
                else:
                    context += f"Product {i+1}:\n"
                    context += f"Name: {chunk.metadata.get('product_name', 'Unknown')}\n"
                    context += f"Brand: {chunk.metadata.get('brand', 'Unknown')}\n"
                    context += f"Category: {chunk.metadata.get('category', 'Unknown')}\n"
                    context += f"Price: Rs.{chunk.metadata.get('price', 'N/A')}\n"
                    context += f"Content: {chunk.page_content}\n\n"
                    source = chunk.metadata.get('category', 'Unknown')

                if source not in sources:
                    sources.append(source)

            # Step 7 - Build domain-specific prompt
            # FIX 1 — STRICT GROUNDING. The model must NOT use training
            # knowledge. If the documents don't contain the answer, it
            # must say so. This is what stops the "based on general
            # medical knowledge" hallucinations.
            # Language instruction handled here too.
            if language == 'urdu':
                lang_line = "Answer in Urdu script only. Do not use English."
            elif language == 'roman_urdu':
                # Generate in English first (reliable), then translate the
                # finished answer to Roman Urdu in a separate step below.
                lang_line = "Answer in English only."
            else:
                lang_line = "Answer in English only."

            if "Medical" in domain:
                system_prompt = f"""You are a medical assistant that answers ONLY from the provided documents.
{lang_line}

STRICT RULES:
- Use ONLY the information in the documents below. Do NOT use any outside or general medical knowledge.
- If the documents do not contain the answer, reply EXACTLY: "I don't have information on that in my medical documents." Do not add anything from general knowledge after that.
- Never write phrases like "based on general medical knowledge" or "current medical knowledge indicates".
- Do not mention document numbers (e.g. "Document 3") in your answer. Write naturally.
- When you DO have an answer, mention the medical topic at the very end. If you do NOT have the information, reply with ONLY the exact refusal sentence and nothing else — no medical topic line, no explanation.
- Do not show your reasoning. Give only the final answer.
"""
            else:
                system_prompt = f"""You are a Flipkart ecommerce assistant that answers ONLY from the provided product documents.
{lang_line}

STRICT RULES:
- Use ONLY the product information below. Do NOT invent products, prices, brands, or specifications.
- Include exact product names, brands, and prices from the documents when relevant.
- If no product below matches the question, reply EXACTLY: "I don't have information on that in my product catalog." Do not guess.
- Never use outside knowledge about products not in the documents.
- Do not mention product/document numbers (e.g. "Product 3") in your answer. Write naturally.
- When you DO have an answer, mention the product category at the very end. If you do NOT have the information, reply with ONLY the exact refusal sentence and nothing else.
- Do not show your reasoning. Give only the final answer.
"""

            # Step 8 - Build messages with conversation history
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(active_llm_history)

            # A final language reminder for Urdu script (which generates
            # directly in Urdu). Roman Urdu generates in English then gets
            # translated afterwards, so no reminder needed for it.
            if language == 'urdu':
                reminder = "\n\nREMEMBER: Reply in Urdu script only."
            else:
                reminder = ""

            current_message = f"""Retrieved Documents:
{context}

Current Question: {prompt}{reminder}"""
            messages.append({"role": "user", "content": current_message})

            # Step 9 - Get answer from selected LLM
            if "Ollama" in llm_choice:
                ollama_prompt = f"""{system_prompt}

Documents:
{context}

Question: {prompt}
Answer:"""
                answer = ask_ollama(ollama_prompt)
            else:
                response = client.chat.completions.create(
                    model="qwen/qwen3-32b",
                    messages=messages
                )
                answer = response.choices[0].message.content
                # Strip Qwen3 reasoning block — handle BOTH closed and
                # unclosed <think> (model can hit token limit mid-thought
                # and never emit </think>, which would leak the monologue)
                if "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()
                elif "<think>" in answer:
                    # Opened but never closed — drop everything after <think>
                    answer = answer.split("<think>")[0].strip()

                # SAFETY NET — if stripping left nothing (or the model just
                # echoed the question / reasoned itself into empty output),
                # fall back to an honest refusal instead of a blank/echoed reply
                if not answer or answer.strip().lower() == prompt.strip().lower():
                    if "Medical" in domain:
                        answer = "I don't have information on that in my medical documents."
                    else:
                        answer = "I don't have information about that product in my catalog."

            # Step 10 - Translate the finished answer to the user's language.
            # Uses a dedicated variable so nothing downstream can revert it.
            refusal_markers = ["I don't have information"]
            is_refusal = any(m in answer for m in refusal_markers)

            try:
                if is_refusal:
                    display_answer = answer
                elif language == 'urdu':
                    display_answer = translate_to_urdu(answer)
                elif language == 'roman_urdu':
                    display_answer = translate_to_roman_urdu(answer)
                else:
                    display_answer = answer
            except Exception:
                # If translation fails (e.g. a rate limit), fall back to the
                # English answer rather than crashing the app.
                display_answer = answer

            # From here on, the user-facing text is display_answer.
            answer = display_answer

        # Step 11 - Update LLM history
        active_llm_history.append({"role": "user", "content": prompt})
        active_llm_history.append({"role": "assistant", "content": answer})

        # Update session state
        if "Medical" in domain:
            st.session_state.medical_llm_history = active_llm_history
        else:
            st.session_state.ecommerce_llm_history = active_llm_history

    # Step 12 - Display answer
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            st.caption(f"📚 Sources: {', '.join(sources)}")

        # Grounding verification — show the exact chunks fed to the LLM
        if show_chunks and chunks:
            with st.expander("🔍 Retrieved chunks (what the LLM was given)"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}** — {sources[i] if i < len(sources) else ''}")
                    st.text(chunk.page_content[:500])
                    st.markdown("---")

    active_messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })

    # Update messages in session state
    if "Medical" in domain:
        st.session_state.medical_messages = active_messages
    else:
        st.session_state.ecommerce_messages = active_messages
