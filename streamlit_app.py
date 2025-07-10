import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import time
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from database import SessionLocal, create_tables
from models import User
from auth import hash_password, verify_password

# === Setup ===
load_dotenv()
create_tables()
api_key = os.getenv("api_key")

st.set_page_config(page_title="RAG Diabetes Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 NHS-Based Diabetes Chatbot")
st.sidebar.info(f"CUDA available: {torch.cuda.is_available()}")

# === Authentication ===
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""

db = next(get_db())

auth_mode = st.sidebar.radio("Login or Signup", ["Login", "Signup"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if auth_mode == "Signup":
    name = st.sidebar.text_input("Name")
    if st.sidebar.button("Create Account"):
        if db.query(User).filter(User.email == email).first():
            st.sidebar.error("❌ Email already registered.")
        else:
            new_user = User(email=email, name=name, hashed_password=hash_password(password))
            db.add(new_user)
            db.commit()
            st.sidebar.success("✅ Account created. Please log in.")
elif auth_mode == "Login":
    if st.sidebar.button("Login"):
        user = db.query(User).filter(User.email == email).first()
        if user and verify_password(password, user.hashed_password):
            st.session_state.logged_in = True
            st.session_state.user_email = user.email
            st.sidebar.success(f"✅ Welcome, {user.name}")
        else:
            st.sidebar.error("❌ Invalid login.")

if not st.session_state.logged_in:
    st.warning("🔐 Please log in to use the chatbot.")
    st.stop()

# === RAG Chatbot Components ===
@st.cache_resource
def load_sentence_transformer():
    with st.spinner("Loading embedding model..."):
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ Embedding model loaded.")
            return model
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None

@st.cache_data
def load_and_chunk_documents(chunk_size=500):
    chunks, sources = [], []
    if not os.path.exists("data"):
        st.error("❌ 'data' folder missing.")
        return [], []

    txt_files = [f for f in os.listdir("data") if f.endswith(".txt")]
    if not txt_files:
        st.error("❌ No .txt files found in 'data'.")
        return [], []

    for filename in txt_files:
        try:
            with open(os.path.join("data", filename), "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                continue
            overlap = chunk_size // 4
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk.strip()) > 50:
                    chunks.append(chunk.strip())
                    sources.append(filename)
        except Exception as e:
            st.error(f"Error reading {filename}: {str(e)}")

    st.success(f"✅ Loaded {len(chunks)} chunks from {len(set(sources))} files.")
    return chunks, sources

@st.cache_resource
def create_faiss_index(_embedder, _chunks):
    if not _chunks:
        return None, None
    try:
        embeddings = _embedder.encode(_chunks, convert_to_numpy=True, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, embeddings
    except Exception as e:
        st.error(f"❌ Error creating FAISS index: {str(e)}")
        return None, None

def get_rag_safety_level(answer, source_chunks):
    try:
        context = " ".join(source_chunks)
        if not context.strip():
            return "🔴 RED – No context available"
        docs = [context, answer]
        vectorizer = TfidfVectorizer().fit_transform(docs)
        sim_score = cosine_similarity([vectorizer.toarray()[1]], [vectorizer.toarray()[0]])[0][0]
        if sim_score > 0.5:
            return f"🟢 GREEN – Safe (similarity: {sim_score:.2f})"
        elif sim_score > 0.2:
            return f"🟠 AMBER – Possibly relevant (similarity: {sim_score:.2f})"
        else:
            return f"🔴 RED – Low similarity (similarity: {sim_score:.2f})"
    except Exception as e:
        return f"🔴 RED – Error calculating similarity: {str(e)}"

def call_groq_llm(system_prompt, user_input, api_key):
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ LLM call failed: {str(e)}")
        return None

def ask_rag_bot(question, api_key, embedder, index, chunks, sources, k=3):
    try:
        q_embedding = embedder.encode([question])[0]
        D, I = index.search(np.array([q_embedding]), k=k)
        top_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
        top_sources = [sources[i] for i in I[0] if i < len(sources)]

        if not top_chunks:
            return "I couldn't find relevant NHS content.", [], []

        context = "\n\n".join(top_chunks)
        system_prompt = f"""You are a helpful NHS-based assistant. Use ONLY the following NHS context to answer:
{context}
Do not invent. If uncertain, say so. Always refer users to real healthcare professionals."""
        answer = call_groq_llm(system_prompt, question, api_key)
        return answer, top_chunks, top_sources
    except Exception as e:
        st.error(f"❌ RAG error: {str(e)}")
        return None, [], []

# === Main Chatbot Logic ===
def main():
    if not api_key:
        st.error("❌ Missing GROQ_API_KEY in .env.")
        return

    embedder = load_sentence_transformer()
    if not embedder:
        return

    chunks, sources = load_and_chunk_documents()
    if not chunks:
        return

    index, embeddings = create_faiss_index(embedder, chunks)
    if not index:
        return

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Documents", len(set(sources)))
    with col2: st.metric("Text Chunks", len(chunks))
    with col3: st.metric("Embedding Dim", embeddings.shape[1])

    st.markdown("---")
    user_input = st.text_area("Ask a question about diabetes:", height=100)
    with st.expander("⚙️ Advanced Options"):
        k = st.slider("Chunks to retrieve", 1, 10, 3)
        show_debug = st.checkbox("Show debug info")

    if st.button("🔍 Ask", disabled=not user_input.strip()):
        start = time.time()
        answer, top_chunks, top_sources = ask_rag_bot(user_input, api_key, embedder, index, chunks, sources, k)
        duration = time.time() - start

        if answer:
            st.markdown("### 💬 Response")
            st.success(answer)

            st.markdown("### 📊 Safety Level")
            safety = get_rag_safety_level(answer, top_chunks)
            if "GREEN" in safety:
                st.success(safety)
            elif "AMBER" in safety:
                st.warning(safety)
            else:
                st.error(safety)

            st.info(f"⏱️ Time: {duration:.2f} seconds")

            with st.expander(f"📚 {len(top_chunks)} Source Chunks"):
                for i, (chunk, source) in enumerate(zip(top_chunks, top_sources)):
                    st.markdown(f"**{i+1}. `{source}`**")
                    st.text_area("", chunk, height=100, key=f"chunk_{i}")

            if show_debug:
                with st.expander("🐛 Debug"):
                    st.json({
                        "question_length": len(user_input),
                        "chunks_retrieved": len(top_chunks),
                        "sources": list(set(top_sources)),
                        "processing_time": round(duration, 2)
                    })

if __name__ == "__main__":
    main()
