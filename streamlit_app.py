import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from passlib.hash import bcrypt
import os

# === Load Environment ===
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
mongo_url = os.getenv("MONGO_URL") or st.secrets["MONGO_URL"]

# === MongoDB Connection ===
client = MongoClient(mongo_url)
db = client["chatbot_db"]
users_col = db["users"]

# === Streamlit UI Setup ===
st.set_page_config(page_title="RAG Diabetes Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 NHS-Based Diabetes Chatbot")

# === Auth ===
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""

auth_mode = st.sidebar.radio("Login or Signup", ["Login", "Signup"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if auth_mode == "Signup":
    name = st.sidebar.text_input("Name")
    if st.sidebar.button("Create Account"):
        if users_col.find_one({"email": email}):
            st.sidebar.error("❌ Email already registered.")
        else:
            hashed = bcrypt.hash(password)
            users_col.insert_one({"email": email, "name": name, "password": hashed})
            st.sidebar.success("✅ Account created. Please log in.")
elif auth_mode == "Login":
    if st.sidebar.button("Login"):
        user = users_col.find_one({"email": email})
        if user and bcrypt.verify(password, user["password"]):
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.sidebar.success(f"✅ Welcome, {user['name']}")
        else:
            st.sidebar.error("❌ Invalid login.")

if not st.session_state.logged_in:
    st.warning("🔐 Please log in to use the chatbot.")
    st.stop()

# === RAG Setup ===
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_data
def load_documents(chunk_size=500):
    chunks, sources = [], []
    if not os.path.exists("data"):
        return [], []
    txt_files = [f for f in os.listdir("data") if f.endswith(".txt")]
    for filename in txt_files:
        with open(os.path.join("data", filename), "r", encoding="utf-8") as f:
            text = f.read().strip()
        overlap = chunk_size // 4
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
                sources.append(filename)
    return chunks, sources

@st.cache_resource
def create_faiss_index(_embedder, _chunks):
    embeddings = _embedder.encode(_chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def get_rag_safety(answer, context):
    docs = [context, answer]
    vectorizer = TfidfVectorizer().fit_transform(docs)
    score = cosine_similarity([vectorizer.toarray()[1]], [vectorizer.toarray()[0]])[0][0]
    if score > 0.5:
        return f"🟢 GREEN (similarity: {score:.2f})"
    elif score > 0.2:
        return f"🟡 AMBER (similarity: {score:.2f})"
    else:
        return f"🔴 RED (similarity: {score:.2f})"

def call_groq(system_prompt, user_input):
    client = Groq(api_key=api_key)
    result = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    return result.choices[0].message.content

def ask_bot(question, embedder, index, chunks, sources, k=3):
    q_embed = embedder.encode([question])[0]
    _, I = index.search(np.array([q_embed]), k=k)
    top_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    top_sources = [sources[i] for i in I[0] if i < len(sources)]
    context = "\n\n".join(top_chunks)
    prompt = f"""You are an NHS-based assistant. Only use this NHS guidance to answer:
{context}
Be honest. If unsure, say so and recommend a doctor."""
    answer = call_groq(prompt, question)
    return answer, top_chunks, top_sources

# === Main App ===
def main():
    embedder = load_model()
    chunks, sources = load_documents()
    if not chunks:
        st.error("No NHS documents found.")
        return

    index, embeddings = create_faiss_index(embedder, chunks)

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
        answer, top_chunks, top_sources = ask_bot(user_input, embedder, index, chunks, sources, k)
        duration = time.time() - start

        st.markdown("### 🖊️ Response")
        st.success(answer)

        rag_score = get_rag_safety(answer, " ".join(top_chunks))
        st.markdown("### 📊 Safety Level")
        if "GREEN" in rag_score:
            st.success(rag_score)
        elif "AMBER" in rag_score:
            st.warning(rag_score)
        else:
            st.error(rag_score)

        st.info(f"⏱️ Time: {duration:.2f} seconds")

        with st.expander(f"📚 Sources ({len(top_chunks)} chunks)"):
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
