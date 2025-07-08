import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file

api_key = os.getenv("api_key")


# Load and chunk NHS documents
chunk_size = 500
chunks = []
sources = []

for filename in os.listdir("data"):
    if filename.endswith(".txt"):
        with open(os.path.join("data", filename), "r", encoding="utf-8") as f:
            text = f.read()
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i + chunk_size])
                sources.append(filename)

if len(chunks) == 0:
    st.error("No valid text chunks found. Make sure your .txt files in 'data/' are not empty.")
    st.stop()

# Embed and index
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks, convert_to_numpy=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Similarity-based RAG safety scoring
def get_rag_safety_level(answer, source_chunks):
    context_text = " ".join(source_chunks)
    docs = [context_text, answer]
    vectorizer = TfidfVectorizer().fit_transform(docs)
    vectors = vectorizer.toarray()
    sim_score = cosine_similarity([vectors[1]], [vectors[0]])[0][0]
    if sim_score > 0.5:
        return "🟢 GREEN – Safe"
    elif sim_score > 0.2:
        return "🟠 AMBER – Possibly correct but unverified"
    else:
        return "🔴 RED – Unsupported or risky"

# RAG-based chatbot function
def ask_rag_bot(question):
    q_embedding = embedder.encode([question])[0]
    D, I = index.search(np.array([q_embedding]), k=3)
    top_chunks = [chunks[i] for i in I[0]]
    top_sources = [sources[i] for i in I[0]]
    context = "\n\n".join(top_chunks)

    response = ollama.chat(
        model='mistral',
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful NHS-based assistant. Use the following NHS guidance to answer safely:\n\n{context}"
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response['message']['content'], top_chunks, top_sources

# Streamlit app
st.set_page_config(page_title="RAG Diabetes Chatbot", page_icon="🤖")
st.title("🤖 NHS-Based Diabetes Chatbot")

user_input = st.text_input("Ask a question related to diabetes management:", "")

if user_input:
    answer, top_chunks, top_sources = ask_rag_bot(user_input)
    rating = get_rag_safety_level(answer, top_chunks)

    st.markdown("### 💬 Bot Response:")
    st.success(answer)

    st.markdown("### 📊 RAG Safety Level:")
    st.info(rating)

    with st.expander("📚 View Supporting NHS Sources"):
        for i, chunk in enumerate(top_chunks):
            st.markdown(f"**Chunk {i+1} from `{top_sources[i]}`**")
            st.code(chunk.strip()[:500] + ("..." if len(chunk) > 500 else ""))
