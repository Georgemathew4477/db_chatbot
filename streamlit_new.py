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

# Configure page
st.set_page_config(page_title="RAG Diabetes Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 NHS-Based Diabetes Chatbot")

# Display CUDA info
st.sidebar.info(f"CUDA available: {torch.cuda.is_available()}")

@st.cache_resource
def load_sentence_transformer():
    """Load and cache the sentence transformer model"""
    with st.spinner("Loading embedding model... (this may take a few minutes on first run)"):
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ Embedding model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Failed to load embedding model: {str(e)}")
            return None

@st.cache_data
def load_and_chunk_documents(chunk_size=500):
    """Load and chunk NHS documents with caching"""
    chunks = []
    sources = []
    
    if not os.path.exists("data"):
        st.error("❌ 'data' directory not found. Please create it and add your .txt files.")
        return [], []
    
    txt_files = [f for f in os.listdir("data") if f.endswith(".txt")]
    
    if not txt_files:
        st.error("❌ No .txt files found in 'data' directory.")
        return [], []
    
    with st.spinner(f"Loading and chunking {len(txt_files)} documents..."):
        for filename in txt_files:
            try:
                file_path = os.path.join("data", filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    
                if not text:
                    st.warning(f"⚠️ File {filename} is empty, skipping...")
                    continue
                    
                # Create overlapping chunks for better context
                overlap = chunk_size // 4
                for i in range(0, len(text), chunk_size - overlap):
                    chunk = text[i:i + chunk_size]
                    if len(chunk.strip()) > 50:  # Only keep meaningful chunks
                        chunks.append(chunk.strip())
                        sources.append(filename)
                        
            except Exception as e:
                st.error(f"❌ Error reading {filename}: {str(e)}")
    
    st.success(f"✅ Loaded {len(chunks)} chunks from {len(set(sources))} documents")
    return chunks, sources

@st.cache_resource
def create_faiss_index(_embedder, _chunks):
    """Create and cache FAISS index"""
    if not _chunks:
        return None, None
        
    with st.spinner("Creating embeddings and FAISS index..."):
        try:
            embeddings = _embedder.encode(_chunks, convert_to_numpy=True, show_progress_bar=True)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            st.success(f"✅ FAISS index created with {len(_chunks)} embeddings")
            return index, embeddings
        except Exception as e:
            st.error(f"❌ Error creating FAISS index: {str(e)}")
            return None, None

def get_rag_safety_level(answer, source_chunks):
    """Calculate RAG safety level using TF-IDF similarity"""
    try:
        if not source_chunks or not answer:
            return "🔴 RED – No context available"
            
        context_text = " ".join(source_chunks)
        if len(context_text.strip()) < 10:
            return "🔴 RED – Insufficient context"
            
        docs = [context_text, answer]
        vectorizer = TfidfVectorizer().fit_transform(docs)
        vectors = vectorizer.toarray()
        sim_score = cosine_similarity([vectors[1]], [vectors[0]])[0][0]
        
        if sim_score > 0.5:
            return f"🟢 GREEN – Safe (similarity: {sim_score:.2f})"
        elif sim_score > 0.2:
            return f"🟠 AMBER – Possibly correct but unverified (similarity: {sim_score:.2f})"
        else:
            return f"🔴 RED – Unsupported or risky (similarity: {sim_score:.2f})"
    except Exception as e:
        return f"🔴 RED – Error calculating safety: {str(e)}"

def call_groq_llm(system_prompt, user_input, api_key):
    """Call Groq LLM with error handling"""
    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1,  # Lower temperature for more consistent medical advice
            max_tokens=1000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Error calling Groq API: {str(e)}")
        return None

def ask_rag_bot(question, api_key, embedder, index, chunks, sources, k=3):
    """RAG-based chatbot function with error handling"""
    try:
        # Generate query embedding
        q_embedding = embedder.encode([question])[0]
        
        # Search for similar chunks
        D, I = index.search(np.array([q_embedding]), k=k)
        
        # Get top chunks and sources
        top_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
        top_sources = [sources[i] for i in I[0] if i < len(sources)]
        
        if not top_chunks:
            return "I couldn't find relevant information in the NHS documents.", [], []
        
        # Create context
        context = "\n\n".join(top_chunks)
        
        # Enhanced system prompt
        system_prompt = f"""You are a helpful NHS-based medical assistant. Use ONLY the following NHS guidance to answer the user's question safely and accurately. 

Important guidelines:
- Base your answer strictly on the provided NHS context
- If the context doesn't contain enough information, say so clearly
- Always recommend consulting healthcare professionals for personalized advice
- Never provide medical advice beyond what's in the NHS guidance

NHS Context:
{context}

Remember: This is general guidance only. Always consult healthcare professionals for personalized medical advice."""
        
        # Get LLM response
        answer = call_groq_llm(system_prompt, question, api_key)
        
        return answer, top_chunks, top_sources
        
    except Exception as e:
        st.error(f"❌ Error in RAG processing: {str(e)}")
        return None, [], []

# Main app logic
def main():
    # Load model and data
    embedder = load_sentence_transformer()
    if embedder is None:
        st.stop()
    
    chunks, sources = load_and_chunk_documents()
    if not chunks:
        st.stop()
    
    index, embeddings = create_faiss_index(embedder, chunks)
    if index is None:
        st.stop()
    
    # Display stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", len(set(sources)))
    with col2:
        st.metric("Text Chunks", len(chunks))
    with col3:
        st.metric("Embedding Dimension", embeddings.shape[1] if embeddings is not None else 0)
    
    # User interface
    st.markdown("---")
    
    # API Key input
    api_key = st.text_input(
        "Enter your Groq API Key:", 
        type="password", 
        help="Get your API key from https://console.groq.com/keys"
    )
    
    # Question input
    user_input = st.text_area(
        "Ask a question about diabetes management:",
        placeholder="e.g., What are the symptoms of type 2 diabetes?",
        height=100
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        k_value = st.slider("Number of source chunks to retrieve:", 1, 10, 3)
        show_debug = st.checkbox("Show debug information")
    
    # Process query
    if st.button("🔍 Ask Question", disabled=not (user_input and api_key)):
        if not user_input.strip():
            st.warning("Please enter a question.")
            return
        
        if not api_key.strip():
            st.warning("Please enter your Groq API key.")
            return
        
        start_time = time.time()
        
        with st.spinner("Searching NHS documents and generating response..."):
            answer, top_chunks, top_sources = ask_rag_bot(
                user_input, api_key, embedder, index, chunks, sources, k_value
            )
        
        processing_time = time.time() - start_time
        
        if answer:
            # Display response
            st.markdown("### 💬 Bot Response:")
            st.success(answer)
            
            # Safety rating
            st.markdown("### 📊 RAG Safety Level:")
            rating = get_rag_safety_level(answer, top_chunks)
            
            if "GREEN" in rating:
                st.success(rating)
            elif "AMBER" in rating:
                st.warning(rating)
            else:
                st.error(rating)
            
            # Processing time
            st.info(f"⏱️ Response generated in {processing_time:.2f} seconds")
            
            # Show sources
            if top_chunks:
                with st.expander(f"📚 View {len(top_chunks)} Supporting NHS Sources"):
                    for i, (chunk, source) in enumerate(zip(top_chunks, top_sources)):
                        st.markdown(f"**Source {i+1}: `{source}`**")
                        st.text_area(
                            f"Content {i+1}:", 
                            chunk[:500] + ("..." if len(chunk) > 500 else ""),
                            height=100,
                            key=f"source_{i}"
                        )
            
            # Debug info
            if show_debug:
                with st.expander("🐛 Debug Information"):
                    st.json({
                        "question_length": len(user_input),
                        "chunks_retrieved": len(top_chunks),
                        "total_context_length": sum(len(chunk) for chunk in top_chunks),
                        "unique_sources": list(set(top_sources)),
                        "processing_time_seconds": round(processing_time, 2)
                    })

if __name__ == "__main__":
    main()