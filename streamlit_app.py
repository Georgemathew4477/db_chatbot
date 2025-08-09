import os
import time
import faiss
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
from passlib.hash import bcrypt
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bson.objectid import ObjectId

from utils.reminders import add_reminder, load_user_reminders
from utils.glucose import store_glucose_reading, get_latest_glucose

# Environment & Connections
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
mongo_url = os.getenv("MONGO_URL") or st.secrets["MONGO_URL"]

client = MongoClient(mongo_url)
db = client["chatbot_db"]
users_col = db["users"]
history_col = db["history"]



# Streamlit Base Setup

st.set_page_config(page_title="RAG Diabetes Chatbot", page_icon="🤖", layout="wide")

def load_css(path: str = "styles.css"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

def get_user_history(email: str, limit: int = 30):
    return list(
        history_col.find({"email": email})
                   .sort("timestamp", -1)
                   .limit(limit)
    )

def clear_user_history(email: str):
    history_col.delete_many({"email": email})




# Session State

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.user_preferences = {}



# Top Bar (Title + Page Menu)

st.markdown('<div class="topbar">', unsafe_allow_html=True)
left, right = st.columns([6, 2])
with left:
    st.title("🤖 NHS-Based Diabetes Chatbot")
with right:
    page = st.selectbox(
        "Page selector",  # Accessible label (not empty)
        ["Chatbot", "User Profile", "Health Tools"],
        index=0,
        label_visibility="collapsed"  # Hidden visually but keeps accessibility
    )
st.markdown('</div>', unsafe_allow_html=True)





# Sidebar (Auth only, fixed bottom)

with st.sidebar:
    # ===== Chat history (only when logged in) =====
    if st.session_state.logged_in:
        st.markdown("### 🕘 Recent chats")
        hist_items = get_user_history(st.session_state.user_email, limit=30)

        if not hist_items:
            st.caption("No chats yet.")
        else:
            # simple filter (optional)
            q_filter = st.text_input("Filter", placeholder="Search history…", key="hist_filter")
            for h in hist_items:
                q = (h.get("question") or "").strip()
                if q_filter and q_filter.lower() not in q.lower():
                    continue
                label = (q[:60] + "…") if len(q) > 60 else q or "(empty)"
                if st.button(label, key=str(h["_id"])):
                    # Prefill the input and navigate to Chatbot
                    st.session_state["user_input"] = q
                    st.session_state["page"] = "Chatbot"
                    st.rerun()

            cols = st.columns(2)
            with cols[0]:
                if st.button("🧹 Clear history"):
                    clear_user_history(st.session_state.user_email)
                    st.rerun()
            with cols[1]:
                st.caption(f"{len(hist_items)} shown")

        st.divider()

    # ===== Bottom-fixed auth =====
    st.markdown('<div class="sidebar-bottom">', unsafe_allow_html=True)

    if not st.session_state.logged_in:
        st.markdown("### 🔐 Login or Signup")
        auth_mode = st.radio("Mode", ["Login", "Signup"], horizontal=True, key="auth_mode")
        email = st.text_input("Email", key="email_login")
        password = st.text_input("Password", type="password", key="pwd_login")

        if auth_mode == "Signup":
            name = st.text_input("Name", key="name_signup")
            age = st.number_input("Age", min_value=0, max_value=120, step=1, key="age_signup")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender_signup")
            diabetes_type = st.selectbox("Diabetes Type", ["Type 1", "Type 2", "Gestational", "None"], key="dtype_signup")
            if st.button("Create Account"):
                if users_col.find_one({"email": email}):
                    st.error("❌ Email already registered.")
                else:
                    hashed = bcrypt.hash(password)
                    users_col.insert_one({
                        "email": email, "name": name, "password": hashed,
                        "age": age, "gender": gender, "diabetes_type": diabetes_type
                    })
                    st.success("✅ Account created. Please log in.")
        else:
            if st.button("Login"):
                user = users_col.find_one({"email": email})
                if user and bcrypt.verify(password, user["password"]):
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.session_state.user_preferences = {
                        "name": user.get("name", ""),
                        "age": user.get("age"),
                        "gender": user.get("gender"),
                        "diabetes_type": user.get("diabetes_type")
                    }
                    load_user_reminders(email)
                    st.rerun()
                else:
                    st.error("❌ Invalid login.")
    else:
        st.success(f"👋 Welcome, {st.session_state.user_preferences.get('name','').capitalize()}")
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.session_state.user_preferences = {}
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)




# RAG Helpers

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

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
    arr = vectorizer.toarray()
    score = cosine_similarity([arr[1]], [arr[0]])[0][0]

    green_threshold = 0.5
    amber_threshold = 0.2

    if score > green_threshold:
        return (
            f"🟢 GREEN (similarity: {score:.2f}) — "
            f"Strong support from NHS guidelines.\n"
            f"(Score > {green_threshold:.2f} is considered GREEN)"
        )
    elif score > amber_threshold:
        return (
            f"🟡 AMBER (similarity: {score:.2f}) — "
            f"Partial support from NHS guidelines. Use caution.\n"
            f"({amber_threshold:.2f} < Score ≤ {green_threshold:.2f} is considered AMBER)"
        )
    else:
        return (
            f"🔴 RED (similarity: {score:.2f}) — "
            f"Weak support from NHS guidelines. Please confirm with a healthcare professional.\n"
            f"(Score ≤ {amber_threshold:.2f} is considered RED)"
        )


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

def ask_bot(question, embedder, index, chunks, sources, preferences, k=5):
    q_embed = embedder.encode([question])[0]
    _, I = index.search(np.array([q_embed]), k=k)
    top_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    top_sources = [sources[i] for i in I[0] if i < len(sources)]

    context = "\n\n".join(top_chunks)

    # Personalization / latest glucose (if any)
    glucose = get_latest_glucose(st.session_state.user_email)
    glucose_line = f"Latest Glucose: {glucose} mg/dL\n" if glucose else ""
    user_info = (
        f"User Info: Age: {preferences.get('age')}, "
        f"Gender: {preferences.get('gender')}, "
        f"Diabetes Type: {preferences.get('diabetes_type')}"
    )
    context += f"\n\n{glucose_line}{user_info}"

    prompt = f"""You are an NHS-based assistant. Only use this NHS guidance to answer:
{context}
Be honest. If unsure, say so and recommend a doctor."""
    answer = call_groq(prompt, question)
    return answer, top_chunks, top_sources



# Pages

def chatbot_page():
    embedder = load_model()
    chunks, sources = load_documents()
    if not chunks:
        st.error("No NHS documents found.")
        return

    index, _ = create_faiss_index(embedder, chunks)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        col1.metric(" NHS Guidelines", len(set(sources)), help="Total number of guideline files loaded into the system.")
        col2.metric(" Info Snippets", len(chunks), help="Small text sections we use to answer your questions.")

    st.markdown("### 💬 Chat with the Assistant")

    # Chat input (no "Press Enter to apply" hint; auto-submits on Enter)
    with st.container(border=True):
        user_input = st.chat_input(
            "Type your question here…",
            key="user_input"  # still works with sidebar prefill
        )

    k = 5  # fixed retrieval depth

    # When user presses Enter, chat_input returns a string this run
    if user_input and user_input.strip():
        q = user_input.strip()

        start = time.time()
        answer, top_chunks, top_sources = ask_bot(
            q, embedder, index, chunks, sources, st.session_state.user_preferences, k
        )
        duration = time.time() - start

        # save history
        history_col.insert_one({
            "email": st.session_state.user_email,
            "question": q,
            "timestamp": time.time()
        })

        with st.chat_message("assistant"):
            st.markdown(answer)

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
                st.text_area(
                    label=f"Chunk {i+1}",
                    value=chunk,
                    height=100,
                    key=f"chunk_{i}",
                    label_visibility="collapsed"
                )


def user_profile_page():
    try:
        import user_profile
        user_profile.show_profile()
    except ModuleNotFoundError:
        st.error("`user_profile.py` not found. Please ensure it exists in the project folder.")

def health_tools_page():
    st.header("🧪 Health Tools")

    st.subheader("🧬 Simulate Glucose Reading")
    sim_glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=500, step=1, key="sim_glucose_main")
    if st.button("➕ Save Glucose Reading", key="save_glucose_main"):
        store_glucose_reading(st.session_state.user_email, sim_glucose)
        st.success("✅ Glucose data saved.")

    st.divider()
    st.subheader("⏰ Reminders")
    default_time = (datetime.now() + timedelta(minutes=1)).time().replace(second=0, microsecond=0)
    default_message = "Check your glucose level"
    reminder_time = st.time_input("Reminder Time", value=default_time, key="rem_time_main")
    reminder_msg = st.text_input("Reminder Message", value=default_message, key="rem_msg_main")

    if st.button("Add Reminder", key="add_reminder_main"):
        if reminder_msg and reminder_time:
            added = add_reminder(
                st.session_state.user_email,
                reminder_time.strftime("%H:%M"),
                reminder_msg
            )
            if added:
                st.success("✅ Reminder scheduled.")
            else:
                st.warning("⚠️ Reminder already exists.")
        else:
            st.error("❌ Please enter both time and message.")

# Router & Guard
if not st.session_state.logged_in:
    st.warning("🔐 Please log in to use the chatbot.")
    st.stop()

if page == "Chatbot":
    chatbot_page()
elif page == "User Profile":
    user_profile_page()
elif page == "Health Tools":
    health_tools_page()
