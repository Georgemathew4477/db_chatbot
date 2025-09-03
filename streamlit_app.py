# --------- imports (top of file) ----------
import os
import time
import faiss
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
from passlib.hash import bcrypt
from groq import Groq, BadRequestError
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.content_checker import verify_text_against_nhs, verify_media_file
from utils.reminders import add_reminder, load_user_reminders
from utils.glucose import store_glucose_reading, get_latest_glucose
from utils.audio_download import download_and_store_audio
import pytesseract
import platform, shutil

# ✅ try to import link_ingest once (don’t exit the script if it fails)
try:
    from utils.link_ingest import ingest_from_url
except Exception as e:
    ingest_from_url = None
    _link_ingest_error = str(e)
else:
    _link_ingest_error = ""

# Tell pytesseract where to look
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "tesseract"



# =========================
# Environment & Connections
# =========================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
mongo_url = os.getenv("MONGO_URL") or st.secrets.get("MONGO_URL", "")

groq_client = Groq(api_key=api_key)
client = None

PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",   # primary: high quality, large context
    "llama-3.1-8b-instant",      # fallback: fast/cheap, good enough for short replies
]

SAFETY_MODEL = "meta-llama/llama-guard-4-12b"  # optional safety check

db = None
users_col = None
history_col = None

if mongo_url:
    try:
        # Short timeout so local dev doesn't hang if URI is wrong/offline
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # force connection test
        db = client["chatbot_db"]
        users_col = db["users"]
        history_col = db["history"]
    except Exception as e:
        st.warning(f"MongoDB not available: {e}")

# =========================
# Streamlit Base Setup
# =========================
st.set_page_config(page_title="RAG Diabetes Chatbot", page_icon="🤖", layout="wide")

def load_css(path: str = "styles.css"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

def get_user_history(email: str, limit: int = 30):
    if history_col is None:
        return []
    return list(
        history_col.find({"email": email})
                   .sort("timestamp", -1)
                   .limit(limit)
    )

def content_checker_page():
    st.header("🕵️ Content Checker — Verify Social Media Posts/Clips")

    if ingest_from_url is None:
        st.error("`utils.link_ingest` couldn’t be imported on this deployment.")
        if _link_ingest_error:
            st.code(_link_ingest_error, language="text")
        st.info("You can still paste a transcript/caption or upload media and we’ll transcribe it.")

    st.caption(
        "Paste a link, upload a short video/audio, or paste the transcript/caption. "
        "We’ll extract medical claims and verify them against NHS/NICE guidance."
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        enable_stt = st.checkbox("Enable STT (Whisper CPU)", value=True)
    with col_b:
        enable_ocr = st.checkbox("Enable OCR (video frames)", value=False)

    os.environ["USE_WHISPER"] = "1" if enable_stt else "0"
    os.environ["USE_OCR"] = "1" if enable_ocr else "0"

    url = st.text_input("🔗 Link (YouTube or direct media preferred)",
                        placeholder="https://www.youtube.com/watch?v=...")
    up_file = st.file_uploader("Upload video/audio (mp4/mov/mp3/wav) — optional",
                               type=["mp4", "mov", "mp3", "wav"])
    txt = st.text_area("Transcript / Caption / Chat text (recommended)",
                       height=160, placeholder="Paste what the post says here...")
    force_transcribe = st.checkbox("Force transcription (ignore scraped transcript/description)")
    run = st.button("Verify claims", type="primary")

    if not run:
        return

    if not api_key:
        st.error("Groq API key missing. Add `GROQ_API_KEY` in Streamlit Secrets or .env.")
        return

    # RAG assets
    embedder = load_model()
    chunks, sources = load_documents()
    if not chunks:
        st.error("No NHS documents loaded. Add .txt files to /data.")
        return
    index, _ = create_faiss_index(embedder, chunks)

    result = None
    transcript_text = None

    # 1) pasted text
    if txt and txt.strip():
        with st.spinner("Verifying pasted text against NHS/NICE..."):
            result = verify_text_against_nhs(
                txt.strip(), embedder, index, chunks, sources, api_key=api_key, k=5
            )
            transcript_text = result.get("transcript") or txt.strip()

    # 2) URL ingestion (only if importer available)
    elif url and url.strip() and ingest_from_url is not None:
        with st.spinner("Fetching content from link..."):
            try:
                ing = ingest_from_url(url.strip())
                debug = {
                    "kind": ing.get("kind"),
                    "text_preview": (ing.get("text") or "")[:120],
                    "has_video": bool(ing.get("video_path")),
                    "has_audio": bool(ing.get("audio_path")),
                    "note": ing.get("note"),
                }
                st.code(debug, language="json")
            except Exception as e:
                ing = None
                st.warning(f"Couldn’t ingest that link: {e}")

        if not ing:
            st.warning("Couldn’t read that link. Try uploading the file or paste a transcript.")
            return

        st.caption(f"Source note: {ing.get('note','')}")
        scraped_text = ing.get("text")
        video_path = ing.get("video_path")
        audio_path = ing.get("audio_path")
        media_path = video_path or audio_path

        if force_transcribe and media_path:
            result = verify_media_file(media_path, embedder, index, chunks, sources,
                                       api_key=api_key, k=5, use_ocr=enable_ocr)
        elif media_path:
            result = verify_media_file(media_path, embedder, index, chunks, sources,
                                       api_key=api_key, k=5, use_ocr=enable_ocr)
        elif scraped_text:
            result = verify_text_against_nhs(scraped_text, embedder, index, chunks, sources,
                                             api_key=api_key, k=5)
        else:
            st.warning("Couldn’t extract text or media. Try uploading the file.")
            return

        # cleanup
        for p in [video_path, audio_path]:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

        if scraped_text:
            with st.expander("📄 Scraped description/captions (not used for Shorts)"):
                st.text_area("Scraped text (first 1000 chars)", scraped_text[:1000],
                             height=160, key="scraped_text_preview")

    # 2b) URL given but importer missing
    elif url and url.strip() and ingest_from_url is None:
        st.error("Link ingestion isn’t available on this deployment. Please upload the media or paste the transcript.")
        return

    # 3) file upload
    elif up_file is not None:
        with st.spinner("Transcribing media and verifying..."):
            tmp_path = None
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False,
                                                 suffix=os.path.splitext(up_file.name)[-1]) as tmp:
                    tmp.write(up_file.read())
                    tmp_path = tmp.name
                result = verify_media_file(tmp_path, embedder, index, chunks, sources,
                                           api_key=api_key, k=5, use_ocr=enable_ocr)
                transcript_text = result.get("transcript", "")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try: os.unlink(tmp_path)
                    except Exception: pass
    else:
        st.warning("Please provide a link, upload a file, or paste text.")
        return

    # ---- transcript preview & verdict (unchanged below this point) ----

    # ---------- Transcript Preview & Send ----------
    transcript_text = transcript_text or result.get("transcript", "")
    if transcript_text:
        st.markdown("### 📝 Transcript Preview")
        st.text_area(
            "First 1000 characters of transcript:",
            transcript_text[:1000],
            height=220,
            key="transcript_preview",
        )
        if st.button("Send transcript to Chatbot"):
            st.session_state["user_input"] = transcript_text[:4000]  # keep concise
            st.session_state["page"] = "Chatbot"
            st.rerun()
    else:
        st.info("⚠️ No transcript text was captured (STT/OCR may have failed).")

    # ---------- Overall Verdict ----------
    st.subheader("Overall Verdict")
    badge = result["badge"]
    summary = result["summary"]
    if "GREEN" in badge:
        st.success(f"{badge} — {summary}")
    elif "AMBER" in badge:
        st.warning(f"{badge} — {summary}")
    else:
        st.error(f"{badge} — {summary}")

    # ---------- Per-claim assessment ----------
    st.subheader("Per-claim assessment")
    for i, r in enumerate(result.get("results", []), 1):
        claim = r.get("claim", "")
        verdict = r.get("verdict", "INSUFFICIENT")
        reason = r.get("reason", "")
        evidence_idxs = r.get("evidence_idxs", [])

        box = st.success if verdict == "SUPPORTED" else st.error if verdict == "CONTRADICTED" else st.warning
        box(f"{i}. {verdict} — {claim}")
        with st.expander("Rationale & evidence"):
            st.write(reason)
            if evidence_idxs:
                st.caption("Top NHS/NICE snippets:")
                for idx in evidence_idxs[:3]:
                    if 0 <= idx < len(chunks):
                        st.markdown(f"- **{sources[idx]}**")
                        st.text(
                            chunks[idx][:600]
                            + ("..." if len(chunks[idx]) > 600 else "")
                        )



def clear_user_history(email: str):
    if history_col is not None:
        history_col.delete_many({"email": email})

# =========================
# Session State
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.user_preferences = {}

# =========================
# Top Bar (Title + Page Menu)
# =========================
st.markdown('<div class="topbar">', unsafe_allow_html=True)
left, right = st.columns([6, 2])
with left:
    st.title("🤖 NHS-Based Diabetes Chatbot")
with right:
    page = st.selectbox(
        "Page selector",
        ["Chatbot", "User Profile", "Health Tools", "Content Checker"],
        index=0,
        label_visibility="collapsed"
    )
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Sidebar (Auth only, fixed bottom)
# =========================
with st.sidebar:
    # ===== Chat history (only when logged in) =====
    if st.session_state.logged_in and (history_col is not None):
        st.markdown("### 🕘 Recent chats")
        hist_items = get_user_history(st.session_state.user_email, limit=30)

        if not hist_items:
            st.caption("No chats yet.")
        else:
            q_filter = st.text_input("Filter", placeholder="Search history…", key="hist_filter")
            for h in hist_items:
                q = (h.get("question") or "").strip()
                if q_filter and q_filter.lower() not in q.lower():
                    continue
                label = (q[:60] + "…") if len(q) > 60 else q or "(empty)"
                if st.button(label, key=str(h["_id"])):
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
                if users_col is not None and users_col.find_one({"email": email}):
                    st.error("❌ Email already registered.")
                elif users_col is not None:
                    hashed = bcrypt.hash(password)
                    users_col.insert_one({
                        "email": email, "name": name, "password": hashed,
                        "age": age, "gender": gender, "diabetes_type": diabetes_type
                    })
                    st.success("✅ Account created. Please log in.")
                else:
                    st.error("Database not configured.")
        else:
            if st.button("Login"):
                if users_col is None:
                    st.error("Database not configured.")
                else:
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

# =========================
# RAG Helpers
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# === PATCH: safety checker (optional but recommended for medical domain) ===
SAFETY_POLICY = """
You are a safety classifier for a diabetes information assistant.
Allowed: general, educational information; referencing public NHS guidance; reminders to seek clinician advice; non-diagnostic lifestyle tips that are generic and low-risk.
Disallowed: diagnosis, prescribing, dosing instructions, interpreting personal glucose data beyond generic ranges, urgent-care triage beyond “seek medical help”.
Output only: "ALLOW" or "BLOCK" and a short reason on the next line.
"""

def safety_check(answer: str) -> tuple[bool, str]:
    """Return (is_safe, reason)."""
    messages = [
        {"role": "system", "content": SAFETY_POLICY.strip()},
        {"role": "user", "content": f"Candidate assistant answer:\n\n{answer}\n\nClassify."}
    ]
    try:
        resp = client.chat.completions.create(
            model=SAFETY_MODEL,
            messages=messages,
            temperature=0.0,
            max_completion_tokens=128
        )
        out = resp.choices[0].message.content.strip()
        first_line, *rest = out.splitlines()
        reason = "\n".join(rest).strip()
        is_safe = first_line.upper().startswith("ALLOW")
        return is_safe, reason
    except Exception as e:
        # if checker fails, default to safe but add a note (don’t break UX)
        return True, f"safety_check_error: {e}"


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

# === PATCH: chat caller with fallback ===
def call_groq(prompt: str, user_question: str, temperature: float = 0.2, max_tokens: int = 1000):
    """Send a chat completion to Groq with a graceful model fallback."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user",   "content": user_question},
    ]

    last_err = None
    for model_id in PREFERRED_MODELS:
        try:
            result = groq_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,   # Groq-preferred name
                # stream=False  # set True if you wire up streaming in Streamlit
            )
            # unified text extraction
            text = result.choices[0].message.content if result.choices else ""
            return text, model_id
        except BadRequestError as e:
            # if model is decommissioned/invalid, try the next
            if "decommissioned" in str(e).lower() or "invalid" in str(e).lower():
                last_err = e
                continue
            # other 400s (bad inputs), bubble up
            raise
        except Exception as e:
            # transient errors (timeouts, rate limits) -> try fallback
            last_err = e
            continue

    # if we got here, both attempts failed
    raise last_err or RuntimeError("No available Groq models responded.")


def ask_bot(question, embedder, index, chunks, sources, preferences, k=5):
    import numpy as np
    import streamlit as st

    # --- 1) Retrieve top-k chunks via FAISS ---
    q_embed = embedder.encode([question])[0]
    q_vec = np.array([q_embed], dtype="float32")
    _, I = index.search(q_vec, k=k)

    idxs = [i for i in I[0] if 0 <= i < len(chunks)]
    top_chunks = [chunks[i] for i in idxs]
    top_sources = [sources[i] for i in idxs]

    # --- 2) Build context (RAG) + light personalization ---
    context = "\n\n".join(top_chunks)

    # Handle session/email safely
    user_email = getattr(st.session_state, "user_email", None)
    glucose = get_latest_glucose(user_email) if user_email else None
    glucose_line = f"Latest Glucose: {glucose} mg/dL\n" if glucose else ""

    # Handle preferences safely
    preferences = preferences or {}
    user_info = (
        f"User Info: Age: {preferences.get('age', 'N/A')}, "
        f"Gender: {preferences.get('gender', 'N/A')}, "
        f"Diabetes Type: {preferences.get('diabetes_type', 'N/A')}"
    )

    full_context = f"{context}\n\n{glucose_line}{user_info}"

    # --- 3) System prompt (kept concise) ---
    prompt = (
        "You are an NHS-based assistant. Answer using ONLY the provided NHS guidance/context.\n"
        "If the answer is not clearly supported, say you’re unsure and suggest speaking to a clinician.\n\n"
        f"{full_context}"
    )

    # --- 4) Call LLM (with model fallback) ---
    model_answer, used_model = call_groq(
        prompt=prompt,
        user_question=question,
        temperature=0.2,
        max_tokens=1000
    )

    # --- 5) Optional safety pass (Llama-Guard) ---
    is_safe, reason = safety_check(model_answer)
    if not is_safe:
        model_answer = (
            "I'm sorry — I can't provide that. For medical questions like dosing, diagnosis, "
            "or urgent symptoms, please consult a qualified healthcare professional or NHS 111."
        )

    # --- 6) Return (keep your original signature) ---
    return model_answer, top_chunks, top_sources


# =========================
# Pages
# =========================
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

    with st.container(border=True):
        user_input = st.chat_input(
            "Type your question here…",
            key="user_input"
        )

    k = 5

    if user_input and user_input.strip():
        q = user_input.strip()

        start = time.time()
        answer, top_chunks, top_sources = ask_bot(
            q, embedder, index, chunks, sources, st.session_state.user_preferences, k
        )
        duration = time.time() - start

        # save history
        if history_col is not None:
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

# =========================
# Router & Guard
# =========================
if not st.session_state.logged_in:
    st.warning("🔐 Please log in to use the chatbot.")
    st.stop()

if page == "Chatbot":
    chatbot_page()
elif page == "User Profile":
    user_profile_page()
elif page == "Health Tools":
    health_tools_page()
elif page == "Content Checker":
    content_checker_page()

