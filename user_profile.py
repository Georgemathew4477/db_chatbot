import streamlit as st
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# user_profile.py

def show_profile():
    import streamlit as st
    from dotenv import load_dotenv
    from pymongo import MongoClient
    import os

    load_dotenv()
    mongo_url = os.getenv("MONGO_URL") or st.secrets["MONGO_URL"]
    client = MongoClient(mongo_url)
    db = client["chatbot_db"]
    users_col = db["users"]
    history_col = db["history"]

    st.set_page_config(page_title="User Profile", page_icon="👤", layout="wide")
    st.title("👤 Your Profile")

    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.warning("🔐 Please log in to view your profile.")
        st.stop()

    email = st.session_state.user_email
    user_data = users_col.find_one({"email": email})

    if not user_data:
        st.error("⚠️ User not found in database.")
        st.stop()

    st.subheader("🧾 Basic Information")
    st.write(f"**Name:** {user_data.get('name', 'N/A')}")
    st.write(f"**Email:** {user_data.get('email', 'N/A')}")
    st.write(f"**Age:** {user_data.get('age', 'Not Provided')}")
    st.write(f"**Gender:** {user_data.get('gender', 'Not Provided')}")
    st.write(f"**Diabetes Type:** {user_data.get('diabetes_type', 'Not Provided')}")

    st.subheader("📜 Search History")
    queries = history_col.find({"email": email}).sort("timestamp", -1).limit(10)
    history = list(queries)

    if history:
        for entry in history:
            st.markdown(f"**Q:** {entry.get('question')}\n\n**A:** {entry.get('answer')}")
            st.markdown("---")
    else:
        st.info("No search history available.")
