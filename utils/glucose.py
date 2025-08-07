from pymongo import MongoClient
from datetime import datetime
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()
mongo_url = os.getenv("MONGO_URL")
email_host = os.getenv("EMAIL_HOST")
email_port = int(os.getenv("EMAIL_PORT", 587))
email_address = os.getenv("EMAIL_ADDRESS")
email_password = os.getenv("EMAIL_PASSWORD")

client = MongoClient(mongo_url)
db = client["chatbot_db"]
glucose_col = db["glucose"]

# === Email Alert Function ===
def send_glucose_alert(user_email, glucose_value, level_type):
    if level_type == "high":
        subject = "🚨 High Glucose Alert"
        msg_text = f"""⚠️ Your glucose level is {glucose_value} mg/dL, which is considered **high**.

🔽 What you can do:
- Drink water to help flush excess sugar.
- Do light physical activity (e.g., a 20-minute walk).
- Avoid sugary or starchy food for now.
- Recheck after 1-2 hours.

If this happens often, consult your healthcare provider.
"""
    elif level_type == "low":
        subject = "🚨 Low Glucose Alert"
        msg_text = f"""⚠️ Your glucose level is {glucose_value} mg/dL, which is considered **low**.

🔼 What you can do:
- Eat/drink 15g of fast-acting carbs (e.g., glucose tablets, juice).
- Wait 15 minutes and check again.
- Avoid driving or intense activity until you're stable.

Repeated lows? Please consult your doctor.
"""
    else:
        return  # Unknown type

    try:
        msg = MIMEText(msg_text)
        msg["Subject"] = subject
        msg["From"] = email_address
        msg["To"] = user_email

        with smtplib.SMTP(email_host, email_port) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.sendmail(email_address, [user_email], msg.as_string())

        print(f"🚨 {level_type.capitalize()} alert sent to {user_email} for glucose {glucose_value}")
    except Exception as e:
        print(f"❌ Failed to send {level_type} glucose alert: {e}")

# === Store Glucose Value and Check for Alerts ===
def store_glucose_reading(user_email, glucose_value):
    glucose_col.insert_one({
        "user_email": user_email,
        "glucose": glucose_value,
        "timestamp": datetime.now()
    })

    if glucose_value >= 180:
        send_glucose_alert(user_email, glucose_value, "high")
    elif glucose_value <= 70:
        send_glucose_alert(user_email, glucose_value, "low")

# === Get Latest Glucose ===
def get_latest_glucose(user_email):
    record = glucose_col.find_one(
        {"user_email": user_email},
        sort=[("timestamp", -1)]
    )
    return record["glucose"] if record else None
