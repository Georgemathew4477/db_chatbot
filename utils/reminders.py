from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
from datetime import datetime
import pytz
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
mongo_url = os.getenv("MONGO_URL")
email_host = os.getenv("EMAIL_HOST")
email_port = int(os.getenv("EMAIL_PORT", 587))
email_address = os.getenv("EMAIL_ADDRESS")
email_password = os.getenv("EMAIL_PASSWORD")

# === MongoDB Connection ===
client = MongoClient(mongo_url)
db = client["chatbot_db"]
reminders_col = db["reminders"]

# === APScheduler Setup ===
scheduler = BackgroundScheduler(timezone=pytz.timezone("Europe/London"))
scheduler.start()

# === Email Sending Function ===
def send_reminder(user_email, message):
    try:
        msg = MIMEText(message)
        msg["Subject"] = "🩺 Glucose Reminder"
        msg["From"] = email_address
        msg["To"] = user_email

        with smtplib.SMTP(email_host, email_port) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.sendmail(email_address, [user_email], msg.as_string())

        print(f"✅ Email reminder sent to {user_email} at {datetime.now()}")

    except Exception as e:
        print(f"❌ Failed to send email to {user_email}: {e}")

# === Add Reminder ===
def add_reminder(user_email, time_str, message):
    hour, minute = map(int, time_str.split(":"))
    job_id = f"{user_email}_{time_str}_{message}"

    if scheduler.get_job(job_id):
        return False  # prevent duplicates

    scheduler.add_job(
        func=send_reminder,
        trigger="cron",
        hour=hour,
        minute=minute,
        id=job_id,
        args=[user_email, message],
        replace_existing=True,
    )

    reminders_col.insert_one({
        "user_email": user_email,
        "reminder_time": time_str,
        "message": message,
        "job_id": job_id,
    })

    return True

# === Load Reminders from DB on Login ===
def load_user_reminders(user_email):
    reminders = reminders_col.find({"user_email": user_email})
    for r in reminders:
        try:
            hour, minute = map(int, r["reminder_time"].split(":"))
            scheduler.add_job(
                func=send_reminder,
                trigger="cron",
                hour=hour,
                minute=minute,
                id=r["job_id"],
                args=[r["user_email"], r["message"]],
                replace_existing=True,
            )
        except Exception as e:
            print(f"⚠️ Failed to load reminder for {user_email}: {e}")
