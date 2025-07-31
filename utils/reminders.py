from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
from datetime import datetime
import pytz
import os
from dotenv import load_dotenv

load_dotenv()
mongo_url = os.getenv("MONGO_URL")

client = MongoClient(mongo_url)
db = client["chatbot_db"]
reminders_col = db["reminders"]

scheduler = BackgroundScheduler(timezone=pytz.timezone("Europe/London"))
scheduler.start()

def send_reminder(user_email, message):
    print(f"[Reminder for {user_email}] {message} at {datetime.now()}")

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
            print(f"⚠️ Failed to load reminder: {e}")
