# utils/glucose.py

from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
mongo_url = os.getenv("MONGO_URL")

client = MongoClient(mongo_url)
db = client["chatbot_db"]
glucose_col = db["glucose"]

def store_glucose_reading(user_email, glucose_value):
    glucose_col.insert_one({
        "user_email": user_email,
        "glucose": glucose_value,
        "timestamp": datetime.now()
    })

def get_latest_glucose(user_email):
    record = glucose_col.find_one(
        {"user_email": user_email},
        sort=[("timestamp", -1)]
    )
    return record["glucose"] if record else None
