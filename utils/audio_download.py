# audio_download.py

import os
import pymongo
import yt_dlp
from bson import Binary
import tempfile
import shutil
import datetime
import streamlit as st

# Use Streamlit's secrets management or fallback to environment variable
mongo_url = os.getenv("MONGO_URL") or st.secrets.get("MONGO_URL", "")


# Ensure the URL is valid
if not mongo_url:
    raise ValueError("MongoDB connection string not found. Please set MONGO_URL environment variable or Streamlit secret.")

# Initialize MongoDB client with the connection string
client = pymongo.MongoClient(mongo_url)

# Example: Using the client to access the database and collection
db = client["chatbot_db"]  # Replace with your actual database name
audio_files_collection = db["audio_files"]  # Replace with your collection name


# Function to delete the oldest file from MongoDB
def delete_oldest_file():
    oldest_file = audio_files_collection.find().sort("timestamp", pymongo.ASCENDING).limit(1)
    if oldest_file:
        audio_files_collection.delete_one({"_id": oldest_file[0]["_id"]})

# Function to download audio and store it in MongoDB as MP3
def download_and_store_audio(url: str):
    # Delete the oldest audio file if a new file is to be added
    delete_oldest_file()

    # Download the audio using yt-dlp and convert to MP3
    tmpdir = tempfile.mkdtemp()
    ydl_opts = {
        "format": "bestaudio/best",  # Download best audio
        "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegAudioConvertor",
            "preferredcodec": "mp3",  # Convert to MP3 format
            "preferredquality": "192",  # 192 kbps quality
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = os.path.join(tmpdir, f"{info['id']}.mp3")

    # Store the audio file in MongoDB
    with open(audio_path, "rb") as audio_file:
        audio_data = Binary(audio_file.read())
        audio_files_collection.insert_one({
            "audio_data": audio_data,
            "url": url,
            "timestamp": datetime.datetime.now(),  # Store the timestamp of when it was added
        })

    # Clean up the temporary files
    shutil.rmtree(tmpdir)
