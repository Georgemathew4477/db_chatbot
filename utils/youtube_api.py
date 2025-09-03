from youtube_api import get_video_info
import re
import streamlit as st
from googleapiclient.discovery import build
API_KEY = 'AIzaSyBBgZ9CyE9CVRvUNFMtKghisLvK53FgDjg'  # Replace with your actual API Key
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_metadata(video_id):
    request = youtube.videos().list(
        part="snippet,statistics",  # metadata fields to fetch
        id=video_id                # YouTube video ID
    )
    response = request.execute()

    # Extract video details
    video_details = response['items'][0]  # First item in response
    title = video_details['snippet']['title']
    description = video_details['snippet']['description']
    views = video_details['statistics']['viewCount']

    return {
        'title': title,
        'description': description,
        'views': views
    }
def get_video_captions(video_id):
    request = youtube.captions().list(
        part="snippet",
        videoId=video_id
    )
    response = request.execute()

    if response['items']:
        caption = response['items'][0]['snippet']['language']
        return f"Captions available in: {caption}"
    else:
        return "No captions available"


def get_video_info(video_id):
    # Get video metadata
    metadata = get_video_metadata(video_id)

    # Get captions (if available)
    captions_info = get_video_captions(video_id)

    return {**metadata, 'captions': captions_info}
