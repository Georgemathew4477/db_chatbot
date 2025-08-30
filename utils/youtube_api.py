from youtube_api import get_video_info
import re
import streamlit as st
from googleapiclient.discovery import build
API_KEY = 'AIzaSyBBgZ9CyE9CVRvUNFMtKghisLvK53FgDjg'  # Replace with your actual API Key
youtube = build('youtube', 'v3', developerKey=API_KEY)