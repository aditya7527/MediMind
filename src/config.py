import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- API Keys ---
# Try loading from Streamlit secrets first, then environment variables
TOGETHER_API_KEY = st.secrets.get("Together_API", os.getenv("TOGETHER_API_KEY"))
GOOGLE_API_KEY = st.secrets.get("Google_API", os.getenv("GOOGLE_API_KEY"))
GOOGLE_CX = st.secrets.get("Google_CX", os.getenv("GOOGLE_CX"))
OPENAI_API_KEY = st.secrets.get("Open_API", os.getenv("OPENAI_API_KEY"))

# Set Environment Variables for libraries that expect them
if TOGETHER_API_KEY:
    os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- UI Constants ---
APP_TITLE = "MediMind"
APP_SUBTITLE = "Advanced Emotion Detection & AI Analysis"

# Theme Colors (Deep Violet/Blue Gradient System)
COLORS = {
    "background_gradient": "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%)",
    "accent_gradient": "linear-gradient(90deg, #7c3aed 0%, #f472b6 100%)",
    "glass_bg": "rgba(255, 255, 255, 0.08)",
    "glass_border": "rgba(255, 255, 255, 0.1)",
    "text_primary": "#f8fafc",
    "text_secondary": "#cbd5e1",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444"
}

# --- Emoji Map ---
EMOTION_EMOJI_MAP = {
    "anger": "😠", "disgust": "🤮", "fear": "😨", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮", "love": "❤️", "grief": "😢", 
    "anxiety": "😰", "hope": "🌈", "determination": "💪", 
    "frustration": "😩", "confusion": "😕", "stressed": "😫"
}
