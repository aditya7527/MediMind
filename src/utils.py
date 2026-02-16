import requests
from bs4 import BeautifulSoup
from langdetect import detect
import streamlit as st
import traceback

# --- Web Search Utils ---
def google_search(query, api_key, cx):
    """
    Performs a Google Custom Search.
    Returns a list of result dictionaries or an empty list on error.
    """
    if not api_key or not cx:
        st.error("Google Search API Key or CX not configured.")
        return []
        
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": api_key, "cx": cx}
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        results = response.json().get("items", [])
        return results
    except Exception as e:
        print(f"Search Error: {e}")
        return []

def extract_article_content(url):
    """
    Extracts main text content from a given URL.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
            
        text = soup.get_text(separator=' ')
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text[:2000] # Return first 2000 chars
    except Exception as e:
        return f"Error extracting content: {str(e)}"

# --- Audio Utils ---
def play_audio(text):
    """
    Placeholder for audio playback. 
    Real implementation requires ffmpeg/pydub which is flaky in some envs.
    """
    # In a real app, use gTTS or ElevenLabs here
    pass

# --- Language Utils ---
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"
