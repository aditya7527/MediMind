import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from together import Together
import os
import time
from gtts import gTTS
import tempfile
from langdetect import detect
import sounddevice as sd
from pydub import AudioSegment
from openai import OpenAI
from streamlit_mic_recorder import speech_to_text
from streamlit_TTS import auto_play, text_to_audio


# API Keys
TOGETHER_API_KEY = st.secrets["Together_API"]
GOOGLE_API_KEY = st.secrets["Google_API"]
GOOGLE_CX = st.secrets["Google_CX"]
OPENAI_API_KEY = st.secrets["Open_API"]

# Set Together API key
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# Initialize API Clients
together_client = Together(api_key=TOGETHER_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.aimlapi.com")

# Load emotion detection model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# Emoji Map
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮", "love": "❤️", "grief": "😢", 
    "anxiety": "😰", "hope": "🌈", "determination": "💪", 
    "frustration": "😩", "confusion": "😕"
}

# Prediction functions
def predict_emotions(docx):
    return pipe_lr.predict([docx])[0]

def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

# AI Analysis
def ai_analysis(text, predicted_emotion):
    try:
        prompt = (
            f"You are an AI assistant tasked with providing an in-depth emotional analysis based on user input. "
            f"The text provided by the user conveys a tone of '{predicted_emotion}'. "
            "Please deliver a thoughtful analysis of the emotions reflected in the text, taking into account the identified tone. "
            "Also, suggest how the user might feel or behave moving forward. "
            "Use appropriate emojis matching the emotional tone."
        )

        response = together_client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[{"role": "system", "content": "You are an AI assistant analyzing emotional tone."},
                      {"role": "user", "content": prompt},
                      {"role": "user", "content": text}],
            max_tokens=500,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=True
        )

        full_response = ""
        for token in response:
            if hasattr(token, 'choices') and token.choices:
                content = token.choices[0].delta.content
                full_response += content
        return full_response or "No analysis content returned."

    except Exception as e:
        print(f"Together AI failed: {str(e)}")
        return "Could not complete the analysis due to an error."

# Emotion category analysis coloring
def get_emotion(analysis, user_query):
    emotion_color_map = {
        "calm": "#A7C7E7", "trust": "#A7C7E7", "serenity": "#A7C7E7",
        "balance": "#C8E6C9", "harmony": "#C8E6C9", "nature": "#C8E6C9",
        "soothe": "#E1BEE7", "relaxation": "#E1BEE7",
        "care": "#F8BBD0", "compassion": "#F8BBD0", "warmth": "#F8BBD0",
        "simplicity": "#D7CCC8", "depressed": "#A7C7E7", "default": "#A7C7E7"
    }

    emotion_extraction_prompt = (
        f"From the assistant's analysis and the user's input, extract the dominant emotion. "
        f"Choose only one from {list(emotion_color_map.keys())}. "
        f"Do not include any additional text.\n\n"
        f"Assistant's response: {analysis}\nUser query: {user_query}"
    )

    try:
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[{"role": "user", "content": emotion_extraction_prompt}]
        )
        emotion = response.choices[0].message.content.strip().lower()
    except:
        emotion = "default"

    color = emotion_color_map.get(emotion, emotion_color_map["default"])
    for paragraph in analysis.split('\n'):
        st.markdown(f"<p style='color: {color};'>{paragraph}</p>", unsafe_allow_html=True)
    return emotion

# Utility Functions
def google_search(query):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    result = service.cse().list(q=query, cx=GOOGLE_CX).execute()
    return result.get('items', [])

def extract_article_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return " ".join([p.get_text() for p in paragraphs])
    except Exception as e:
        print(f"Error extracting article: {e}")
        return None

def play_audio(file_path):
    audio_segment = AudioSegment.from_mp3(file_path)
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    sd.play(samples, samplerate=audio_segment.frame_rate)
    sd.wait()

def detect_language(text):
    return detect(text)

def callback():
    if st.session_state.my_stt_output:
        st.write(st.session_state.my_stt_output)

# Main App
def main():
    st.set_page_config(page_title="Emotion Detection App", layout="wide")
    st.title("🧠 Advanced Text Emotion Detection & Summarization App")

    theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("""<style>
            .stApp { background-color: #121212; color: white; }
            .stSidebar { background-color: #1e1e1e; color: white; }
            </style>""", unsafe_allow_html=True)

    option = st.sidebar.selectbox("Choose input type", [
        "Audio Input", "Text Input", "Email Input", "Article URL"
    ])

    st.sidebar.write("### Instructions")
    st.sidebar.write("""
        1. Text Input: Type or paste any text.
        2. Email Input: Paste your email body.
        3. Article URL: Provide the link to the article.
        4. Audio Input: Speak your text for analysis.
    """)
    st.sidebar.markdown("---")


    if option in ["Text Input", "Email Input"]:
        input_label = "Type your text here:" if option == "Text Input" else "Paste your email content:"
        input_text = st.text_area(input_label, height=150)
        if st.button("Analyze Emotions"):
            with st.spinner("Analyzing..."):
                if input_text:
                    prediction = predict_emotions(input_text)
                    probability = get_prediction_proba(input_text)
                    st.success(f"**Predicted Emotion:** {prediction} {emotions_emoji_dict.get(prediction, '')}")
                    st.write(f"**Prediction Confidence:** {np.max(probability):.2f}")

                    df = pd.DataFrame(probability, columns=pipe_lr.classes_).T.reset_index()
                    df.columns = ["emotions", "probability"]
                    fig = alt.Chart(df).mark_bar().encode(x='emotions', y='probability', color='emotions')
                    st.altair_chart(fig, use_container_width=True)

                    analysis = ai_analysis(input_text, prediction)
                    st.write("### AI Analysis:")
                    st.markdown(f"<div style='color:black;'>{analysis}</div>", unsafe_allow_html=True)
                    auto_play(text_to_audio(analysis))

                    user_response = st.text_input("🤔 Your thoughts?")
                    if user_response:
                        reply = ai_analysis(user_response, prediction)
                        st.write("### AI Empathetic Response:")
                        st.markdown(f"<div style='color:black;'>{reply}</div>", unsafe_allow_html=True)

    elif option == "Article URL":
        url = st.text_input("Enter the article URL:")
        if st.button("Extract and Analyze"):
            with st.spinner("Fetching and analyzing article..."):
                content = extract_article_content(url)
                if content:
                    prediction = predict_emotions(content)
                    probability = get_prediction_proba(content)
                    st.success(f"**Predicted Emotion:** {prediction} {emotions_emoji_dict.get(prediction, '')}")
                    st.write(f"**Prediction Confidence:** {np.max(probability):.2f}")

                    df = pd.DataFrame(probability, columns=pipe_lr.classes_).T.reset_index()
                    df.columns = ["emotions", "probability"]
                    fig = alt.Chart(df).mark_bar().encode(x='emotions', y='probability', color='emotions')
                    st.altair_chart(fig, use_container_width=True)

                    analysis = ai_analysis(content, prediction)
                    st.write("### AI Analysis:")
                    st.markdown(f"<div style='color:black;'>{analysis}</div>", unsafe_allow_html=True)
                    auto_play(text_to_audio(analysis))

    elif option == "Google Search":
        query = st.text_input("Enter your search query:")
        if st.button("Search"):
            with st.spinner("Searching..."):
                results = google_search(query)
                for r in results:
                    st.write(f"[{r['title']}]({r['link']})")
                    st.write(r['snippet'])
                    st.markdown("---")

    elif option == "Audio Input":
        st.subheader("🎤 Emotion Detection from Audio")
        audio_input = speech_to_text(key='my_stt', callback=callback)
        if st.button("Analyze Audio"):
            if audio_input:
                with st.spinner("Analyzing..."):
                    prediction = predict_emotions(audio_input)
                    probability = get_prediction_proba(audio_input)
                    st.success(f"**Predicted Emotion:** {prediction} {emotions_emoji_dict.get(prediction, '')}")
                    st.write(f"**Prediction Confidence:** {np.max(probability):.2f}")

                    df = pd.DataFrame(probability, columns=pipe_lr.classes_).T.reset_index()
                    df.columns = ["emotions", "probability"]
                    fig = alt.Chart(df).mark_bar().encode(x='emotions', y='probability', color='emotions')
                    st.altair_chart(fig, use_container_width=True)

                    analysis = ai_analysis(audio_input, prediction)
                    st.write("### AI Analysis:")
                    st.markdown(f"<div style='color:black;'>{analysis}</div>", unsafe_allow_html=True)
                    auto_play(text_to_audio(analysis))

if __name__ == "__main__":
    main()
