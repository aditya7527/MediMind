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
import numpy as np
from pydub import AudioSegment
from openai import OpenAI
from streamlit_mic_recorder import speech_to_text

# API Keys
TOGETHER_API_KEY = st.secrets["Together_API"]
GOOGLE_API_KEY = st.secrets["Google_API"]
GOOGLE_CX = st.secrets["Google_CX"]
OPENAI_API_KEY = st.secrets["Open_API"]

# Set Together API key
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# Initialize Together AI client
together_client = Together(api_key=TOGETHER_API_KEY)

# Initialize OpenAI client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.aimlapi.com",
)

# Load emotion detection model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# Emojis for different emotions
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", 
    "happy": "🤗", "joy": "😂", "neutral": "😐", 
    "sad": "😔", "sadness": "😔", "shame": "😳", 
    "surprise": "😮"
}

# Functions for emotion detection
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def ai_analysis(text, predicted_emotion):
    try:
        prompt = f"You are an AI assistant that provides detailed emotional analysis based on user input. The user text reflects a tone of '{predicted_emotion}'. " \
                 "Please offer a thoughtful analysis of the emotions, considering the detected tone, and give suggestions on how the user might feel or act next."

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
        print(f"Error in AI analysis: {str(e)}")
        return "Could not complete the analysis due to an error."

# Function for Google Custom Search API
def google_search(query):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    result = service.cse().list(q=query, cx=GOOGLE_CX).execute()
    return result.get('items', [])

# Scraping article content from a URL
def extract_article_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting article content: {e}")
        return None

# Convert text to audio
def text_to_audio(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        audio_file_path = fp.name
        tts.save(audio_file_path)
    return audio_file_path

# Play audio using sounddevice
def play_audio(file_path):
    audio_segment = AudioSegment.from_mp3(file_path)
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    sd.play(samples, samplerate=audio_segment.frame_rate)
    sd.wait()

# Detect language
def detect_language(text):
    return detect(text)

# Main App
def main():
    st.set_page_config(page_title="Emotion Detection App", layout="wide")
    st.title("🧠 Advanced Text Emotion Detection & Summarization App")
    st.write("### Detect emotions in text, emails, and articles.")
    
    theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("""<style>
            .stApp {
                background-color: #121212;
                color: white;
            }
            .stSidebar {
                background-color: #1e1e1e;
                color: white;
            }
            </style>""", unsafe_allow_html=True)

    option = st.sidebar.selectbox("Choose input type", ["Text Input", "Email Input", "Article URL", "Google Search", "Audio Input"])

    st.sidebar.write("### Instructions")
    st.sidebar.write("""\
    1. **Text Input**: Type or paste any text you want to analyze for emotional tone.
    2. **Email Input**: Paste the content of your email to understand its emotional context.
    3. **Article URL**: Provide a URL of an article, and the app will extract and analyze its emotional tone.
    4. **Google Search**: Enter a search query to find articles and analyze their emotional content.
    5. **Audio Input**: Speak your text, and the app will analyze the emotions in your spoken words.
    """)
    st.sidebar.write("---")

    st.sidebar.write("### Find the source code here:")
    st.sidebar.markdown("[GitHub Repository](https://github.com/yourusername/your-repo-name)")

    if option == "Text Input":
        st.subheader("💬 Emotion Detection from Text")
        raw_text = st.text_area("Type your text here:", height=150)
        if st.button("Analyze Text Emotions"):
            with st.spinner("Analyzing..."):
                if raw_text:
                    prediction = predict_emotions(raw_text)
                    probability = get_prediction_proba(raw_text)
                    st.success(f"**Predicted Emotion:** {prediction} {emotions_emoji_dict[prediction]}")
                    st.write(f"**Prediction Confidence:** {np.max(probability):.2f}")

                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(
                        x='emotions',
                        y='probability',
                        color='emotions'
                    ).properties(title="Emotion Probabilities")
                    st.altair_chart(fig, use_container_width=True)

                    analysis = ai_analysis(raw_text, prediction)
                    st.write("### AI Analysis:")
                    st.write(analysis)

                    audio_file = text_to_audio(analysis)
                    st.audio(audio_file, format='audio/mp3')

                    user_response = st.text_input("🤔 How do you feel about this analysis? What would you like to discuss?", "")
                    if user_response:
                        empathetic_response = ai_analysis(user_response, prediction)
                        st.write("### AI Empathetic Response:")
                        st.write(empathetic_response)

                    st.download_button(
                        "📥 Download Results as CSV", 
                        proba_df_clean.to_csv(index=False), 
                        "emotion_probabilities.csv", 
                        "text/csv"
                    )

    elif option == "Email Input":
        st.subheader("📧 Emotion Detection from Email")
        email_text = st.text_area("Paste your email content here:", height=150)
        if st.button("Analyze Email Emotions"):
            with st.spinner("Analyzing..."):
                if email_text:
                    prediction = predict_emotions(email_text)
                    probability = get_prediction_proba(email_text)
                    st.success(f"**Predicted Emotion:** {prediction} {emotions_emoji_dict[prediction]}")
                    st.write(f"**Prediction Confidence:** {np.max(probability):.2f}")

                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(
                        x='emotions',
                        y='probability',
                        color='emotions'
                    ).properties(title="Emotion Probabilities")
                    st.altair_chart(fig, use_container_width=True)

                    analysis = ai_analysis(email_text, prediction)
                    st.write("### AI Analysis:")
                    st.write(analysis)

                    audio_file = text_to_audio(analysis)
                    st.audio(audio_file, format='audio/mp3')

                    user_response = st.text_input("🤔 How do you feel about this analysis? What would you like to discuss?", "")
                    if user_response:
                        empathetic_response = ai_analysis(user_response, prediction)
                        st.write("### AI Empathetic Response:")
                        st.write(empathetic_response)

    elif option == "Article URL":
        st.subheader("📄 Emotion Detection from Article URL")
        article_url = st.text_input("Enter the URL of the article:")
        if st.button("Extract and Analyze Article"):
            with st.spinner("Extracting article content..."):
                article_content = extract_article_content(article_url)
                if article_content:
                    prediction = predict_emotions(article_content)
                    probability = get_prediction_proba(article_content)
                    st.success(f"**Predicted Emotion:** {prediction} {emotions_emoji_dict[prediction]}")
                    st.write(f"**Prediction Confidence:** {np.max(probability):.2f}")

                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(
                        x='emotions',
                        y='probability',
                        color='emotions'
                    ).properties(title="Emotion Probabilities")
                    st.altair_chart(fig, use_container_width=True)

                    analysis = ai_analysis(article_content, prediction)
                    st.write("### AI Analysis:")
                    st.write(analysis)

                    audio_file = text_to_audio(analysis)
                    st.audio(audio_file, format='audio/mp3')

                    user_response = st.text_input("🤔 How do you feel about this analysis? What would you like to discuss?", "")
                    if user_response:
                        empathetic_response = ai_analysis(user_response, prediction)
                        st.write("### AI Empathetic Response:")
                        st.write(empathetic_response)

    elif option == "Google Search":
        st.subheader("🔍 Google Search for Emotion Detection")
        search_query = st.text_input("Enter your search query:")
        if st.button("Search"):
            with st.spinner("Searching..."):
                results = google_search(search_query)
                if results:
                    for result in results:
                        st.write(f"[{result['title']}]({result['link']})")
                        st.write(result['snippet'])
                        st.write("---")
                else:
                    st.write("No results found.")

    elif option == "Audio Input":
        st.subheader("🎤 Emotion Detection from Audio Input")
        audio_input = speech_to_text()
        
        if st.button("Analyze Audio"):
            if audio_input:
                st.session_state.last_input_time = time.time()
                with st.spinner("Analyzing..."):
                    prediction = predict_emotions(audio_input)
                    probability = get_prediction_proba(audio_input)
                    st.success(f"**Predicted Emotion:** {prediction} {emotions_emoji_dict[prediction]}")
                    st.write(f"**Prediction Confidence:** {np.max(probability):.2f}")

                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(
                        x='emotions',
                        y='probability',
                        color='emotions'
                    ).properties(title="Emotion Probabilities")
                    st.altair_chart(fig, use_container_width=True)

                    analysis = ai_analysis(audio_input, prediction)
                    st.write("### AI Analysis:")
                    st.write(analysis)

                    audio_file = text_to_audio(analysis)
                    st.audio(audio_file, format='audio/mp3')

                    user_response = st.text_input("🤔 How do you feel about this analysis? What would you like to discuss?", "")
                    if user_response:
                        empathetic_response = ai_analysis(user_response, prediction)
                        st.write("### AI Empathetic Response:")
                        st.write(empathetic_response)

if __name__ == "__main__":
    main()
