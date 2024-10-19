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

# API Keys
TOGETHER_API_KEY = ""
GOOGLE_API_KEY = ""
GOOGLE_CX = ""

# Set Together API key
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# Initialize Together AI client
together_client = Together(api_key=TOGETHER_API_KEY)

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
        # Update the LLaMA prompt to include emotional context
        prompt = f"You are an AI assistant that provides detailed emotional analysis based on user input. The user text reflects a tone of '{predicted_emotion}'. " \
                 "Please offer a thoughtful analysis of the emotions, considering the detected tone, and give suggestions on how the user might feel or act next."

        # Call to the LLaMA model with the new prompt
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

        # Collect the response
        full_response = ""
        for token in response:
            if hasattr(token, 'choices') and token.choices:
                content = token.choices[0].delta.content
                full_response += content
            else:
                print("No choices returned in the response.")

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

# Function to convert text to audio
def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        audio_file_path = f"{fp.name}.mp3"
        tts.save(audio_file_path)
        return audio_file_path

# Main App
def main():
    st.set_page_config(page_title="Emotion Detection App", layout="wide")
    st.title("🧠 Advanced Text Emotion Detection & Summarization App")
    st.write("### Detect emotions in text, emails, and articles.")
    
    # Sidebar for theme selection
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

    # Sidebar options
    option = st.sidebar.selectbox("Choose input type", ["Text Input", "Email Input", "Article URL", "Google Search"])

    st.sidebar.write("### Instructions")
    st.sidebar.write("""\
    1. **Text Input**: Type or paste any text you want to analyze for emotional tone.
    2. **Email Input**: Paste the content of your email to understand its emotional context.
    3. **Article URL**: Provide a URL of an article, and the app will extract and analyze its emotional tone.
    4. **Google Search**: Enter a search query to find articles and analyze their emotional content.
    """)
    st.sidebar.write("---")

    # GitHub link
    st.sidebar.write("### Find the source code here:")
    st.sidebar.markdown("[GitHub Repository](https://github.com/yourusername/your-repo-name)")

    # Initialize session state variables
    if 'last_input_time' not in st.session_state:
        st.session_state.last_input_time = time.time()

    # Track response speed
    current_time = time.time()
    response_time = current_time - st.session_state.last_input_time

    if option == "Text Input":
        st.subheader("💬 Emotion Detection from Text")
        raw_text = st.text_area("Type your text here:", height=150)
        if st.button("Analyze Text Emotions"):
            st.session_state.last_input_time = current_time  # Update the last input time
            with st.spinner("Analyzing..."):
                if raw_text:
                    # Emotion prediction
                    prediction = predict_emotions(raw_text)
                    probability = get_prediction_proba(raw_text)
                    st.success(f"**Predicted Emotion:** {prediction} {emotions_emoji_dict[prediction]}")
                    st.write(f"**Prediction Confidence:** {np.max(probability):.2f}")

                    # Plot probabilities
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(
                        x='emotions',
                        y='probability',
                        color='emotions'
                    ).properties(title="Emotion Probabilities")
                    st.altair_chart(fig, use_container_width=True)

                    # AI Analysis with emotion input
                    analysis = ai_analysis(raw_text, prediction)
                    st.write("### AI Analysis:")
                    st.write(analysis)

                    # Convert text to audio
                    audio_file = text_to_audio(analysis)
                    st.audio(audio_file, format='audio/mp3')

                    # Input for user response
                    user_response = st.text_input("🤔 How do you feel about this analysis? What would you like to discuss?", "")
                    if user_response:  # Check if the user has provided input
                        # Get empathetic response from AI
                        empathetic_response = ai_analysis(user_response, prediction)
                        st.write("### AI Empathetic Response:")
                        st.write(empathetic_response)

                    # Download button
                    st.download_button(
                        "📥 Download Results as CSV", 
                        proba_df_clean.to_csv(index=False), 
                        "emotion_probabilities.csv", 
                        "text/csv"
                    )

                    # Check response speed and provide empathetic messages
                    if response_time > 10:  # Change threshold as needed
                        st.warning("Are you okay? Take your time.")
                    elif response_time < 5:  # Change threshold as needed
                        st.success("Oh, you seem happy! 😊")

    elif option == "Email Input":
        st.subheader("📧 Emotion Detection from Email")
        email_text = st.text_area("Paste your email content here:", height=150)
        if st.button("Analyze Email Emotions"):
            st.session_state.last_input_time = current_time
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

                    st.download_button(
                        "📥 Download Results as CSV", 
                        proba_df_clean.to_csv(index=False), 
                        "emotion_probabilities.csv", 
                        "text/csv"
                    )

    elif option == "Article URL":
        st.subheader("📰 Emotion Detection from Article URL")
        article_url = st.text_input("Enter the URL of the article:")
        if st.button("Fetch Article"):
            st.session_state.last_input_time = current_time
            with st.spinner("Fetching article..."):
                article_content = extract_article_content(article_url)
                if article_content:
                    st.write("### Article Content:")
                    st.write(article_content)

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

                    st.download_button(
                        "📥 Download Results as CSV", 
                        proba_df_clean.to_csv(index=False), 
                        "emotion_probabilities.csv", 
                        "text/csv"
                    )
                else:
                    st.error("Could not extract content from the provided URL.")

    elif option == "Google Search":
        st.subheader("🔍 Google Search for Articles")
        search_query = st.text_input("Enter your search query:")
        if st.button("Search Articles"):
            st.session_state.last_input_time = current_time
            with st.spinner("Searching..."):
                search_results = google_search(search_query)
                if search_results:
                    for item in search_results:
                        st.write(f"### [{item['title']}]({item['link']})")
                        st.write(f"{item['snippet']}")
                        st.write("---")
                    st.success("Found articles! Click on the titles to view.")
                else:
                    st.error("No articles found for your search query.")

if __name__ == "__main__":
    main()
