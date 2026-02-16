import os
import streamlit as st
from openai import OpenAI
from .config import TOGETHER_API_KEY, OPENAI_API_KEY

# Initialize Clients
together_client = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz/v1",
) if TOGETHER_API_KEY else None

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def get_ai_analysis(emotion, confidence, text_input):
    """
    Generates AI-powered insights based on the detected emotion and input text.
    Handles API errors gracefully and provides fallbacks.
    """
    if not together_client and not openai_client:
        return "⚠️ AI Analysis Unavailable: Please configure API keys in settings."

    prompt = f"""
    You are an empathetic mental health AI assistant.
    The user is feeling: {emotion} (Confidence: {confidence:.2f}).
    User's input: "{text_input}"

    Provide a structured response with these sections:
    1. 🔍 **Emotion Analysis**: Briefly explain why they might feel this way based on the text.
    2. 💡 **Coping Mechanism**: A specific, actionable tip to manage or embrace this emotion.
    3. 🧘 **Mindfulness Exercise**: A short 2-minute activity (e.g., breathing, grounding).

    Keep it concise, supportive, and warm. Avoid generic medical advice.
    """

    # Try Together AI First
    if together_client:
        try:
            response = together_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful and empathetic AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=350,
                temperature=0.7,
                timeout=10 # Timeout after 10 seconds
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Together AI Error: {e}")
            # Fallback to OpenAI if configured
            if not openai_client:
                return f"⚠️ Analysis Error: Could not connect to AI service. ({str(e)})"

    # Fallback to OpenAI
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful mental health assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=350
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ OpenAI Error: {str(e)}"

    return "⚠️ AI Service Unavailable"
