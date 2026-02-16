import joblib
import streamlit as st
import os
import numpy as np

# Use absolute path for reliability
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "text_emotion.pkl")

@st.cache_resource(show_spinner="Loading Emotion Model...")
def load_emotion_model():
    """Loads the pre-trained emotion detection model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    return joblib.load(open(MODEL_PATH, "rb"))

# Load model instance
pipe_lr = load_emotion_model()

# Manual Overrides for improved accuracy
KEYWORD_OVERRIDES = {
    "stressed": "sadness", "stress": "sadness",
    "depressed": "sadness", "depression": "sadness",
    "anxious": "fear", "anxiety": "fear",
    "panicked": "fear", "panic": "fear",
    "afraid": "fear", "scared": "fear",
    "angry": "anger", "furious": "anger", "mad": "anger",
    "sad": "sadness", "grief": "sadness", "pain": "sadness"
}

def get_override(text):
    """Checks text for keywords that trigger a manual override."""
    text_lower = text.lower()
    for word, emotion in KEYWORD_OVERRIDES.items():
        # Basic containment check - can be improved with regex
        if word in text_lower:
            return emotion
    return None

def predict_emotions(text):
    """
    Predicts emotion and returns a structured dictionary.
    Returns:
        dict: {
            "emotion": str,
            "confidence": float,
            "probabilities": dict {emotion: score}
        }
    """
    if pipe_lr is None:
        return {"emotion": "Error", "confidence": 0.0, "probabilities": {}}

    # 1. Check for manual override
    override = get_override(text)
    classes = pipe_lr.classes_
    
    if override:
        # Create synthetic 1.0 confidence for the override emotion
        probs = {c: 0.0 for c in classes}
        if override in classes:
            probs[override] = 1.0
        
        return {
            "emotion": override,
            "confidence": 1.00,
            "probabilities": probs
        }

    # 2. Model Prediction
    prediction = pipe_lr.predict([text])[0]
    proba_list = pipe_lr.predict_proba([text])[0]
    
    # Create probability dictionary
    probs = {c: p for c, p in zip(classes, proba_list)}
    
    # Get confidence of the predicted class
    confidence = probs.get(prediction, 0.0)

    return {
        "emotion": prediction,
        "confidence": confidence,
        "probabilities": probs
    }
