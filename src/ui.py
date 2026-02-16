import streamlit as st
import pandas as pd
import altair as alt
from .config import APP_TITLE, APP_SUBTITLE, EMOTION_EMOJI_MAP
from .utils import extract_article_content

def load_css():
    """Injects the custom CSS."""
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_hero():
    """Renders the Hero section with title and subtitle."""
    st.markdown(f"""
        <div class="hero-container">
            <h1 class="hero-title">🧠 {APP_TITLE}</h1>
            <p class="hero-subtitle">{APP_SUBTITLE}</p>
        </div>
    """, unsafe_allow_html=True)

def render_tabs():
    """Renders the navigation tabs and returns the selected mode."""
    # Using Streamlit's native tabs which we styled in CSS to look like pills
    tab_names = ["📝 Text", "🎙️ Audio", "📧 Email", "📰 Article", "🌐 Search"]
    tabs = st.tabs(tab_names)
    
    # Map index to mode string for logic
    mode_map = {
        0: "Text", 1: "Audio", 2: "Email", 3: "Article", 4: "Search"
    }
    
    # Return both the active tab container and the mode name
    # We find which tab is active by context (streamlit executes the block within the tab)
    return tabs, mode_map

def render_input_card(mode, key_suffix=""):
    """
    Renders the appropriate input mechanism based on mode within a glass card.
    Returns the user input text or None.
    """
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    input_text = None
    
    if mode in ["Text", "Email"]:
        placeholder = "How are you feeling right now?" if mode == "Text" else "Paste email content here..."
        input_text = st.text_area(
            label="Input Text", 
            placeholder=placeholder, 
            height=150, 
            label_visibility="collapsed",
            key=f"text_input_{key_suffix}"
        )
        
    elif mode == "Article":
        article_url = st.text_input(
            "Article URL", 
            placeholder="Paste article link (e.g., https://medium.com/...)",
            label_visibility="collapsed",
            key=f"url_input_{key_suffix}"
        )
        if article_url:
            with st.spinner("Extracting content..."):
                input_text = extract_article_content(article_url)
                if input_text and not input_text.startswith("Error"):
                    st.success("Article extracted successfully!")
                    with st.expander("View Content"):
                        st.write(input_text[:500] + "...")
                elif input_text:
                    st.error(input_text)
                    input_text = None

    elif mode == "Search":
        search_query = st.text_input(
            "Search Query", 
            placeholder="What would you like to research?",
            label_visibility="collapsed",
            key=f"search_input_{key_suffix}"
        )
        if search_query:
            # For search mode, usually we return result text, but here we might want to trigger search flow
            # For consistency in this refactor, let's treat query as text to analyze for emotion (intent)
            # OR pass it back for the controller to handle search logic.
            # Let's return the query as input_text for now.
            input_text = search_query

    elif mode == "Audio":
        st.info("🎙️ Audio recording feature coming soon! (Use Text mode for now)")
        # Placeholder for audio input if we integrate it later
        input_text = None

    st.markdown('</div>', unsafe_allow_html=True)
    return input_text

def render_analyze_button():
    """Renders the main action button."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        return st.button("✨ Analyze Emotion", use_container_width=True)

def render_result_card(result):
    """
    Renders the emotion analysis results in a highlighted glass card.
    """
    emotion = result["emotion"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]
    
    emoji = EMOTION_EMOJI_MAP.get(emotion, "😐")
    
    st.markdown(f"""
        <div class="glass-card emotion-result-container">
            <div class="emotion-emoji">{emoji}</div>
            <div class="emotion-label">{emotion.capitalize()}</div>
            <p style="color: #cbd5e1; margin-top: 5px;">Confidence: {confidence*100:.1f}%</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Progress bar with custom color
    st.progress(confidence)
    
    # Detailed Probability Chart
    with st.expander("📊 View Detailed Probabilities"):
        df = pd.DataFrame(list(probabilities.items()), columns=['Emotion', 'Probability'])
        chart = alt.Chart(df).mark_bar().encode(
            x='Probability',
            y=alt.Y('Emotion', sort='-x'),
            color=alt.Color('Probability', scale=alt.Scale(scheme='magma')),
            tooltip=['Emotion', 'Probability']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

def render_insights_card(ai_response):
    """Renders the AI analysis text in a structured card."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 AI Insights")
    st.markdown("---")
    st.markdown(ai_response)
    st.markdown('</div>', unsafe_allow_html=True)
