import streamlit as st
from src.ui import (
    load_css, render_hero, render_tabs, 
    render_input_card, render_analyze_button, 
    render_result_card, render_insights_card
)
from src.models import predict_emotions
from src.analysis import get_ai_analysis
from src.utils import google_search

# Configure Page
st.set_page_config(
    page_title="MediMind",
    page_icon="🧠",
    layout="centered", # Better for SaaS look than wide
    initial_sidebar_state="collapsed"
)

def main():
    # 1. Load Design System
    load_css()
    
    # 2. Render Hero (Title & Subtitle)
    render_hero()
    
    # 3. Main Navigation (Tabs)
    tabs, mode_map = render_tabs()
    
    # 4. Render Content per Tab
    # Streamlit tabs act as context managers
    for i, tab in enumerate(tabs):
        with tab:
            mode = mode_map[i]
            
            # Input Section
            user_input = render_input_card(mode, key_suffix=mode)
            
            # Analyze Action
            if user_input:
                if render_analyze_button():
                    with st.spinner("🧠 Analyzing emotional patterns..."):
                        # A. Emotion Prediction
                        result = predict_emotions(user_input)
                        
                        # B. Render Core Result
                        render_result_card(result)
                        
                        # C. AI Insights
                        with st.spinner("💡 Generating insights..."):
                            ai_response = get_ai_analysis(
                                result["emotion"], 
                                result["confidence"], 
                                user_input
                            )
                            render_insights_card(ai_response)
                        
                        # D. Search Results (Specific to Search Mode or as extra)
                        if mode == "Search":
                            st.subheader("🌐 Related Resources")
                            search_results = google_search(user_input)
                            if search_results:
                                for item in search_results[:3]:
                                    with st.container():
                                        st.markdown(f"**[{item['title']}]({item['link']})**")
                                        st.caption(item.get('snippet', ''))

            elif mode == "Audio":
                # Audio placeholder handled in input card
                pass
            else:
                st.info("👋 Enter text above to begin analysis.")

if __name__ == "__main__":
    main()
