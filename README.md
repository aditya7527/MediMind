# 🧠 MediMind: Advanced AI Emotion Analysis

MediMind is a powerful Streamlit application that uses Machine Learning and Large Language Models (LLMs) to analyze the emotional tone of text, audio, and articles. It provides deep psychological insights and empathetic responses.

## ✨ Features

- **Multi-Modal Input**: Analyze text, audio (speech-to-text), emails, and article URLs.
- **Emotion Detection**: Classifies content into emotions like Joy, Sadness, Anger, Fear, etc.
- **AI Analysis**: Uses Llama-3 (via Together AI) to provide a detailed psychological breakdown of the user's state.
- **Empathetic Response**: Generates supportive and constructive advice.
- **Audio Playback**: Listen to the AI's analysis.
- **Web Search**: Integrated Google Search for finding relevant resources.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- API Keys for:
    - [Together AI](https://api.together.xyz/)
    - [Google Custom Search](https://developers.google.com/custom-search/v1/overview)
    - [OpenAI](https://platform.openai.com/) (Optional, for additional analysis)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/aditya7527/MediMind.git
    cd MediMind
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up secrets**:
    Create a `.env` file in the root directory (or use Streamlit secrets `.streamlit/secrets.toml`):

    ```ini
    TOGETHER_API_KEY=your_together_api_key
    GOOGLE_API_KEY=your_google_api_key
    GOOGLE_CX=your_google_cse_id
    OPENAI_API_KEY=your_openai_api_key
    ```

### Running the App

```bash
streamlit run app.py
```

## 📂 Project Structure

```
root/
├── app.py              # Main entry point
├── src/                # core logic
│   ├── config.py       # Configuration & Keys
│   ├── models.py       # ML Model loading
│   ├── analysis.py     # AI Integration
│   ├── ui.py           # UI Components
│   └── utils.py        # Helper functions
├── assets/             # Static assets
│   └── style.css       # Custom styling
├── model/              # Pre-trained Emotion Models
└── data/               # Data files
```

## 🛠️ Built With

- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Together AI](https://www.together.ai/)
- [Altair](https://altair-viz.github.io/)

## 📄 License

This project is licensed under the MIT License.
