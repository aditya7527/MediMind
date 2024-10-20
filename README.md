# 🧠 Advanced Text Emotion Detection & Summarization App

## Overview

The **Emotion Detection & Summarization App** is a powerful web application built with Streamlit that enables users to analyze the emotional tone of text, emails, articles, and audio inputs. The app utilizes various AI models to provide detailed emotional analysis and generate empathetic responses, making it an excellent tool for understanding emotional contexts in communication.

### Features

- **Text Emotion Detection**: Input any text to analyze its emotional tone.
- **Email Emotion Analysis**: Paste email content for emotional context understanding.
- **Article URL Extraction**: Provide a URL to extract and analyze article content.
- **Google Search Integration**: Search for articles and analyze their emotional content.
- **Audio Input**: Speak your text, and the app will analyze emotions in your spoken words.
- **AI Emotional Analysis**: Receive detailed emotional analysis and suggestions based on the detected tone.
- **Interactive Visualizations**: View emotional probabilities in a bar chart format.
- **Downloadable Results**: Download analysis results as a CSV file.

### Technologies Used

- **Streamlit**: For building the web application.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Altair**: For creating interactive data visualizations.
- **Joblib**: For loading machine learning models.
- **BeautifulSoup**: For web scraping article content.
- **Google API**: For conducting Google searches.
- **Together API**: For AI emotional analysis.
- **Cerebras**: For advanced AI model interactions.
- **gTTS**: For converting text to speech.
- **Langdetect**: For language detection.
- **SoundDevice & Pydub**: For playing audio.
- **Streamlit Mic Recorder**: For capturing audio input.

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed. You can download it from [python.org](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

```

## Setup Instructions

### Creating and Using a Virtual Environment

A virtual environment helps manage project-specific dependencies and avoids conflicts with other Python projects. Follow these steps to create and use a virtual environment for this project:

#### 1. Create a Virtual Environment

Open a terminal or command prompt and navigate to your project directory. Run the following command to create a virtual environment named `venv`:

```sh
python -m venv venv
```
#### 2. Activating the Virtual Environment
```
venv\Scripts\activate
```

### 3. Download all the dependencies and Libraries

```
pip install -r requirements.txt

```

### 4. Run the File
```
streamlit run <filename>.py
```
## Note:
### Make sure to use APIs in the code.

## License

### This project is licensed under the MIT License. See the LICENSE file for more details.


