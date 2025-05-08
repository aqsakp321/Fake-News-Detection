# Import libraries
import streamlit as st
import joblib
import re
import string
import spacy
import json
import time
import requests
import streamlit_lottie as st_lottie

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("FND_linear_svc_model_india.pkl")
    vectorizer = joblib.load("FND_vectorizer_india.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Text preprocessing
def preprocess(text):
    text = text.lower().replace('’', "'")
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r"\d+", '[NUMBER]', text)
    text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 2]
    return ' '.join(lemmatized)

# Load Lottie animation from URL
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animation
lottie_news = load_lottie_url("https://lottie.host/cf02946c-7ce0-474e-8908-babcefc90f84/tAUhORByT7.json")

# Custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Raleway', sans-serif;
            background: linear-gradient(135deg, #f6f9fc, #e0f7fa);
        }

        .stApp {
            background: linear-gradient(135deg, #f6f9fc, #e0f7fa);
            padding: 2rem;
        }

        .main-title {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            animation: fadeInDown 1s ease-out;
        }

        .subtext {
            text-align: center;
            font-size: 1.2rem;
            color: #34495e;
            margin-bottom: 30px;
            animation: fadeIn 2s ease-in;
        }

        .stTextArea textarea {
            border-radius: 0.75rem;
            border: 2px solid #dee2e6;
            transition: 0.3s ease;
        }

        .stTextArea textarea:hover {
            border-color: #66afe9;
        }

        .stButton > button {
            background: linear-gradient(to right, #36d1dc, #5b86e5);
            color: white;
            font-weight: bold;
            padding: 0.6em 1.2em;
            border-radius: 0.5rem;
            border: none;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .result-box {
            padding: 1.5rem;
            border-radius: 1rem;
            font-size: 1.5rem;
            font-weight: 600;
            text-align: center;
            animation: fadeIn 0.8s ease-in-out;
        }

        .real {
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #10b981;
        }

        .fake {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #ef4444;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom right, #f9f9f9, #e0f7fa);
            border-right: 2px solid #e3f2fd;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# Lottie Animation
st_lottie.st_lottie(lottie_news, speed=1, loop=True, quality="high", height=250)

# App Title
st.markdown('<div class="main-title">📰 Fake News Detector</div>', unsafe_allow_html=True)

# Subheading
st.markdown('<div class="subtext">Analyze the authenticity of a news article using a trained SVC model.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🔍 Analyze", "📊 Insights"])

with tab1:
    user_input = st.text_area("✍️ Enter News Article Text Below:", height=200)
    if st.button("🔍 Analyze"):
        if user_input.strip():
            with st.spinner("Analyzing... please wait..."):
                time.sleep(1)
                processed = preprocess(user_input)
                vector = vectorizer.transform([processed])
                prediction = model.predict(vector)[0]

            if prediction == 1:
                st.markdown('<div class="result-box real">✅ The news appears to be <b>REAL</b>.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box fake">❌ The news appears to be <b>FAKE</b>.</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

with tab2:
    st.markdown("## 🧠 Model Insights")
    st.info("Coming soon: analytics on prediction distribution, word importance, and model performance.")

# Footer
st.markdown("""
    <hr style="margin-top: 3rem; border-top: 1px dashed #bbb;">
    <div style="text-align:center; font-size: 0.9rem; color: #555;">
        Built with ❤️ using <b>Streamlit</b> & <b>SVC ML Model</b> <br>
        <a href="https://github.com/yourprofile" style="color:#5b86e5; text-decoration:none;" target="_blank">GitHub</a> |
        <a href="mailto:your@email.com" style="color:#5b86e5; text-decoration:none;" target="_blank">Contact</a>
    </div>
""", unsafe_allow_html=True)
