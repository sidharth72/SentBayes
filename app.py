import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
from preprocess import preprocess_text, download_nltk_data
from predict import predict

# Ensure NLTK data is downloaded
download_nltk_data()

# Load the vectorizer
vectorizer = joblib.load('model/tfidf_vec.pkl')

# Page configuration
st.set_page_config(
    page_title="SentBayes",
    page_icon="🎭",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Headers */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #1E1E1E;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #1E1E1E;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        background-color: #1E1E1E;
        color: white;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #333333;
        border: none;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 4px;
        border-color: #E0E0E0;
    }
    
    /* Card-like container for results */
    .sentiment-result {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-top: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Main app layout
st.title('SentBayes')
st.markdown("##### A minimalist sentiment analysis tool")

# Create tabs
tab1, tab2 = st.tabs(["✏️ Analyze", "🔍 Examples"])

def predict_and_display(text):
    with st.spinner('Analyzing...'):
        preprocessed_text = preprocess_text(text)
        preprocessed_text_counts = vectorizer.transform([preprocessed_text])
        predicted_sentiment = predict(preprocessed_text_counts)
        
        st.markdown(f"""
        <div class="sentiment-result">
            <h3 style="margin-bottom: 0.5rem;">Result</h3>
            <p style="font-size: 18px; margin-bottom: 0.5rem;">{predicted_sentiment[0]} 
            {'😊' if predicted_sentiment[0] == 'Positive' else '😔'}</p>
        </div>
        """, unsafe_allow_html=True)

with tab1:
    text_input = st.text_area('Enter your text here:', height=150, 
                              placeholder="Type or paste your text for analysis...")
    
    if st.button('Analyze Sentiment'):
        if text_input.strip():
            predict_and_display(text_input)
        else:
            st.error('Please enter some text.')

with tab2:
    examples = [
        "I absolutely loved the movie! The acting was superb.",
        "The customer service was terrible. I waited for hours.",
        "The new restaurant is okay. The food is decent but overpriced.",
        "This product has completely changed my life for the better!"
    ]
    
    for i, example in enumerate(examples, 1):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_area(f"Example {i}", example, height=100, key=f"example_{i}")
        with col2:
            if st.button('Try', key=f"button_{i}"):
                predict_and_display(example)

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("SentBayes uses Naive Bayes to classify text sentiment as Positive or Negative.")
    
    st.markdown("### Accuracy")
    st.markdown("88% accuracy on test data")
    
    st.markdown("### Creator")
    st.markdown("Sidharth")
    
    st.markdown("### Links")
    st.markdown("[GitHub Repository](https://github.com/yourusername/sentbayes)")