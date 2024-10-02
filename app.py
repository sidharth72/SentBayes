import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
import joblib
from preprocess import preprocess_text, download_nltk_data
from predict import predict

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="centered"
)


# Ensure NLTK data is downloaded
download_nltk_data()

# Load the vectorizer
vectorizer = joblib.load('model/tfidf_vec.pkl')

# Page config
# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .example-button {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.2s;
    }
    .example-button:hover {
        background-color: #e0e2e6;
        border-color: #d0d0d0;
    }
    .sentiment-box {
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .positive {
        background-color: #dcffe4;
        border: 1px solid #00cc44;
    }
    .negative {
        background-color: #ffe0e0;
        border: 1px solid #ff4444;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title('🎭 Sentiment Analyzer')

# Initialize session state for text input
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

# Example texts
examples = [
    "I absolutely loved the movie! The acting was superb and the plot kept me engaged throughout.",
    "The customer service was terrible. I waited for hours and still didn't get my issue resolved.",
    "The new restaurant in town is okay. The food is decent but a bit overpriced.",
    "I can't believe how amazing this product is! It has completely changed my life for the better."
]

# Two columns layout
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.text_input,
        height=150,
        key="text_input_area"
    )

with col2:
    st.write("**Try an example:**")
    for i, example in enumerate(examples):
        if st.button(f"Example {i+1}", key=f"example_{i}"):
            st.session_state.text_input = example
            st.experimental_rerun()

# Analyze button
if st.button('Analyze Sentiment', type='primary'):
    if text_input.strip() == '':
        st.error('Please enter some text.')
    else:
        with st.spinner('Analyzing...'):
            # Preprocess and predict
            preprocessed_text = preprocess_text(text_input)
            preprocessed_text_counts = vectorizer.transform([preprocessed_text])
            predicted_sentiment = predict(preprocessed_text_counts)[0]
            
            # Display result
            sentiment_html = f"""
            <div class="sentiment-box {'positive' if predicted_sentiment == 'Positive' else 'negative'}">
                <h3>Sentiment: {predicted_sentiment}</h3>
                <p>{'😊 The text appears to be positive.' if predicted_sentiment == 'Positive' else '😔 The text appears to be negative.'}</p>
            </div>
            """
            st.markdown(sentiment_html, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About")
    st.info("""
    This app uses a Naive Bayes model to analyze the sentiment of text.
    Enter your text or try one of the examples to see how it works!
    """)
    
    st.title("How it works")
    st.write("""
    1. Enter text or select an example
    2. Click 'Analyze Sentiment'
    3. Get instant sentiment prediction
    """)
    
    st.title("Created by")
    st.info("Sidharth")