import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st

# Download required NLTK data
@st.cache_resource  # This decorator ensures the download only happens once
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Call the download function
download_nltk_data()

# Now define your preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text