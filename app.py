import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
from preprocess import preprocess_text, download_nltk_data
from predict import predict


# Set page config
st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="wide")


# Ensure NLTK data is downloaded
download_nltk_data()

# Load the vectorizer
vectorizer = joblib.load('model/tfidf_vec.pkl')

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main {
        background: #ffffff;
        padding: 3rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit app
st.title('🎭 Sentiment Analysis with Naive Bayes')

# Create tabs
tab1, tab2 = st.tabs(["Analyze Text", "Try Examples"])

with tab1:
    st.header("Analyze Your Own Text")
    # Text input area
    text_input = st.text_area('Enter text:', '', height=150)

    # Predict button
    if st.button('Predict Sentiment'):
        if text_input.strip() == '':
            st.error('Please enter some text.')
        else:
            with st.spinner('Analyzing...'):
                # Preprocess the input text
                preprocessed_text = preprocess_text(text_input)
                
                # Vectorize the preprocessed text
                preprocessed_text_counts = vectorizer.transform([preprocessed_text])
                
                # Predict sentiment
                predicted_sentiment = predict(preprocessed_text_counts)
                
                # Display predicted sentiment
                if predicted_sentiment[0] == 'Positive':
                    st.success(f'Predicted Sentiment: {predicted_sentiment[0]} 😊')
                else:
                    st.warning(f'Predicted Sentiment: {predicted_sentiment[0]} 😔')

with tab2:
    st.header("Try These Examples")
    examples = [
        "I absolutely loved the movie! The acting was superb and the plot kept me engaged throughout.",
        "The customer service was terrible. I waited for hours and still didn't get my issue resolved.",
        "The new restaurant in town is okay. The food is decent but a bit overpriced.",
        "I can't believe how amazing this product is! It has completely changed my life for the better."
    ]
    
    for i, example in enumerate(examples, 1):
        if st.button(f"Example {i}"):
            st.text_area("Text:", example, height=100)
            with st.spinner('Analyzing...'):
                preprocessed_text = preprocess_text(example)
                preprocessed_text_counts = vectorizer.transform([preprocessed_text])
                predicted_sentiment = predict(preprocessed_text_counts)
                
                if predicted_sentiment[0] == 'Positive':
                    st.success(f'Predicted Sentiment: {predicted_sentiment[0]} 😊')
                else:
                    st.warning(f'Predicted Sentiment: {predicted_sentiment[0]} 😔')

# Add information about the model
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a Naive Bayes model to predict the sentiment of text. "
    "It classifies text as either Positive or Negative."
)

# Add your name or team name
st.sidebar.title("Created by")
st.sidebar.info(
    "Sidharth"
)

# You can add more sections like this to the sidebar
st.sidebar.title("Data Source & Code")
st.sidebar.info(
    "You can find more information about the dataset and code on the github repo"
)