import streamlit as st
import joblib
from preprocess import preprocess_text, download_nltk_data
from predict import predict

# Set page config
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🎭", layout="centered")

# Ensure NLTK data is downloaded
download_nltk_data()

# Load the vectorizer
vectorizer = joblib.load('model/tfidf_vec.pkl')

# Custom CSS for a modern, minimalistic look
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit app
st.title('🎭 Sentiment Analysis')

# Text input area
text_input = st.text_area('Enter text to analyze:', height=150)

# Sample texts
st.subheader("Or try a sample:")
samples = {
    "Positive": "I absolutely loved the movie! The acting was superb.",
    "Negative": "The customer service was terrible. I'm very disappointed.",
    "Neutral": "The weather today is okay, not great but not bad either."
}

for sentiment, sample in samples.items():
    if st.button(sentiment):
        text_input = sample
        st.experimental_rerun()

# Predict button
if st.button('Predict Sentiment'):
    if text_input.strip() == '':
        st.error('Please enter some text.')
    else:
        with st.spinner('Analyzing...'):
            # Preprocess and predict
            preprocessed_text = preprocess_text(text_input)
            preprocessed_text_counts = vectorizer.transform([preprocessed_text])
            predicted_sentiment = predict(preprocessed_text_counts)
            
            # Display predicted sentiment
            sentiment_emoji = {"Positive": "😊", "Negative": "😔", "Neutral": "😐"}
            st.success(f'Predicted Sentiment: {predicted_sentiment[0]} {sentiment_emoji.get(predicted_sentiment[0], "")}')

# Sidebar information
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a Naive Bayes model to predict the sentiment of text. "
    "It classifies text as either Positive, Negative, or Neutral."
)

st.sidebar.title("Created by")
st.sidebar.info("Sidharth")

st.sidebar.title("More Information")
st.sidebar.info(
    "For more information about the dataset and code, "
    "please visit the [GitHub repository](your_github_repo_link)."
)