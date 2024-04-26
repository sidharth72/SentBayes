import streamlit as st
import pandas as pd
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import joblib
from preprocess import preprocess_text
from predict import predict


vectorizer = joblib.load('model/tfidf_vec.pkl')

# Streamlit app
st.title('Naive Bayes Sentiment Analysis')

# Text input area
text_input = st.text_area('Enter text:', '')

# Predict button
if st.button('Predict'):
    if text_input.strip() == '':
        st.error('Please enter some text.')
    else:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text_input)
        
        # Vectorize the preprocessed text
        preprocessed_text_counts = vectorizer.transform([preprocessed_text])
        
        # Predict sentiment
        predicted_sentiment = predict(preprocessed_text_counts)
        
        # Display predicted sentiment
        if predicted_sentiment[0] == 'Positive':
            st.success(f'Predicted Sentiment: {predicted_sentiment[0]}')
        else:
            st.warning(f'Predicted Sentiment: {predicted_sentiment[0]}')