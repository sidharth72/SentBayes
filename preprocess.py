
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in stop_words]
    #stemmed_tokens = [token for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text