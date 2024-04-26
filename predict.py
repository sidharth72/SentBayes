import joblib

model = joblib.load('model/naive_bayes.pkl')

def predict(input_vector):
    prediction = model.predict(input_vector)
    return prediction


