from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained SVM model and TF-IDF vectorizer
model = joblib.load('models/svm_model.pkl') 
vectorizer = joblib.load('models/tfidf_vectorization.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['reviewText']
        # Transform the input text using the same TF-IDF vectorizer
        text_vector = vectorizer.transform([text])
        # Predict sentiment using the SVM model
        prediction = model.predict(text_vector)
        
        # Map the result to human-readable text
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        # Pass prediction to the template
        return render_template('index.html', prediction=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
