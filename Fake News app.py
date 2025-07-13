from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob

app = Flask(__name__)

def train_model():
    df = pd.read_csv('news.csv')
    df.dropna(subset=['text'], inplace=True)
    df['text'] = df['text'].str.lower()
    unique_words = set(word for text in df['text'] for word in text.split())
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, vectorizer, unique_words

model, vectorizer, unique_words = train_model()

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('message') or request.json.get('message', '')
        if not text:
            return jsonify({'error': 'No text provided'}) if request.is_json else render_template('index.html', error="Please enter some text")
        
        corrected_text = str(TextBlob(text.lower()).correct())
        input_words = set(corrected_text.split())
        
        if not input_words.intersection(unique_words):
            return jsonify({'error': 'Input text is invalid or unrelated to the dataset'}) if request.is_json else render_template('index.html', error="Invalid input: No relation to dataset vocabulary")
        
        text_vector = vectorizer.transform([corrected_text])
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        confidence = probability[0] if prediction == 'FAKE' else probability[1]

        result = {
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'corrected_text': corrected_text,
            'original_text': text
        }

        if request.is_json:
            return jsonify(result)
        return render_template('index.html', **result)

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return jsonify({'error': error_msg}) if request.is_json else render_template('index.html', error=error_msg)

if __name__ == '__main__':
    app.run(debug=True)
