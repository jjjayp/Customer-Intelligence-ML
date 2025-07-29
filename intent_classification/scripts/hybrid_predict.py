import joblib
from transformers import pipeline

# Load both models
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
hf_classifier = pipeline("sentiment-analysis")

def tfidf_predict(text):
    vec = vectorizer.transform([text.lower()])
    return model.predict(vec)[0]

def hf_predict(text):
    result = hf_classifier(text)[0]
    return result['label'].lower()

def hybrid_predict(text):
    # Keyword override
    keywords = ["frustrated", "angry", "mad", "hate", "terrible", "help", "cancel"]
    if any(kw in text.lower() for kw in keywords):
        return "negative (rule)"
    
    # Else, fallback to HF prediction
    return hf_predict(text)

if __name__ == "__main__":
    import sys
    text = sys.argv[1]
    print(hybrid_predict(text))