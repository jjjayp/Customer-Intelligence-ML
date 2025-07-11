import joblib
import sys

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict(text):
    vec = vectorizer.transform([text.lower()])
    print(f"Vectorized input shape: {vec.shape}")
    return model.predict(vec)[0]


if __name__ == "__main__":
    text = sys.argv[1]
    print(predict(text))