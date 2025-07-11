import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def clean_text(text):
    return text.lower()

def preprocess_and_save():
    df = pd.read_csv("intent_classification/data/raw_chat_data.csv")
    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["label"]  # sentiment or intent class

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    joblib.dump((X_train_vec, y_train), "data/train.pkl")
    joblib.dump((X_test_vec, y_test), "data/test.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

if __name__ == "__main__":
    preprocess_and_save()