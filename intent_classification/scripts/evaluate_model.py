import joblib
from sklearn.metrics import classification_report
import time
import numpy as np

def evaluate():
    model = joblib.load("models/model.pkl")
    X_test, y_test = joblib.load("data/test.pkl")

    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    latency_ms = (end - start) * 1000 / len(y_pred)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Avg Inference Latency per sample: {latency_ms:.2f} ms")

if __name__ == "__main__":
    evaluate()