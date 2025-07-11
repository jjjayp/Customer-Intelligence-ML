from transformers import pipeline
import sys

# Load sentiment pipeline
classifier = pipeline("sentiment-analysis")

def predict(text):
    result = classifier(text)[0]
    print(f"{result['label']} ({result['score']:.2f})")


if __name__ == "__main__":
    text = sys.argv[1]
    label, confidence = predict(text)
    print(f"{label} ({confidence:.2f})")