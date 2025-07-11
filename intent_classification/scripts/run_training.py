import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model():
    X_train, y_train = joblib.load("data/train.pkl")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    train_model()
