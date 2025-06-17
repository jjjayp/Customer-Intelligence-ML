from sklearn.linear_model import LogisticRegression
from sklift.models import TwoModels
from sklift.metrics import qini_auc_score

def train_x_learner(X_train, T_train, y_train):
    model_treated = LogisticRegression()
    model_control = LogisticRegression()

    x_learner = TwoModels(
        estimator_trmnt=model_treated,
        estimator_ctrl=model_control,
        method='x_learner'
    )

    x_learner.fit(X_train, y_train, T_train)
    return x_learner

def evaluate_x_learner(model, X_test, y_test, T_test):
    uplift_scores = model.predict(X_test)
    auc_score = qini_auc_score(y_test, uplift_scores, T_test)
    return uplift_scores, auc_score