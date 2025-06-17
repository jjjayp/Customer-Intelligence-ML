from uplift_modeling.x_learner import XLearner
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklift.metrics import qini_auc_score
import pandas as pd

# Assume your train/test variables are already defined:
# X_train_final, y_train, T_train, X_test_final, y_test, T_test

base_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
x_learner = XLearner(base_model, base_model, base_model)

x_learner.fit(X_train_final.values, y_train.values, T_train.values)
uplift_preds = x_learner.predict(X_test_final.values)

# Evaluate
print("Qini AUC:", qini_auc_score(y_test, uplift_preds, T_test))