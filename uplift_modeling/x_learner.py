from sklearn.base import clone
import numpy as np

class XLearner:
    def __init__(self, model_treat, model_control, final_model):
        self.model_treat = clone(model_treat)
        self.model_control = clone(model_control)
        self.final_model_0 = clone(final_model)
        self.final_model_1 = clone(final_model)

    def fit(self, X, y, treatment):
        treat_idx = treatment == 1
        control_idx = treatment == 0

        X_treat, y_treat = X[treat_idx], y[treat_idx]
        X_control, y_control = X[control_idx], y[control_idx]

        self.model_treat.fit(X_treat, y_treat)
        self.model_control.fit(X_control, y_control)

        mu0 = self.model_control.predict(X_treat)
        D1 = y_treat - mu0

        mu1 = self.model_treat.predict(X_control)
        D0 = mu1 - y_control

        self.final_model_1.fit(X_treat, D1)
        self.final_model_0.fit(X_control, D0)

    def predict(self, X):
        tau0 = self.final_model_0.predict(X)
        tau1 = self.final_model_1.predict(X)
        propensity = 0.5
        return (1 - propensity) * tau0 + propensity * tau1