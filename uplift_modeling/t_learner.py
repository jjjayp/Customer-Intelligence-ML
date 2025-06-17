from sklearn.linear_model import LogisticRegression

def train_t_learner(X, y, treatment):
    treat_idx = treatment == 1
    control_idx = treatment == 0

    model_treat = LogisticRegression()
    model_control = LogisticRegression()

    model_treat.fit(X[treat_idx], y[treat_idx])
    model_control.fit(X[control_idx], y[control_idx])

    return model_treat, model_control