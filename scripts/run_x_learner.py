import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uplift_modeling')))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from x_learner import XLearner
from simulation import simulate_free_delivery_subscription_extended


def run_custom_x_learner():
    df = simulate_free_delivery_subscription_extended(n_samples=5000)

    # Encode 'region' column
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded_region = encoder.fit_transform(df[['region']])
    encoded_region_df = pd.DataFrame(encoded_region, columns=encoder.get_feature_names_out(['region']), index=df.index)

    # Replace 'region' column with encoded version
    X = df.drop(columns=['subscription', 'treatment', 'region'])
    X = pd.concat([X, encoded_region_df], axis=1)

    T = df['treatment']
    y = df['subscription']


    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
        X, T, y, test_size=0.2, random_state=42
    )

    model_treat = RandomForestRegressor(n_estimators=100)
    model_control = RandomForestRegressor(n_estimators=100)
    final_model = RandomForestRegressor(n_estimators=100)

    x_learner = XLearner(model_treat, model_control, final_model)
    x_learner.fit(X_train, y_train, T_train)

    uplift_preds = x_learner.predict(X_test)
    print("Sample uplift predictions:", uplift_preds[:10])


if __name__ == "__main__":
    run_custom_x_learner()