import numpy as np
import pandas as pd
from scipy.special import expit

def simulate_free_delivery_subscription_extended(n_samples=5000, seed=42):
    np.random.seed(seed)
    age = np.random.randint(18, 70, size=n_samples)
    purchase_freq = np.random.poisson(lam=2, size=n_samples)
    avg_spend = np.round(np.random.gamma(shape=2.0, scale=50.0, size=n_samples), 2)
    gender = np.random.binomial(1, p=0.5, size=n_samples)
    regions = np.random.choice(['Urban', 'Suburban', 'Rural'], size=n_samples, p=[0.5, 0.3, 0.2])
    treatment = np.random.binomial(1, 0.5, size=n_samples)

    region_effect_map = {'Urban': 0.3, 'Suburban': 0.2, 'Rural': 0.0}
    region_effect = np.array([region_effect_map[r] for r in regions])

    log_odds_baseline = (-4.0 
                         - 0.02 * age 
                         + 0.5 * purchase_freq 
                         + 0.01 * avg_spend 
                         + 0.1 * gender 
                         + region_effect)

    treatment_effect = 1.0
    log_odds = log_odds_baseline + treatment * treatment_effect
    prob_subscription = expit(log_odds)
    subscription = np.random.binomial(1, prob_subscription)

    df = pd.DataFrame({
        'age': age,
        'purchase_freq': purchase_freq,
        'avg_spend': avg_spend,
        'gender': gender,
        'region': regions,
        'treatment': treatment,
        'subscription': subscription
    })

    return df