import numpy as np
import pandas as pd

def generate_loan_data(num_samples=10000, random_state=42):
    """
    Generates a realistic synthetic loan portfolio with non-linear credit risk dynamics.
    
    Features:
    - FICO_score: Realistic credit bureau distribution with prime cluster and subprime left tail [300, 850]
    - income: Lognormal annual gross income ($15k - $500k+)
    - loan_amount: Requested loan principal ($1k - $50k)
    - dti: Debt-to-income ratio (0.05 - 0.60)
    - employment_length: Years of employment (0 - 40)
    - default: Observed binary default indicator (0 or 1) generated from calibrated log-odds
    """
    if num_samples <= 0:
        return pd.DataFrame(columns=[
            'FICO_score', 'income', 'loan_amount', 'dti', 'employment_length', 'default'
        ])
        
    rng = np.random.default_rng(random_state)
    
    # Realistic credit bureau FICO distribution (mixture of subprime tail ~25% and prime mass ~75%)
    subprime_draws = rng.normal(loc=590, scale=60, size=num_samples)
    prime_draws = rng.normal(loc=720, scale=55, size=num_samples)
    is_subprime = rng.random(size=num_samples) < 0.25
    fico_scores = np.where(is_subprime, subprime_draws, prime_draws)
    fico_scores = np.clip(fico_scores, 300, 850).astype(int)
    
    # Financial covariates
    income = rng.lognormal(mean=11.0, sigma=0.5, size=num_samples)
    loan_amount = rng.uniform(low=1000, high=50000, size=num_samples)
    dti = rng.uniform(low=0.05, high=0.60, size=num_samples)
    employment_length = rng.integers(low=0, high=40, size=num_samples)
    
    # Non-linear credit risk tiers (captures credit score cliffs in underwriting)
    fico_tier_penalty = np.where(
        fico_scores < 580, 1.0,
        np.where(fico_scores < 660, 0.35,
        np.where(fico_scores >= 740, -0.5, 0.0))
    )
    
    # Latent credit risk score (log-odds of default)
    base_score = (
        -0.012 * (fico_scores - 650)
        + fico_tier_penalty
        - 0.00001 * income
        + 0.00005 * loan_amount
        + 4.5 * dti
        - 0.08 * employment_length
        - 0.80  # Baseline intercept balancing default rate to ~35-40%
    )
    
    # Latent Gaussian noise added directly to log-odds (proper GLM specification)
    latent_score = base_score + rng.normal(loc=0.0, scale=0.35, size=num_samples)
    
    # Calibrated logistic sigmoid link function
    prob_default = 1.0 / (1.0 + np.exp(-latent_score))
    
    # Bernoulli default realization
    defaults = rng.binomial(n=1, p=prob_default)
    
    df = pd.DataFrame({
        'FICO_score': fico_scores,
        'income': np.round(income, 2),
        'loan_amount': np.round(loan_amount, 2),
        'dti': np.round(dti, 4),
        'employment_length': employment_length,
        'default': defaults
    })
    
    return df

if __name__ == "__main__":
    df = generate_loan_data(10000, random_state=42)
    df.to_csv("loan_data.csv", index=False)
    print(f"Generated loan_data.csv with {len(df)} records.")
    print(f"Default rate: {df['default'].mean():.2%}")
    print("FICO summary:", df['FICO_score'].describe())
