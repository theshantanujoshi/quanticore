"""
Shared test configuration, fixtures, and relative path helpers for Quantitative Credit Risk tests.
All path references must strictly adhere to relative paths (e.g. data/...).
"""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from preprocessing import build_preprocessor


@pytest.fixture
def relative_data_dir() -> Path:
    """Provides relative Path to the data directory, asserting it is not absolute."""
    path = Path("data")
    assert not path.is_absolute(), f"Data directory path must be relative, got: {path}"
    return path


@pytest.fixture
def get_relative_data_path():
    """Factory fixture returning relative path to a dataset within data/."""
    def _resolver(filename: str) -> str:
        rel_path = os.path.join("data", filename)
        assert not os.path.isabs(rel_path), f"Path must be relative, got: {rel_path}"
        return rel_path
    return _resolver


@pytest.fixture
def synthetic_loan_data_small() -> pd.DataFrame:
    """
    Provides a small, deterministic synthetic credit portfolio (200 rows)
    with standard credit features and realistic default rates.
    """
    np.random.seed(42)
    n = 200
    fico = np.random.randint(300, 850, n)
    income = np.random.exponential(scale=65000, size=n) + 20000
    loan_amount = np.random.uniform(2000, 35000, n)
    dti = np.random.uniform(5, 45, n)
    emp_length = np.random.choice([0, 1, 3, 5, 10], n)
    purpose = np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase'], n)
    home_ownership = np.random.choice(['RENT', 'MORTGAGE', 'OWN'], n)
    
    # Calculate probability of default with non-linear logistic risk gradient
    z = -1.5 - 0.008 * (fico - 600) + 0.04 * (dti - 20) + 0.00002 * (loan_amount - 10000) - 0.00001 * (income - 50000)
    prob = 1.0 / (1.0 + np.exp(-z))
    default = np.random.binomial(1, prob)
    
    # Ensure both classes exist
    if np.all(default == 0):
        default[0] = 1
    elif np.all(default == 1):
        default[0] = 0
        
    return pd.DataFrame({
        'FICO_score': fico,
        'income': income,
        'loan_amount': loan_amount,
        'dti': dti,
        'employment_length': emp_length,
        'purpose': purpose,
        'home_ownership': home_ownership,
        'default': default
    })


@pytest.fixture
def synthetic_loan_data_with_nans(synthetic_loan_data_small) -> pd.DataFrame:
    """Returns a copy of synthetic loan data with injected NaNs across continuous, categorical, and FICO columns."""
    df = synthetic_loan_data_small.copy()
    rng = np.random.RandomState(123)
    
    # Inject NaNs
    nan_indices = rng.choice(len(df), size=20, replace=False)
    df.loc[nan_indices[:5], 'income'] = np.nan
    df.loc[nan_indices[5:10], 'dti'] = np.nan
    df.loc[nan_indices[10:15], 'purpose'] = np.nan
    df.loc[nan_indices[15:], 'FICO_score'] = np.nan
    
    return df


@pytest.fixture
def binary_categorical_df() -> pd.DataFrame:
    """Returns a minimal dataset with a binary categorical column to test drop='if_binary'."""
    return pd.DataFrame({
        'income': [50000.0, 60000.0, 75000.0, 80000.0, 45000.0, 90000.0],
        'loan_amount': [5000.0, 10000.0, 15000.0, 8000.0, 12000.0, 20000.0],
        'is_employed': ['Y', 'N', 'Y', 'Y', 'N', 'Y'],
        'default': [0, 1, 0, 0, 1, 0]
    })


@pytest.fixture
def high_cardinality_df() -> pd.DataFrame:
    """Returns a DataFrame containing a high-cardinality categorical column (25 unique values)."""
    np.random.seed(99)
    n = 100
    zip_codes = [f"ZIP_{i:03d}" for i in range(25)]
    return pd.DataFrame({
        'income': np.random.uniform(30000, 100000, n),
        'loan_amount': np.random.uniform(1000, 25000, n),
        'zip_code': np.random.choice(zip_codes, n),  # 25 unique >= 20
        'low_card_purpose': np.random.choice(['car', 'wedding', 'medical'], n),  # 3 unique < 20
        'default': np.random.choice([0, 1], n, p=[0.8, 0.2])
    })


@pytest.fixture
def fitted_classification_pipeline(synthetic_loan_data_small) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Provides a fitted Pipeline and holdout test set for metric evaluation tests."""
    df = synthetic_loan_data_small
    train_df = df.iloc[:150]
    test_df = df.iloc[150:]
    
    X_train = train_df.drop(columns=['default'])
    y_train = train_df['default']
    X_test = test_df.drop(columns=['default'])
    y_test = test_df['default']
    
    pipe = Pipeline([
        ('preprocessor', build_preprocessor(
            max_fico_buckets=3,
            fico_col='FICO_score',
            cont_cols=['income', 'loan_amount', 'dti', 'employment_length'],
            cat_cols=['purpose', 'home_ownership']
        )),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


@pytest.fixture
def lending_club_sample_record() -> pd.DataFrame:
    """Sample record formatted according to raw Lending Club dataset schema."""
    return pd.DataFrame([{
        'loan_amnt': 15000.0,
        'emp_length': '5 years',
        'purpose': 'debt_consolidation',
        'loan_status': 0
    }])


@pytest.fixture
def credit_risk_sample_record() -> pd.DataFrame:
    """Sample record formatted according to Generic Credit Risk dataset schema."""
    return pd.DataFrame([{
        'loan_amnt': 10000.0,
        'person_emp_length': 3.5,
        'loan_intent': 'VENTURE',
        'loan_status': 0
    }])


@pytest.fixture
def german_credit_sample_record() -> pd.DataFrame:
    """Sample record formatted according to German Credit dataset schema."""
    return pd.DataFrame([{
        'hoehe': 3500.0,
        'beszeit': 3,
        'verw': '1',
        'kredit': 0
    }])


@pytest.fixture
def harmonized_canonical_record() -> pd.DataFrame:
    """Sample record formatted according to the unified 4-feature canonical schema."""
    return pd.DataFrame([{
        'loan_amount': 12500.0,
        'emp_length': '1-4 years',
        'purpose': 'debt_consolidation'
    }])
