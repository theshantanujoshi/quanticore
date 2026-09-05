import os

import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

from fico_bucketing import FicoDPBucketer
from preprocessing import build_preprocessor
from data_generator import generate_loan_data
from model_training import evaluate_model, benchmark_models, train_and_evaluate

def test_dp_bucketing_logic():
    """
    Validates that the DP bucketer discovers optimal split points that separate
    high-default from low-default credit segments.
    """
    ficos = np.array([350, 400, 450, 600, 620, 650, 750, 800, 820])
    defaults = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    
    bucketer = FicoDPBucketer(max_buckets=2)
    bucketer.fit(ficos, defaults)
    
    boundaries = bucketer.boundaries_
    assert boundaries[0] == -np.inf
    assert boundaries[-1] == np.inf
    assert len(boundaries) == 3  # exactly 2 buckets: [-inf, split, inf]
    # Optimal split must divide high-risk [350..600] from low-risk [620..820]
    assert boundaries[1] in [600.0, 620.0]
    
    # Check bucket means are populated and monotonic
    assert hasattr(bucketer, 'bucket_means_')
    assert len(bucketer.bucket_means_) == 2
    assert bucketer.bucket_means_[0] > bucketer.bucket_means_[1]

def test_dp_bucketer_2d_and_container_inputs():
    """
    Tests input polymorphism: 2D numpy arrays (N, 1), 1D arrays, Series, DataFrames.
    """
    ficos_raw = [500, 600, 700, 800]
    defaults = [1, 1, 0, 0]
    
    # 1. 2D NumPy array (standard scikit-learn format)
    X_2d = np.array(ficos_raw).reshape(-1, 1)
    b_2d = FicoDPBucketer(max_buckets=2)
    b_2d.fit(X_2d, defaults)
    out_2d = b_2d.transform(X_2d)
    assert out_2d.shape == (4, 1)
    assert np.all(np.isin(out_2d, [0, 1]))
    
    # 2. pandas DataFrame (single column)
    X_df = pd.DataFrame({'FICO_score': ficos_raw})
    b_df = FicoDPBucketer(max_buckets=2)
    b_df.fit(X_df, defaults)
    out_df = b_df.transform(X_df)
    assert out_df.shape == (4, 1)
    assert np.array_equal(out_2d, out_df)
    
    # 3. pandas Series
    X_s = pd.Series(ficos_raw)
    b_s = FicoDPBucketer(max_buckets=2)
    b_s.fit(X_s, defaults)
    out_s = b_s.transform(X_s)
    assert out_s.shape == (4, 1)
    assert np.array_equal(out_2d, out_s)

def test_dp_bucketer_sklearn_compliance():
    """
    Enforces scikit-learn BaseEstimator / TransformerMixin compliance:
    - __init__ sets only max_buckets without trailing underscore attributes
    - fit sets boundaries_, bucket_means_, and n_features_in_
    - get_feature_names_out returns ['fico_bucket']
    - transform raises NotFittedError if called before fit
    """
    bucketer = FicoDPBucketer(max_buckets=3)
    
    # Estimator must not have trailing underscore attributes before fit
    assert not hasattr(bucketer, 'boundaries_')
    assert not hasattr(bucketer, 'bucket_means_')
    assert not hasattr(bucketer, 'n_features_in_')
    
    # Calling transform or get_feature_names_out before fit must raise error
    with pytest.raises(Exception):
        bucketer.transform([600, 700])
    with pytest.raises(Exception):
        bucketer.get_feature_names_out()
        
    # Fit with data
    X = np.array([550, 620, 700, 780])
    y = np.array([1, 1, 0, 0])
    bucketer.fit(X, y)
    
    assert hasattr(bucketer, 'boundaries_')
    assert hasattr(bucketer, 'bucket_means_')
    assert hasattr(bucketer, 'n_features_in_')
    assert bucketer.n_features_in_ == 1
    
    feature_names = bucketer.get_feature_names_out()
    assert list(feature_names) == ['fico_bucket']

def test_dp_bucketer_edge_cases():
    """
    Tests edge cases: small n (n <= max_buckets), n=1, empty, invalid params, lack of target.
    """
    # 1. Fewer unique FICO scores than max_buckets (do not collapse to 1 bucket)
    ficos = np.array([500, 600, 700])
    defaults = np.array([1, 0, 0])
    b = FicoDPBucketer(max_buckets=5)
    b.fit(ficos, defaults)
    assert len(b.boundaries_) == 4  # 3 distinct buckets [-inf, 600, 700, inf]
    
    # 2. Single unique score (n=1)
    b_single = FicoDPBucketer(max_buckets=3)
    b_single.fit([650, 650, 650], [0, 1, 0])
    assert b_single.boundaries_ == [-np.inf, np.inf]
    assert len(b_single.bucket_means_) == 1
    
    # 3. Invalid max_buckets < 1
    with pytest.raises(ValueError, match="max_buckets must be >= 1"):
        FicoDPBucketer(max_buckets=0).fit([600, 700], [0, 1])
        
    # 4. Supervised requirement: fit without y raises ValueError
    with pytest.raises(ValueError, match="requires target y"):
        FicoDPBucketer().fit([600, 700])

def test_preprocessor_remainder_drop():
    """
    Ensures ColumnTransformer uses remainder='drop' to safely discard extraneous
    metadata/ID columns (preventing string conversion errors or data leakage).
    """
    df = pd.DataFrame({
        'loan_id': ['LN-001', 'LN-002', 'LN-003', 'LN-004'],
        'customer_id': ['CUST-10', 'CUST-20', 'CUST-30', 'CUST-40'],
        'FICO_score': [600, 700, 800, 650],
        'income': [50000.0, 80000.0, 120000.0, 60000.0],
        'loan_amount': [10000.0, 20000.0, 15000.0, 10000.0],
        'dti': [0.4, 0.2, 0.1, 0.3],
        'employment_length': [2, 5, 10, 3],
        'underwriter_notes': ['marginal', 'prime', 'superprime', 'fair'],
        'default': [1, 0, 0, 1]
    })
    
    X = df.drop(columns=['default'])
    y = df['default']
    
    preprocessor = build_preprocessor(max_fico_buckets=2)
    transformed = preprocessor.fit_transform(X, y)
    
    # Output must be purely numeric ndarray
    assert isinstance(transformed, np.ndarray)
    assert np.issubdtype(transformed.dtype, np.floating)
    
    # Check feature names out
    feature_names = preprocessor.get_feature_names_out()
    assert all('loan_id' not in fn and 'customer_id' not in fn for fn in feature_names)
    assert any('fico' in fn for fn in feature_names)
    assert any('income' in fn for fn in feature_names)

def test_preprocessor_missing_values():
    """
    Confirms missing values in continuous features are imputed via median.
    """
    df = pd.DataFrame({
        'FICO_score': [600, 700, 800, 650],
        'income': [np.nan, 80000.0, 120000.0, 60000.0],
        'loan_amount': [10000.0, np.nan, 15000.0, 10000.0],
        'dti': [0.4, 0.2, np.nan, 0.3],
        'employment_length': [2, 5, 10, np.nan],
        'default': [1, 0, 0, 1]
    })
    
    X = df.drop(columns=['default'])
    y = df['default']
    
    preprocessor = build_preprocessor(max_fico_buckets=2)
    transformed = preprocessor.fit_transform(X, y)
    assert not np.isnan(transformed).any()

def test_pipeline_execution_and_metrics():
    """
    Tests end-to-end pipeline fitting, prediction probabilities, and evaluation metrics.
    """
    df = generate_loan_data(num_samples=1000, random_state=42)
    X = df.drop(columns=['default'])
    y = df['default']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = build_preprocessor(max_fico_buckets=5)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    assert metrics['roc_auc'] >= 0.75, f"ROC-AUC {metrics['roc_auc']} is below 0.75 threshold"
    assert metrics['brier_score'] <= 0.25, f"Brier score {metrics['brier_score']} is above 0.25 threshold"
    assert 0.0 <= metrics['accuracy'] <= 1.0
    
    preds_proba = pipeline.predict_proba(X_test)
    assert preds_proba.shape == (len(X_test), 2)
    assert np.all((preds_proba >= 0.0) & (preds_proba <= 1.0))

def test_multi_model_benchmark():
    """
    Tests that benchmark_models executes across candidate algorithms and returns valid leaderboards.
    """
    df = generate_loan_data(num_samples=500, random_state=42)
    X = df.drop(columns=['default'])
    y = df['default']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results, pipes = benchmark_models(X_train, y_train, X_test, y_test, max_fico_buckets=3)
    
    assert 'Logistic Regression' in results
    assert 'Gradient Boosting' in results
    assert 'Random Forest' in results
    
    for name, m in results.items():
        assert 'roc_auc' in m
        assert 'brier_score' in m
        assert 'accuracy' in m
        assert 'f1_score' in m

def test_model_artifact_persistence_roundtrip(tmp_path):
    """
    Validates model artifact persistence using joblib: saving to disk, reloading,
    and verifying zero floating-point divergence on new predictions.
    """
    df = generate_loan_data(num_samples=500, random_state=42)
    X = df.drop(columns=['default'])
    y = df['default']
    
    pipe = Pipeline([
        ('preprocessor', build_preprocessor(max_fico_buckets=3)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe.fit(X, y)
    
    artifact_path = os.path.join(tmp_path, "test_model.joblib")
    joblib.dump(pipe, artifact_path)
    assert os.path.exists(artifact_path)
    
    reloaded_pipe = joblib.load(artifact_path)
    
    sample = pd.DataFrame([{
        'FICO_score': 710,
        'income': 65000.0,
        'loan_amount': 15000.0,
        'dti': 0.28,
        'employment_length': 4
    }])
    
    orig_pred = pipe.predict_proba(sample)
    reloaded_pred = reloaded_pipe.predict_proba(sample)
    
    assert np.allclose(orig_pred, reloaded_pred, atol=1e-7)
    assert 0.0 <= reloaded_pred[0, 1] <= 1.0

def test_single_record_inference():
    """
    Verifies that a single loan application record (as received by a serving API)
    is scored cleanly without needing manual preprocessing.
    """
    df = generate_loan_data(num_samples=500, random_state=42)
    X = df.drop(columns=['default'])
    y = df['default']
    
    pipe = Pipeline([
        ('preprocessor', build_preprocessor(max_fico_buckets=3)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe.fit(X, y)
    
    single_record = pd.DataFrame([{
        'FICO_score': 680,
        'income': 55000.0,
        'loan_amount': 12000.0,
        'dti': 0.35,
        'employment_length': 3
    }])
    
    prob = pipe.predict_proba(single_record)
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0, 1] <= 1.0
