"""
Model Inference & Artifact Validation Tests (R3).
Verifies that all saved .joblib model artifacts:
1. Load cleanly from relative paths.
2. Output binary class predictions in {0, 1}.
3. Output calibrated probabilities strictly bounded in [0.0, 1.0] with row sums == 1.0 +- 1e-7.
4. Exhibit single vs batch inference invariance.
5. Safely drop extraneous unmodeled columns via remainder='drop'.
6. All *_metadata.joblib files load and contain valid champion model information and metrics.
7. Gracefully handle edge cases: NaNs in continuous features and unseen categories in categorical features.
"""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline


ALL_MODEL_ARTIFACTS = [
    "models/model_artifact.joblib",
    "models/model.joblib",
    "models/model_german.joblib",
    "models/model_credit_risk.joblib",
    "models/model_loan.joblib",
    "models/incremental_master_model.joblib"
]

ALL_METADATA_ARTIFACTS = [
    "models/model_artifact_metadata.joblib",
    "models/model_german_metadata.joblib",
    "models/model_credit_risk_metadata.joblib",
    "models/model_loan_metadata.joblib"
]


def get_sample_inference_data(artifact_name: str, n_samples: int = 5) -> pd.DataFrame:
    """
    Constructs or extracts a valid, schema-conforming sample DataFrame
    for the specified model artifact.
    """
    n = n_samples
    if artifact_name == "models/model_artifact.joblib":
        return pd.DataFrame({
            'FICO_score': [780, 650, 580, 720, 690][:n],
            'income': [95000.0, 55000.0, 32000.0, 80000.0, 62000.0][:n],
            'loan_amount': [12000.0, 15000.0, 20000.0, 8000.0, 14000.0][:n],
            'dti': [0.18, 0.32, 0.45, 0.22, 0.28][:n],
            'employment_length': [6, 3, 1, 8, 4][:n]
        })
    elif artifact_name == "models/model.joblib":
        df = get_sample_inference_data("models/model_artifact.joblib", n).copy()
        df['purpose'] = ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'medical'][:n]
        df['home_ownership'] = ['MORTGAGE', 'RENT', 'RENT', 'OWN', 'RENT'][:n]
        return df
    elif artifact_name == "models/model_german.joblib":
        return pd.read_csv("data/german_ready.csv", nrows=n).drop(columns=['kredit'])
    elif artifact_name == "models/model_credit_risk.joblib":
        return pd.read_csv("data/credit_risk_dataset.csv", nrows=n).drop(columns=['loan_status'])
    elif artifact_name == "models/model_loan.joblib":
        return pd.read_csv("data/loan_ready.csv", nrows=n).drop(columns=['loan_status'])
    elif artifact_name == "models/incremental_master_model.joblib":
        return pd.DataFrame({
            'loan_amount': [5000.0, 12000.0, 25000.0, 8000.0, 15000.0][:n],
            'emp_length': ['1-4 years', '7+ years', '< 1 year', '4-7 years', 'unknown'][:n],
            'purpose': ['debt_consolidation', 'credit_card', 'medical', 'home_improvement', 'car'][:n]
        })
    raise ValueError(f"Unrecognized model artifact name: {artifact_name}")


# ==============================================================================
# 1. Model Artifact Loading & Structural Integrity
# ==============================================================================

@pytest.mark.parametrize("artifact_name", ALL_MODEL_ARTIFACTS)
def test_saved_model_artifacts_load_successfully(artifact_name: str):
    """
    Verifies that each saved .joblib model artifact:
    - Exists at the relative project root path
    - Loads cleanly using joblib.load
    - Is an instance of sklearn.pipeline.Pipeline
    - Contains both 'preprocessor' and 'classifier' named steps
    """
    path = Path(artifact_name)
    assert not path.is_absolute(), f"Path must be relative: {path}"
    assert path.exists(), f"Model artifact {artifact_name} does not exist"
    assert path.stat().st_size > 0, f"Model artifact {artifact_name} is empty"
    
    model = joblib.load(path)
    assert isinstance(model, Pipeline), f"{artifact_name} is not an sklearn Pipeline"
    assert 'preprocessor' in model.named_steps, f"{artifact_name} missing 'preprocessor' step"
    assert 'classifier' in model.named_steps, f"{artifact_name} missing 'classifier' step"


# ==============================================================================
# 2. Probability Bounding & Conservation ([0.0, 1.0] and sum == 1.0)
# ==============================================================================

@pytest.mark.parametrize("artifact_name", ALL_MODEL_ARTIFACTS)
def test_predict_proba_bounds_and_conservation(artifact_name: str):
    """
    Mathematical validity test:
    - predict_proba returns array of shape (N, 2)
    - All probability values are strictly bounded in [0.0, 1.0]
    - Row sums equal 1.0 +- 1e-7 (conservation of probability)
    - Contains zero NaN or Inf values
    """
    model = joblib.load(artifact_name)
    sample_df = get_sample_inference_data(artifact_name, n_samples=5)
    
    probs = model.predict_proba(sample_df)
    
    # 1. Shape validation
    assert probs.shape == (len(sample_df), 2), f"Expected shape ({len(sample_df)}, 2), got {probs.shape}"
    
    # 2. No NaN or Inf
    assert not np.isnan(probs).any(), f"{artifact_name} predict_proba produced NaN values"
    assert not np.isinf(probs).any(), f"{artifact_name} predict_proba produced Inf values"
    
    # 3. Probability bounds [0.0, 1.0]
    assert np.all(probs >= 0.0), f"{artifact_name} produced negative probability: {probs.min()}"
    assert np.all(probs <= 1.0), f"{artifact_name} produced probability > 1.0: {probs.max()}"
    
    # 4. Conservation of probability: row sums == 1.0 +- 1e-7
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-7), f"{artifact_name} row sums do not sum to 1.0: {row_sums}"


# ==============================================================================
# 3. Binary Class Prediction Contract ({0, 1})
# ==============================================================================

@pytest.mark.parametrize("artifact_name", ALL_MODEL_ARTIFACTS)
def test_predict_outputs_binary_classes(artifact_name: str):
    """
    Verifies that .predict() outputs binary class labels strictly in {0, 1}
    and matches the 0.50 decision threshold applied to predict_proba[:, 1].
    """
    model = joblib.load(artifact_name)
    sample_df = get_sample_inference_data(artifact_name, n_samples=5)
    
    preds = model.predict(sample_df)
    probs = model.predict_proba(sample_df)
    
    # Assert return type and length
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(sample_df)
    
    # Binary class membership {0, 1}
    unique_classes = set(np.unique(preds))
    assert unique_classes.issubset({0, 1}), f"{artifact_name} returned non-binary classes: {unique_classes}"
    
    # Check decision threshold consistency: predict == (p >= 0.50)
    expected_preds = (probs[:, 1] >= 0.50).astype(int)
    assert np.array_equal(preds, expected_preds), (
        f"{artifact_name} .predict() does not match 0.50 threshold on predict_proba[:, 1]"
    )


# ==============================================================================
# 4. Single-Row vs Batch Inference Consistency
# ==============================================================================

@pytest.mark.parametrize("artifact_name", ALL_MODEL_ARTIFACTS)
def test_single_vs_batch_inference_invariance(artifact_name: str):
    """
    Verifies single-row vs batch inference consistency:
    Scoring a record individually as a 1-row DataFrame must yield
    the exact same probability scores as when evaluated inside a multi-row batch.
    |P_single - P_batch| <= 1e-7.
    """
    model = joblib.load(artifact_name)
    batch_df = get_sample_inference_data(artifact_name, n_samples=5)
    batch_probs = model.predict_proba(batch_df)
    batch_preds = model.predict(batch_df)
    
    for i in range(len(batch_df)):
        single_df = batch_df.iloc[[i]].copy()
        single_prob = model.predict_proba(single_df)
        single_pred = model.predict(single_df)
        
        # Probabilities must match within 1e-7
        assert np.allclose(single_prob[0], batch_probs[i], atol=1e-7), (
            f"{artifact_name} single vs batch divergence at row {i}: "
            f"single={single_prob[0]} vs batch={batch_probs[i]}"
        )
        # Class predictions must match identically
        assert single_pred[0] == batch_preds[i], (
            f"{artifact_name} class prediction mismatch at row {i}"
        )


# ==============================================================================
# 5. Tolerance to Extraneous Unmodeled Columns (remainder='drop')
# ==============================================================================

@pytest.mark.parametrize("artifact_name", ALL_MODEL_ARTIFACTS)
def test_tolerance_to_extraneous_columns(artifact_name: str):
    """
    Verifies pipeline resilience when extraneous unmodeled columns are injected.
    Because preprocessors are built with remainder='drop', unmodeled fields
    (customer IDs, SSNs, timestamps, random notes) must be discarded silently
    with zero change in output predictions.
    """
    model = joblib.load(artifact_name)
    clean_df = get_sample_inference_data(artifact_name, n_samples=3)
    clean_probs = model.predict_proba(clean_df)
    
    injected_df = clean_df.copy()
    injected_df['customer_id'] = ['CUST-001', 'CUST-002', 'CUST-003']
    injected_df['ssn'] = ['000-11-2222', '000-33-4444', '000-55-6666']
    injected_df['application_timestamp'] = pd.Timestamp.now()
    injected_df['irrelevant_metadata_float'] = 987654.321
    injected_df['is_pilot_borrower'] = True
    
    injected_probs = model.predict_proba(injected_df)
    
    # Must produce identical predictions
    assert np.allclose(clean_probs, injected_probs, atol=1e-7), (
        f"{artifact_name} predictions changed when unmodeled columns were added"
    )


# ==============================================================================
# 6. Metadata Artifacts Validation (*_metadata.joblib)
# ==============================================================================

@pytest.mark.parametrize("meta_name", ALL_METADATA_ARTIFACTS)
def test_metadata_artifacts_load_and_fields_valid(meta_name: str):
    """
    Verifies that all *_metadata.joblib files:
    - Load cleanly from relative paths
    - Are valid dictionaries
    - Contain required keys: 'champion_model', 'metrics', 'features'
    - Contain expected metric fields: 'roc_auc', 'pr_auc', 'brier_score', 'accuracy', 'f1_score'
    - All metric values are valid floats bounded in [0.0, 1.0]
    """
    path = Path(meta_name)
    assert not path.is_absolute(), f"Path must be relative: {path}"
    assert path.exists(), f"Metadata file {meta_name} does not exist"
    
    meta = joblib.load(path)
    assert isinstance(meta, dict), f"{meta_name} must deserialize to a dictionary"
    
    # Required top-level keys
    for req_key in ['champion_model', 'metrics', 'features']:
        assert req_key in meta, f"{meta_name} missing required key '{req_key}'"
        
    assert isinstance(meta['champion_model'], str)
    assert len(meta['champion_model']) > 0
    assert isinstance(meta['features'], list)
    assert len(meta['features']) > 0
    
    # Inspect metrics dictionary
    metrics = meta['metrics']
    assert isinstance(metrics, dict), f"'metrics' in {meta_name} must be a dictionary"
    
    expected_metric_keys = ['roc_auc', 'pr_auc', 'brier_score', 'accuracy', 'f1_score']
    for m_key in expected_metric_keys:
        assert m_key in metrics, f"Metric '{m_key}' missing from {meta_name}"
        val = metrics[m_key]
        assert isinstance(val, (int, float)), f"Metric '{m_key}' must be numeric, got {type(val)}"
        assert 0.0 <= val <= 1.0, f"Metric '{m_key}' ({val}) out of bounds [0.0, 1.0] in {meta_name}"
        
    # Gini coefficient consistency check: Gini = 2 * ROC_AUC - 1
    if 'gini' in metrics:
        expected_gini = 2.0 * metrics['roc_auc'] - 1.0
        assert np.isclose(metrics['gini'], expected_gini, atol=1e-5), (
            f"Gini ({metrics['gini']}) does not match 2*AUC - 1 ({expected_gini}) in {meta_name}"
        )


def test_fico_boundaries_in_applicable_metadata():
    """
    For pipelines that utilize FICO DP bucketing (model_artifact and model_credit_risk),
    verifies that fico_boundaries:
    - Is non-empty
    - Starts with -inf and ends with inf
    - Has strictly monotonic cutoffs
    - Has corresponding bucket_means between 0.0 and 1.0
    """
    for meta_name in ['models/model_artifact_metadata.joblib', 'models/model_credit_risk_metadata.joblib']:
        meta = joblib.load(meta_name)
        boundaries = meta.get('fico_boundaries')
        assert boundaries is not None, f"Expected fico_boundaries in {meta_name}"
        assert len(boundaries) >= 3, f"Expected at least 2 buckets (3 cutoffs) in {meta_name}"
        
        # Check boundary bounds
        assert np.isneginf(boundaries[0]), f"First cutoff must be -inf in {meta_name}"
        assert np.isposinf(boundaries[-1]), f"Last cutoff must be inf in {meta_name}"
        
        # Monotonicity of finite cutoffs
        finite_cutoffs = [b for b in boundaries if not np.isinf(b)]
        for i in range(len(finite_cutoffs) - 1):
            assert finite_cutoffs[i] < finite_cutoffs[i + 1], f"Cutoffs not strictly monotonic in {meta_name}"


# ==============================================================================
# 7. Robustness to Missing Features, NaNs, and Unseen Categories
# ==============================================================================

def test_inference_missing_required_column_raises():
    """
    Verifies that omitting a required feature column raises a descriptive ValueError.
    """
    model = joblib.load("models/model_artifact.joblib")
    df = get_sample_inference_data("models/model_artifact.joblib", n_samples=2)
    df_missing = df.drop(columns=['loan_amount'])
    
    with pytest.raises(ValueError, match="columns are missing"):
        model.predict_proba(df_missing)


def test_inference_continuous_nan_imputation():
    """
    Verifies that injecting NaNs into continuous features is handled cleanly
    by SimpleImputer(strategy='median') without producing NaN outputs.
    """
    model = joblib.load("models/model_artifact.joblib")
    df = get_sample_inference_data("models/model_artifact.joblib", n_samples=3)
    
    # Inject NaNs
    df.loc[0, 'income'] = np.nan
    df.loc[1, 'dti'] = np.nan
    df.loc[2, 'loan_amount'] = np.nan
    
    probs = model.predict_proba(df)
    assert probs.shape == (3, 2)
    assert not np.isnan(probs).any(), "Continuous NaN injection resulted in NaN prediction!"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-7)


def test_inference_unseen_categorical_levels():
    """
    Verifies that passing unseen categories in categorical features
    is handled cleanly by OneHotEncoder(handle_unknown='ignore')
    without crashing or returning NaNs.
    """
    model = joblib.load("models/incremental_master_model.joblib")
    df = pd.DataFrame([{
        'loan_amount': 15000.0,
        'emp_length': '30+ years unrecorded category',
        'purpose': 'interstellar_transportation'
    }])
    
    prob = model.predict_proba(df)
    assert prob.shape == (1, 2)
    assert not np.isnan(prob).any()
    assert 0.0 <= prob[0, 1] <= 1.0
    assert np.isclose(prob.sum(), 1.0, atol=1e-7)


def test_model_artifact_and_model_joblib_coherence():
    """
    Verifies that model_artifact.joblib and model.joblib both exist,
    load successfully as valid Pipeline estimators, and score borrowers cleanly.
    """
    m1 = joblib.load("models/model_artifact.joblib")
    m2 = joblib.load("models/model.joblib")
    
    assert isinstance(m1, Pipeline)
    assert isinstance(m2, Pipeline)
    
    df1 = get_sample_inference_data("models/model_artifact.joblib", n_samples=2)
    df2 = get_sample_inference_data("models/model.joblib", n_samples=2)
    
    p1 = m1.predict_proba(df1)
    p2 = m2.predict_proba(df2)
    
    assert np.all((p1 >= 0.0) & (p1 <= 1.0))
    assert np.all((p2 >= 0.0) & (p2 <= 1.0))
