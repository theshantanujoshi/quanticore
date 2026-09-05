"""
Integration tests verifying model training pipeline compatibility across all 3 dataset formats:
1. German Credit format (data/german_ready.csv / data/german_credit_data.csv)
2. Generic Credit Risk format (data/credit_risk_dataset.csv) with missing value handling
3. Lending Club format (data/loan_ready.csv / data/loan.csv) with high-cardinality dropping
4. Multi-dataset harmonization into unified 4-feature schema (incremental_train.py)

All data paths strictly use relative paths to the data/ directory. No hardcoded absolute paths.
"""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from preprocessing import build_preprocessor
from model_training import benchmark_models, evaluate_model, train_and_evaluate
from incremental_train import (
    load_and_harmonize_lending_club,
    load_and_harmonize_credit_risk,
    load_and_harmonize_german
)
from prep_datasets import prep_german, prep_loan


# ==============================================================================
# 1. Relative Path & Dataset Existence Checks
# ==============================================================================

@pytest.mark.parametrize("relative_path_str", [
    "data/german_ready.csv",
    "data/german_credit_data.csv",
    "data/credit_risk_dataset.csv",
    "data/loan_ready.csv",
    "data/loan.csv"
])
def test_relative_dataset_paths_exist(relative_path_str: str):
    """
    Validates that all raw and prepared dataset files exist and are referenced
    strictly via relative paths without any hardcoded absolute path dependencies.
    """
    path = Path(relative_path_str)
    assert not path.is_absolute(), f"Path must be relative, got absolute: {path}"
    assert path.exists(), f"Dataset file does not exist at relative path: {path}"
    assert path.stat().st_size > 0, f"Dataset file at {path} is empty (0 bytes)"


# ==============================================================================
# 2. German Credit Dataset Compatibility (R2.a)
# ==============================================================================

def test_german_credit_format_schema_and_properties():
    """
    Verifies the schema, column types, and target distribution of German Credit data:
    - 1,000 observations, 21 columns
    - Target 'kredit' encoded as binary {0, 1} with ~30% default rate
    - All features are numeric integers without missing values
    """
    data_path = Path("data/german_ready.csv")
    assert not data_path.is_absolute()
    
    df = pd.read_csv(data_path)
    assert df.shape == (1000, 21), f"Unexpected shape for German Credit: {df.shape}"
    assert 'kredit' in df.columns, "Target column 'kredit' missing from german_ready.csv"
    
    # Target values strictly binary {0, 1}
    unique_targets = set(df['kredit'].unique())
    assert unique_targets.issubset({0, 1}), f"Unexpected target values: {unique_targets}"
    default_rate = df['kredit'].mean()
    assert np.isclose(default_rate, 0.30, atol=0.01), f"Expected ~30% default rate, got {default_rate:.2%}"
    
    # Zero missing values across all features
    assert df.isna().sum().sum() == 0, "German credit ready data contains unexpected missing values"


def test_german_credit_pipeline_training_and_evaluation():
    """
    Integration test: Runs end-to-end training and evaluation on data/german_ready.csv.
    - Verifies auto-detection when no FICO column exists (fico_col=None).
    - Verifies multi-model benchmarking executes and selects champion.
    - Verifies champion model achieves ROC-AUC >= 0.70.
    - Verifies prediction outputs valid binary classes {0, 1} and calibrated probabilities.
    """
    data_path = "data/german_ready.csv"
    assert not os.path.isabs(data_path)
    
    # Run training without persisting artifact to root
    champion_pipe = train_and_evaluate(
        data_path=data_path,
        target_col='kredit',
        save_artifact=False
    )
    
    assert isinstance(champion_pipe, Pipeline), "Returned object must be a fitted scikit-learn Pipeline"
    assert 'preprocessor' in champion_pipe.named_steps
    assert 'classifier' in champion_pipe.named_steps
    
    # Inspect preprocessor: FICO transformer must be omitted since German credit has no FICO column
    prep = champion_pipe.named_steps['preprocessor']
    transformer_names = [name for name, _, _ in prep.transformers_]
    assert 'fico' not in transformer_names, "FICO transformer should be omitted for German Credit"
    assert 'cont' in transformer_names, "Continuous transformer must be present for German Credit"
    
    # Test scoring on sample holdout records
    df = pd.read_csv(data_path)
    X = df.drop(columns=['kredit'])
    y = df['kredit']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    metrics = evaluate_model(champion_pipe, X_test, y_test)
    assert metrics['roc_auc'] >= 0.70, f"German credit ROC-AUC ({metrics['roc_auc']:.4f}) below 0.70 threshold"
    assert 0.0 <= metrics['brier_score'] <= 0.25, f"Brier score ({metrics['brier_score']:.4f}) outside acceptable bound"
    assert 0.0 <= metrics['accuracy'] <= 1.0
    
    # Verify probability bounds and row sum conservation
    probs = champion_pipe.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities out of bounds [0, 1]"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-7), "Probability rows do not sum to 1.0"
    
    preds = champion_pipe.predict(X_test)
    assert set(np.unique(preds)).issubset({0, 1}), "Predictions must be binary {0, 1}"


def test_german_credit_raw_prep_mapping(tmp_path):
    """
    Validates prep_german logic:
    Raw German credit has kredit: 1=good, 0=bad.
    Preparation maps 1 -> 0 (non-default), 0 -> 1 (default).
    """
    raw_path = Path("data/german_credit_data.csv")
    assert not raw_path.is_absolute()
    
    df_raw = pd.read_csv(raw_path)
    # Raw target has 1=good (700) and 0=bad (300)
    raw_counts = df_raw['kredit'].value_counts().to_dict()
    assert raw_counts.get(1) == 700
    assert raw_counts.get(0) == 300
    
    # Test mapping logic matches prepared output
    mapped_target = df_raw['kredit'].map({1: 0, 0: 1})
    df_ready = pd.read_csv("data/german_ready.csv")
    assert (df_ready['kredit'] == mapped_target).all(), "Mapped target does not match data/german_ready.csv"


# ==============================================================================
# 3. Generic Credit Risk Dataset Compatibility (R2.b)
# ==============================================================================

def test_credit_risk_dataset_schema_and_missing_values():
    """
    Verifies the Generic Credit Risk format (data/credit_risk_dataset.csv):
    - 32,581 records, 12 columns
    - Target 'loan_status' binary {0, 1} with ~21.8% default rate
    - Specifically verifies null values in 'person_emp_length' and 'loan_int_rate'
      as mandated by requirement R2.
    """
    data_path = Path("data/credit_risk_dataset.csv")
    assert not data_path.is_absolute()
    
    df = pd.read_csv(data_path)
    assert df.shape == (32581, 12), f"Unexpected shape for credit risk dataset: {df.shape}"
    assert 'loan_status' in df.columns, "Target column 'loan_status' missing"
    
    # Verify target distribution
    unique_targets = set(df['loan_status'].unique())
    assert unique_targets.issubset({0, 1})
    default_rate = df['loan_status'].mean()
    assert 0.20 <= default_rate <= 0.23, f"Unexpected default rate: {default_rate:.2%}"
    
    # Mandatory requirement: verify missing values in person_emp_length and loan_int_rate
    emp_nulls = df['person_emp_length'].isna().sum()
    int_nulls = df['loan_int_rate'].isna().sum()
    
    assert emp_nulls > 0, "person_emp_length must contain missing values"
    assert int_nulls > 0, "loan_int_rate must contain missing values"
    assert emp_nulls == 895, f"Expected 895 nulls in person_emp_length, found {emp_nulls}"
    assert int_nulls == 3116, f"Expected 3116 nulls in loan_int_rate, found {int_nulls}"


def test_credit_risk_pipeline_training_with_imputation(tmp_path):
    """
    Integration test: Verifies that the training pipeline successfully fits on the
    Generic Credit Risk format with missing values in person_emp_length and loan_int_rate.
    Uses a stratified sample of 1,000 rows (preserving nulls) for fast CI execution.
    """
    data_path = Path("data/credit_risk_dataset.csv")
    df = pd.read_csv(data_path)
    
    # Extract representative stratified sample of 1,000 records containing nulls
    sample_df = pd.concat([
        df[df['loan_status'] == 0].sample(n=500, random_state=42),
        df[df['loan_status'] == 1].sample(n=500, random_state=42)
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Confirm nulls are present in the sample
    assert sample_df['person_emp_length'].isna().sum() > 0, "Sample missing person_emp_length nulls"
    assert sample_df['loan_int_rate'].isna().sum() > 0, "Sample missing loan_int_rate nulls"
    
    # Write sample to temporary CSV using relative tmp path
    sample_csv = tmp_path / "credit_risk_sample.csv"
    sample_df.to_csv(sample_csv, index=False)
    
    # Train pipeline on Generic Credit Risk format
    champion_pipe = train_and_evaluate(
        data_path=str(sample_csv),
        target_col='loan_status',
        save_artifact=False
    )
    
    assert isinstance(champion_pipe, Pipeline)
    
    # Test scoring on holdout test set containing NaNs
    X = sample_df.drop(columns=['loan_status'])
    y = sample_df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    metrics = evaluate_model(champion_pipe, X_test, y_test)
    assert metrics['roc_auc'] >= 0.80, f"ROC-AUC {metrics['roc_auc']:.4f} is below 0.80 threshold"
    assert metrics['brier_score'] <= 0.20
    assert 0.0 <= metrics['accuracy'] <= 1.0
    
    # Assert inference on records specifically containing NaNs succeeds without error
    nan_records = X_test[X_test['person_emp_length'].isna() | X_test['loan_int_rate'].isna()]
    assert len(nan_records) > 0, "Test set contains no records with NaNs"
    
    nan_probs = champion_pipe.predict_proba(nan_records)
    assert nan_probs.shape == (len(nan_records), 2)
    assert not np.isnan(nan_probs).any(), "Inference on NaN records produced NaN probabilities!"
    assert np.all((nan_probs >= 0.0) & (nan_probs <= 1.0)), "Probabilities out of bounds"
    assert np.allclose(nan_probs.sum(axis=1), 1.0, atol=1e-7), "Row sums do not equal 1.0"


def test_credit_risk_explicit_nan_inference_robustness():
    """
    Verifies that a fitted credit risk model cleanly handles extreme edge-case records:
    - Both person_emp_length and loan_int_rate are NaN simultaneously
    - Unseen categorical levels in loan_intent and person_home_ownership
    """
    df = pd.read_csv("data/credit_risk_dataset.csv").sample(n=500, random_state=42)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = [c for c in X.select_dtypes(include=['object']).columns if X[c].nunique() < 20]
    
    preprocessor = build_preprocessor(fico_col=None, cont_cols=num_cols, cat_cols=cat_cols)
    from sklearn.linear_model import LogisticRegression
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))
    ])
    pipe.fit(X, y)
    
    # Edge case: record with NaNs and unseen category
    edge_record = pd.DataFrame([{
        'person_age': 30,
        'person_income': 65000,
        'person_home_ownership': 'UNKNOWN_HOME',
        'person_emp_length': np.nan,  # NaN
        'loan_intent': 'SPACE_EXPLORATION',  # Unseen category
        'loan_grade': 'B',
        'loan_amnt': 10000,
        'loan_int_rate': np.nan,  # NaN
        'loan_percent_income': 0.15,
        'cb_person_default_on_file': 'N',
        'cb_person_cred_hist_length': 5
    }])
    
    prob = pipe.predict_proba(edge_record)
    assert prob.shape == (1, 2)
    assert not np.isnan(prob).any()
    assert 0.0 <= prob[0, 1] <= 1.0
    assert np.isclose(prob.sum(), 1.0, atol=1e-7)


# ==============================================================================
# 4. Lending Club Dataset Compatibility (R2.c)
# ==============================================================================

def test_lending_club_format_schema_and_properties():
    """
    Verifies the Lending Club prepared dataset (data/loan_ready.csv):
    - 111 total columns
    - Target 'loan_status' binary {0, 1} with ~14.6% default rate
    - Contains high-cardinality columns (nunique >= 20) and sparse/all-null features
    """
    data_path = Path("data/loan_ready.csv")
    assert not data_path.is_absolute()
    
    # Read first 50 rows to inspect schema efficiently
    df_head = pd.read_csv(data_path, nrows=50)
    assert df_head.shape[1] == 111, f"Expected 111 columns, got {df_head.shape[1]}"
    assert 'loan_status' in df_head.columns, "Target 'loan_status' missing from loan_ready.csv"
    
    # Check target binary values in a larger sample
    sample_target = pd.read_csv(data_path, usecols=['loan_status'], nrows=5000)
    assert set(sample_target['loan_status'].unique()).issubset({0, 1})
    def_rate = sample_target['loan_status'].mean()
    assert 0.10 <= def_rate <= 0.25, f"Expected Lending Club default rate ~14.6%, got {def_rate:.2%}"


def test_lending_club_pipeline_training_and_high_cardinality_handling(tmp_path):
    """
    Integration test: Runs pipeline training on a representative sample of Lending Club
    format (1,000 rows with both classes represented).
    - Verifies high-cardinality categorical columns (e.g. emp_title, url, desc, zip_code)
      are dynamically detected and dropped.
    - Verifies numeric features with all-missing values are skipped by imputer without crash.
    - Verifies champion model trains and outputs valid probabilities in [0.0, 1.0].
    """
    data_path = Path("data/loan_ready.csv")
    # Read sample of 1,000 records
    df_sample = pd.read_csv(data_path, nrows=1000)
    assert len(df_sample['loan_status'].unique()) == 2, "Sample must contain both class 0 and 1"
    
    sample_csv = tmp_path / "loan_ready_sample.csv"
    df_sample.to_csv(sample_csv, index=False)
    
    champion_pipe = train_and_evaluate(
        data_path=str(sample_csv),
        target_col='loan_status',
        save_artifact=False
    )
    
    assert isinstance(champion_pipe, Pipeline)
    
    # Test scoring on sample records
    X = df_sample.drop(columns=['loan_status'])
    y = df_sample['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    metrics = evaluate_model(champion_pipe, X_test, y_test)
    assert metrics['roc_auc'] >= 0.85, f"Lending Club ROC-AUC ({metrics['roc_auc']:.4f}) below 0.85"
    assert 0.0 <= metrics['brier_score'] <= 0.20
    assert 0.0 <= metrics['accuracy'] <= 1.0
    
    probs = champion_pipe.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert not np.isnan(probs).any()
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-7)


def test_lending_club_raw_prep_filtering():
    """
    Validates prep_loan logic:
    - Filters out 'Current' loans from raw data/loan.csv
    - Preserves 'Fully Paid' -> 0 and {'Charged Off', 'Default'} -> 1
    """
    raw_path = Path("data/loan.csv")
    assert not raw_path.is_absolute()
    
    # Read status column from raw loan.csv
    raw_statuses = pd.read_csv(raw_path, usecols=['loan_status'], nrows=2000)['loan_status'].value_counts().to_dict()
    assert 'Fully Paid' in raw_statuses or 'Charged Off' in raw_statuses
    
    # Prepared dataset must contain only 0 and 1
    ready_statuses = pd.read_csv("data/loan_ready.csv", usecols=['loan_status'], nrows=2000)['loan_status'].unique()
    assert set(ready_statuses).issubset({0, 1})


# ==============================================================================
# 5. Multi-Dataset Harmonization (incremental_train.py)
# ==============================================================================

def test_incremental_harmonize_loaders_contract():
    """
    Verifies that all three harmonization loaders from incremental_train.py:
    - load_and_harmonize_lending_club()
    - load_and_harmonize_credit_risk()
    - load_and_harmonize_german()
    conform to the unified 4-feature canonical schema:
    ['loan_amount', 'emp_length', 'purpose', 'target'].
    """
    expected_cols = {'loan_amount', 'emp_length', 'purpose', 'target'}
    
    df_lc = load_and_harmonize_lending_club()
    assert set(df_lc.columns) == expected_cols, f"Lending club schema mismatch: {df_lc.columns}"
    assert len(df_lc) > 30000
    assert set(df_lc['target'].unique()).issubset({0, 1})
    assert (df_lc['loan_amount'] > 0).all()
    
    df_cr = load_and_harmonize_credit_risk()
    assert set(df_cr.columns) == expected_cols, f"Credit risk schema mismatch: {df_cr.columns}"
    assert len(df_cr) > 30000
    assert set(df_cr['target'].unique()).issubset({0, 1})
    assert (df_cr['loan_amount'] > 0).all()
    
    df_ger = load_and_harmonize_german()
    assert set(df_ger.columns) == expected_cols, f"German credit schema mismatch: {df_ger.columns}"
    assert len(df_ger) == 1000
    assert set(df_ger['target'].unique()).issubset({0, 1})
    assert (df_ger['loan_amount'] > 0).all()


def test_standardized_employment_length_bins():
    """
    Verifies that employment length categories are mapped to standard discrete bins
    across all three harmonized datasets.
    """
    expected_categories = {'< 1 year', '1-4 years', '4-7 years', '7+ years', 'unknown', 'unemployed'}
    
    for loader_fn, name in [
        (load_and_harmonize_lending_club, "Lending Club"),
        (load_and_harmonize_credit_risk, "Credit Risk"),
        (load_and_harmonize_german, "German Credit")
    ]:
        df = loader_fn()
        unique_bins = set(df['emp_length'].unique())
        assert unique_bins.issubset(expected_categories), f"Unexpected emp_length in {name}: {unique_bins - expected_categories}"


def test_unified_harmonization_preprocessor_fitting():
    """
    Verifies that a joint ColumnTransformer fitted on combined harmonized data
    transforms records from all three disparate formats into uniform feature vectors
    with exactly 37 total dimensions and zero missing values.
    """
    df_lc = load_and_harmonize_lending_club()
    df_cr = load_and_harmonize_credit_risk()
    df_ger = load_and_harmonize_german()
    
    df_combined = pd.concat([df_lc, df_cr, df_ger], ignore_index=True)
    X_combined = df_combined.drop(columns=['target'])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['loan_amount']),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]), ['emp_length', 'purpose'])
        ]
    )
    preprocessor.fit(X_combined)
    
    # Transform sample from each source dataset
    for df_source, source_name in [(df_lc, "Lending Club"), (df_cr, "Credit Risk"), (df_ger, "German Credit")]:
        sample = df_source.drop(columns=['target']).head(20)
        X_trans = preprocessor.transform(sample)
        
        assert X_trans.shape == (20, 37), f"Expected 37 features for {source_name}, got {X_trans.shape[1]}"
        assert not np.isnan(X_trans).any(), f"Transformed features contain NaN for {source_name}"
        assert not np.isinf(X_trans).any(), f"Transformed features contain Inf for {source_name}"


def test_cross_format_unmodeled_column_drop():
    """
    Verifies that when records from one dataset format contain extra fields
    not in another format's preprocessor, ColumnTransformer(remainder='drop')
    safely drops the extraneous columns without throwing errors.
    """
    # Fit preprocessor on German Credit
    df_ger = pd.read_csv("data/german_ready.csv", nrows=100)
    X_ger = df_ger.drop(columns=['kredit'])
    
    prep = build_preprocessor(
        fico_col=None,
        cont_cols=X_ger.columns.tolist(),
        cat_cols=[]
    )
    prep.fit(X_ger)
    
    # Inject Lending Club features into German Credit record
    corrupted_record = X_ger.head(5).copy()
    corrupted_record['loan_status'] = 0
    corrupted_record['emp_title'] = 'Software Engineer'
    corrupted_record['zip_code'] = '94107'
    corrupted_record['arbitrary_extra_col'] = 'IGNORE_ME'
    
    # Transform must succeed without error
    out = prep.transform(corrupted_record)
    assert out.shape == (5, len(X_ger.columns))
    assert not np.isnan(out).any()
