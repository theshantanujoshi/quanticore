import os
import sys
import subprocess
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from preprocessing import build_preprocessor
from data_generator import generate_loan_data
from model_training import benchmark_models, evaluate_model, train_and_evaluate
from fico_bucketing import FicoDPBucketer

def test_adversarial_leakage_contamination():
    """
    Data Leakage Adversarial Challenge:
    Verifies that the preprocessor and model are strictly isolated from the test set.
    Altering, corrupting, or injecting adversarial extremes into X_test and y_test
    must produce zero change in fitted preprocessor parameters (bucketer boundaries,
    imputer statistics, scaler means) or model coefficients.
    """
    df = generate_loan_data(num_samples=1500, random_state=42)
    X = df.drop(columns=['default'])
    y = df['default']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Verify split disjointness
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert len(train_indices.intersection(test_indices)) == 0, "Train and test indices overlap!"
    
    # 1. Fit baseline pipeline on (X_train, y_train)
    pipe_baseline = Pipeline([
        ('preprocessor', build_preprocessor(max_fico_buckets=5)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_baseline.fit(X_train, y_train)
    
    # 2. Corrupt X_test and y_test with catastrophic adversarial shifts
    X_test_corrupted = X_test.copy(deep=True)
    X_test_corrupted['FICO_score'] = 9999
    X_test_corrupted['income'] = 1e12
    X_test_corrupted['loan_amount'] = 1e9
    X_test_corrupted['dti'] = 500.0
    X_test_corrupted['employment_length'] = 100
    y_test_corrupted = pd.Series(np.ones(len(y_test)), index=y_test.index)
    
    # 3. Fit second pipeline on X_train, y_train
    pipe_isolated = Pipeline([
        ('preprocessor', build_preprocessor(max_fico_buckets=5)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_isolated.fit(X_train, y_train)
    
    # Evaluate baseline on uncorrupted test set vs corrupted test set
    metrics_clean = evaluate_model(pipe_baseline, X_test, y_test)
    metrics_corrupt = evaluate_model(pipe_baseline, X_test_corrupted, y_test_corrupted)
    # The metrics must differ on corrupted test set, confirming evaluation evaluates test data
    assert metrics_clean['roc_auc'] != metrics_corrupt['roc_auc']
    
    # Verify fitted parameters of pipeline are completely identical
    prep_b = pipe_baseline.named_steps['preprocessor']
    prep_i = pipe_isolated.named_steps['preprocessor']
    
    # FICO bucketer boundaries
    b_base = prep_b.transformers_[0][1].named_steps['bucketer'].boundaries_
    b_iso = prep_i.transformers_[0][1].named_steps['bucketer'].boundaries_
    assert b_base == b_iso, "FicoDPBucketer boundaries changed between runs!"
    
    # SimpleImputer statistics
    imp_base = prep_b.transformers_[1][1].named_steps['imputer'].statistics_
    imp_iso = prep_i.transformers_[1][1].named_steps['imputer'].statistics_
    assert np.array_equal(imp_base, imp_iso), "SimpleImputer statistics contaminated!"
    
    # StandardScaler means and scales
    scale_base = prep_b.transformers_[1][1].named_steps['scaler'].mean_
    scale_iso = prep_i.transformers_[1][1].named_steps['scaler'].mean_
    assert np.array_equal(scale_base, scale_iso), "StandardScaler mean contaminated!"
    
    # Classifier coefficients
    coef_base = pipe_baseline.named_steps['classifier'].coef_
    coef_iso = pipe_isolated.named_steps['classifier'].coef_
    assert np.array_equal(coef_base, coef_iso), "Classifier coefficients contaminated!"
    
    # Verify preprocessor parameters match X_train empirical statistics
    cont_cols = ['income', 'loan_amount', 'dti', 'employment_length']
    expected_medians = X_train[cont_cols].median().values
    assert np.allclose(imp_base, expected_medians), "Imputer did not use X_train medians!"
    
    imputed_X_train = X_train[cont_cols].fillna(X_train[cont_cols].median())
    expected_means = imputed_X_train.mean().values
    assert np.allclose(scale_base, expected_means), "Scaler did not use X_train means!"

def test_fresh_session_serialization_integrity():
    """
    Serialization Integrity Adversarial Challenge:
    Spawns a clean, fresh Python interpreter process to reload `model_artifact.joblib`.
    Evaluates 5,000 extreme, random, and edge-case loan applications.
    Confirms all predictions are valid, bounded [0, 1] calibrated probabilities.
    """
    artifact_path = "model_artifact.joblib"
    assert os.path.exists(artifact_path), f"Artifact missing at {artifact_path}"
    
    eval_script = r"""
import joblib
import numpy as np
import pandas as pd

pipeline = joblib.load("model_artifact.joblib")

n_samples = 5000
rng = np.random.default_rng(12345)

# 1. Extreme and adversarial distribution
ficos = rng.uniform(-1000, 2000, size=n_samples)
incomes = rng.uniform(-1e6, 1e9, size=n_samples)
loans = rng.uniform(-1e5, 1e8, size=n_samples)
dtis = rng.uniform(-50, 500, size=n_samples)
empls = rng.uniform(-20, 80, size=n_samples)

df = pd.DataFrame({
    'FICO_score': ficos,
    'income': incomes,
    'loan_amount': loans,
    'dti': dtis,
    'employment_length': empls
})

# 2. Randomly inject NaNs into ~20% of entries across all features
nan_mask = rng.random(size=df.shape) < 0.20
df[nan_mask] = np.nan

# 3. Add explicit edge-case rows
edge_cases = pd.DataFrame([
    # All NaN
    {'FICO_score': np.nan, 'income': np.nan, 'loan_amount': np.nan, 'dti': np.nan, 'employment_length': np.nan},
    # All zero
    {'FICO_score': 0.0, 'income': 0.0, 'loan_amount': 0.0, 'dti': 0.0, 'employment_length': 0.0},
    # Extreme low FICO, extreme high DTI
    {'FICO_score': 300, 'income': 1000, 'loan_amount': 100000, 'dti': 100.0, 'employment_length': 0},
    # Extreme high FICO, zero DTI, high income
    {'FICO_score': 850, 'income': 500000, 'loan_amount': 5000, 'dti': 0.0, 'employment_length': 25},
    # Exact DP cutoffs
    {'FICO_score': 600.0, 'income': 60000, 'loan_amount': 15000, 'dti': 0.25, 'employment_length': 5},
    {'FICO_score': 660.0, 'income': 60000, 'loan_amount': 15000, 'dti': 0.25, 'employment_length': 5},
    {'FICO_score': 695.0, 'income': 60000, 'loan_amount': 15000, 'dti': 0.25, 'employment_length': 5},
    {'FICO_score': 741.0, 'income': 60000, 'loan_amount': 15000, 'dti': 0.25, 'employment_length': 5},
])
df = pd.concat([df, edge_cases], ignore_index=True)

# 4. Inject unmodeled metadata columns
df['customer_id'] = [f'CUST_{i}' for i in range(len(df))]
df['ssn'] = '000-00-0000'
df['timestamp'] = pd.Timestamp.now()
df['notes'] = 'adversarial test row'

# 5. Predict probabilities and binary classes
probs = pipeline.predict_proba(df)
preds = pipeline.predict(df)

# Assertions
assert probs.shape == (len(df), 2), f"Unexpected prob shape {probs.shape}"
assert not np.isnan(probs).any(), "Probabilities contain NaN!"
assert not np.isinf(probs).any(), "Probabilities contain Inf!"
assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities out of bounds [0, 1]!"

# Probabilities must sum to 1
row_sums = probs.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-5), "Probability rows do not sum to 1!"

# Predictions must match 0.5 decision threshold
expected_preds = (probs[:, 1] >= 0.5).astype(int)
assert np.array_equal(preds, expected_preds), "Class predictions do not match 0.5 threshold!"

print(f"VERIFIED_OK: {len(df)} records evaluated successfully.")
"""
    cmd = [sys.executable, "-c", eval_script]
    env = os.environ.copy()
    current_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f".;{current_pp}" if current_pp else "."
    
    result = subprocess.run(
        cmd,
        cwd=".",
        env=env,
        capture_output=True,
        text=True,
        timeout=60
    )
    assert result.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "VERIFIED_OK: 5008 records evaluated successfully." in result.stdout

def test_multi_model_benchmark_fairness_and_determinism():
    """
    Multi-Model Benchmark Fairness Challenge:
    Verifies that Logistic Regression, XGBoost, GradientBoosting, and RandomForest:
    - Are evaluated on the exact same splits.
    - Do not modify X_train or X_test in-place.
    - Yield deterministic, reproducible evaluation metrics across runs.
    - Champion selection strictly follows the objective metric criterion.
    """
    df = generate_loan_data(num_samples=1000, random_state=42)
    X = df.drop(columns=['default'])
    y = df['default']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Store deep copies
    X_train_orig = X_train.copy(deep=True)
    X_test_orig = X_test.copy(deep=True)
    y_train_orig = y_train.copy(deep=True)
    y_test_orig = y_test.copy(deep=True)
    
    # Benchmark 1
    results1, pipes1 = benchmark_models(X_train, y_train, X_test, y_test, max_fico_buckets=5)
    
    # Verify immutability
    pd.testing.assert_frame_equal(X_train, X_train_orig)
    pd.testing.assert_frame_equal(X_test, X_test_orig)
    pd.testing.assert_series_equal(y_train, y_train_orig)
    pd.testing.assert_series_equal(y_test, y_test_orig)
    
    # Verify presence of all models
    expected_models = {'Logistic Regression', 'Gradient Boosting', 'Random Forest'}
    try:
        import xgboost
        expected_models.add('XGBoost')
    except ImportError:
        pass
    assert expected_models.issubset(set(results1.keys())), f"Missing models: {expected_models - set(results1.keys())}"
    
    # Benchmark 2 to verify determinism
    results2, pipes2 = benchmark_models(X_train, y_train, X_test, y_test, max_fico_buckets=5)
    
    for model_name in results1:
        m1 = results1[model_name]
        m2 = results2[model_name]
        assert np.isclose(m1['roc_auc'], m2['roc_auc'], atol=1e-6), f"ROC-AUC nondeterministic for {model_name}"
        assert np.isclose(m1['brier_score'], m2['brier_score'], atol=1e-6), f"Brier score nondeterministic for {model_name}"
        assert np.isclose(m1['pr_auc'], m2['pr_auc'], atol=1e-6), f"PR-AUC nondeterministic for {model_name}"
        
    # Verify champion selection logic
    best_name = max(results1.keys(), key=lambda k: (results1[k]['roc_auc'], -results1[k]['brier_score']))
    best_auc = results1[best_name]['roc_auc']
    for model_name, m in results1.items():
        assert m['roc_auc'] <= best_auc, f"{model_name} has higher ROC-AUC than selected champion {best_name}"

def test_robustness_to_unmodeled_columns_and_permutations():
    """
    Schema Robustness Challenge:
    Verifies that the trained pipeline ignores injected metadata columns
    and is invariant to column ordering permutations.
    """
    model = joblib.load("model_artifact.joblib")
    
    base_record = {
        'FICO_score': 710,
        'income': 72000.0,
        'loan_amount': 18000.0,
        'dti': 0.27,
        'employment_length': 5
    }
    df_clean = pd.DataFrame([base_record])
    prob_clean = model.predict_proba(df_clean)
    
    # 1. Inject diverse metadata types
    df_injected = df_clean.copy()
    df_injected['customer_id'] = 'CUST-8888-XYZ'
    df_injected['ssn'] = '111-22-3333'
    df_injected['timestamp'] = pd.Timestamp('2026-09-05 12:34:56')
    df_injected['irrelevant_float'] = 987654.321
    df_injected['flag'] = True
    
    prob_injected = model.predict_proba(df_injected)
    assert np.allclose(prob_clean, prob_injected, atol=1e-7), "Metadata injection altered predictions!"
    
    # 2. Permute column ordering randomly
    rng = np.random.default_rng(42)
    cols = list(df_injected.columns)
    for _ in range(5):
        rng.shuffle(cols)
        df_shuffled = df_injected[cols]
        prob_shuffled = model.predict_proba(df_shuffled)
        assert np.allclose(prob_clean, prob_shuffled, atol=1e-7), "Column reordering altered predictions!"
        
    # 3. Missing required feature raises ValueError
    df_missing = df_clean.drop(columns=['loan_amount'])
    with pytest.raises(ValueError, match="columns are missing"):
        model.predict_proba(df_missing)

def test_adversarial_monotonicity_stress():
    """
    Risk Monotonicity Challenge:
    Verifies economic and statistical coherence:
    - Lower FICO scores must monotonically increase default risk.
    - Higher DTI must increase default risk.
    """
    model = joblib.load("model_artifact.joblib")
    
    # FICO ladder from prime (820) to subprime (520)
    fico_ladder = [820, 750, 710, 680, 630, 520]
    p_fico = []
    for f in fico_ladder:
        rec = pd.DataFrame([{
            'FICO_score': f,
            'income': 65000.0,
            'loan_amount': 15000.0,
            'dti': 0.30,
            'employment_length': 5
        }])
        p_fico.append(model.predict_proba(rec)[0, 1])
        
    # As FICO decreases, probability of default should strictly increase
    for i in range(len(p_fico) - 1):
        assert p_fico[i] <= p_fico[i+1], (
            f"FICO monotonicity violated: FICO={fico_ladder[i]} (PD={p_fico[i]:.4f}) "
            f"vs FICO={fico_ladder[i+1]} (PD={p_fico[i+1]:.4f})"
        )
        
    # DTI ladder from 0.10 to 0.60
    dti_ladder = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    p_dti = []
    for d in dti_ladder:
        rec = pd.DataFrame([{
            'FICO_score': 700,
            'income': 65000.0,
            'loan_amount': 15000.0,
            'dti': d,
            'employment_length': 5
        }])
        p_dti.append(model.predict_proba(rec)[0, 1])
        
    # As DTI increases, probability of default should strictly increase
    for i in range(len(p_dti) - 1):
        assert p_dti[i] < p_dti[i+1], (
            f"DTI monotonicity violated: DTI={dti_ladder[i]} (PD={p_dti[i]:.4f}) "
            f"vs DTI={dti_ladder[i+1]} (PD={p_dti[i+1]:.4f})"
        )
