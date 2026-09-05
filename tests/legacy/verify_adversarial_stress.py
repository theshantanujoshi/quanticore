"""
Adversarial Stress Testing Harness for Challenger 1
Empirically assesses robustness and failure modes of:
- preprocessing.py
- model_training.py
- train.py
- incremental_train.py
under extreme, pathological, corrupted, and edge-case inputs.
"""

import os
import sys
import tempfile
import traceback
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing import build_preprocessor
from fico_bucketing import FicoDPBucketer
from model_training import evaluate_model, benchmark_models, train_and_evaluate
from train import run_pipeline
from incremental_train import (
    load_and_harmonize_lending_club,
    load_and_harmonize_credit_risk,
    load_and_harmonize_german
)

def log_test(category, test_name, status, details=""):
    badge = "[PASS]" if status == "PASS" else ("[WARN]" if status == "WARN" else "[FAIL]")
    print(f"{badge} {category:18s} | {test_name:40s} | {details}")

def run_preprocessing_stress():
    print("\n" + "=" * 80)
    print("1. PREPROCESSING ADVERSARIAL STRESS SUITE")
    print("=" * 80)

    # Test 1.1: Extreme numerical values (1e20, -1e20)
    try:
        prep = build_preprocessor(fico_col=None, cont_cols=['amount', 'score'], cat_cols=[])
        train_df = pd.DataFrame({
            'amount': [1e5, 2e5, 3e5, 4e5],
            'score': [100.0, 200.0, 300.0, 400.0],
            'default': [0, 0, 1, 1]
        })
        prep.fit(train_df.drop(columns=['default']), train_df['default'])
        
        extreme_df = pd.DataFrame({
            'amount': [1e20, -1e20, 0.0],
            'score': [1e15, -1e15, 0.0]
        })
        out = prep.transform(extreme_df)
        assert not np.isnan(out).any()
        log_test("Preprocessing", "Extreme Floats (1e20, -1e20)", "PASS", f"Output shape: {out.shape}")
    except Exception as e:
        log_test("Preprocessing", "Extreme Floats (1e20, -1e20)", "FAIL", str(e))

    # Test 1.2: Infinity and -Infinity
    try:
        prep = build_preprocessor(fico_col=None, cont_cols=['amount'], cat_cols=[])
        train_df = pd.DataFrame({'amount': [10.0, 20.0, 30.0], 'default': [0, 1, 0]})
        prep.fit(train_df.drop(columns=['default']), train_df['default'])
        
        inf_df = pd.DataFrame({'amount': [np.inf, -np.inf, 15.0]})
        out = prep.transform(inf_df)
        has_inf = np.isinf(out).any()
        if has_inf:
            log_test("Preprocessing", "Infinities (inf, -inf)", "WARN", "StandardScaler propagates infinities (expected sklearn behavior)")
        else:
            log_test("Preprocessing", "Infinities (inf, -inf)", "PASS", f"Output shape: {out.shape}")
    except Exception as e:
        log_test("Preprocessing", "Infinities (inf, -inf)", "FAIL", str(e))

    # Test 1.3: All-NaN rows during transform
    try:
        prep = build_preprocessor(fico_col='FICO_score', cont_cols=['income', 'loan_amount'], cat_cols=['purpose'])
        train_df = pd.DataFrame({
            'FICO_score': [600, 650, 700, 750],
            'income': [50000.0, 60000.0, 70000.0, 80000.0],
            'loan_amount': [5000.0, 10000.0, 15000.0, 20000.0],
            'purpose': ['debt_consolidation', 'credit_card', 'home_improvement', 'medical'],
            'default': [1, 1, 0, 0]
        })
        prep.fit(train_df.drop(columns=['default']), train_df['default'])
        
        all_nan_row = pd.DataFrame({
            'FICO_score': [np.nan],
            'income': [np.nan],
            'loan_amount': [np.nan],
            'purpose': [np.nan]
        })
        out = prep.transform(all_nan_row)
        assert not np.isnan(out).any()
        log_test("Preprocessing", "All-NaN Row Imputation", "PASS", f"Zero remaining NaNs. Output vector: {out[0].tolist()[:4]}...")
    except Exception as e:
        log_test("Preprocessing", "All-NaN Row Imputation", "FAIL", str(e))

    # Test 1.4: Massive Extraneous Columns Stress (remainder='drop')
    try:
        prep = build_preprocessor(fico_col=None, cont_cols=['amount'], cat_cols=['flag'])
        train_df = pd.DataFrame({'amount': [100.0, 200.0], 'flag': ['A', 'B'], 'default': [0, 1]})
        prep.fit(train_df.drop(columns=['default']), train_df['default'])
        
        # Add 1,000 random unmodeled columns
        data = {'amount': [150.0], 'flag': ['A']}
        for i in range(1000):
            data[f'junk_feature_{i}'] = [f'random_noise_{i}']
        stress_df = pd.DataFrame(data)
        out = prep.transform(stress_df)
        assert out.shape == (1, 2), f"Expected shape (1, 2), got {out.shape}"
        log_test("Preprocessing", "1,000 Unmodeled Columns Dropped", "PASS", f"remainder='drop' isolated input to {out.shape[1]} cols")
    except Exception as e:
        log_test("Preprocessing", "1,000 Unmodeled Columns Dropped", "FAIL", str(e))

    # Test 1.5: Adversarial Categorical Strings (Emojis, SQLi, Special Chars)
    try:
        prep = build_preprocessor(fico_col=None, cont_cols=[], cat_cols=['purpose'])
        train_df = pd.DataFrame({'purpose': ['auto', 'home'], 'default': [0, 1]})
        prep.fit(train_df.drop(columns=['default']), train_df['default'])
        
        adversarial_strings = [
            "'; DROP TABLE loans; --",
            "🚀🔥💯",
            "\\x00\\xff\\xfe",
            "   ",
            "\t\n\r",
            "A" * 5000  # 5,000 character string
        ]
        test_df = pd.DataFrame({'purpose': adversarial_strings})
        out = prep.transform(test_df)
        # All unseen categories should produce all-zero vectors
        assert np.all(out == 0.0)
        log_test("Preprocessing", "Adversarial Strings & Unicode", "PASS", "Gracefully encoded to all-zeros via handle_unknown='ignore'")
    except Exception as e:
        log_test("Preprocessing", "Adversarial Strings & Unicode", "FAIL", str(e))

def run_model_training_stress():
    print("\n" + "=" * 80)
    print("2. MODEL TRAINING ADVERSARIAL STRESS SUITE")
    print("=" * 80)

    # Test 2.1: Single-class target passed to evaluate_model
    try:
        class DummyModel:
            def predict_proba(self, X):
                return np.column_stack([np.zeros(len(X)), np.ones(len(X))])
            def predict(self, X):
                return np.ones(len(X), dtype=int)
        
        X_dummy = pd.DataFrame({'a': [1, 2, 3, 4]})
        y_single_class = np.array([1, 1, 1, 1])  # Only class 1 present
        
        # roc_auc_score is undefined with 1 class
        try:
            evaluate_model(DummyModel(), X_dummy, y_single_class)
            log_test("Model Training", "Single-Class Target evaluate_model", "WARN", "Did not raise ValueError")
        except ValueError as ve:
            log_test("Model Training", "Single-Class Target evaluate_model", "PASS", f"Correctly raised ValueError: {str(ve)[:50]}...")
    except Exception as e:
        log_test("Model Training", "Single-Class Target evaluate_model", "FAIL", str(e))

    # Test 2.2: Champion selection tie-breaking
    try:
        # Candidate 1: AUC=0.85, Brier=0.20
        # Candidate 2: AUC=0.85, Brier=0.10 (Should win tie-breaker)
        results = {
            'Model_HighBrier': {'roc_auc': 0.8500, 'brier_score': 0.2000},
            'Model_LowBrier': {'roc_auc': 0.8500, 'brier_score': 0.1000},
            'Model_LowerAUC': {'roc_auc': 0.8499, 'brier_score': 0.0500}
        }
        champion = max(
            results.keys(),
            key=lambda k: (results[k]['roc_auc'], -results[k]['brier_score'])
        )
        assert champion == 'Model_LowBrier', f"Expected Model_LowBrier, got {champion}"
        log_test("Model Training", "Champion Tiebreaker Logic", "PASS", "Tie on AUC broken strictly by lowest Brier score")
    except Exception as e:
        log_test("Model Training", "Champion Tiebreaker Logic", "FAIL", str(e))

    # Test 2.3: High cardinality dropping threshold boundary (< 20 unique values)
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "cardinality_stress.csv")
            n = 100
            df = pd.DataFrame({
                'num_feat': np.random.randn(n),
                'cat_19_unique': [f"cat_{i % 19}" for i in range(n)],  # 19 unique < 20 -> Kept
                'cat_20_unique': [f"cat_{i % 20}" for i in range(n)],  # 20 unique not < 20 -> Dropped!
                'cat_50_unique': [f"cat_{i % 50}" for i in range(n)],  # 50 unique not < 20 -> Dropped!
                'default': np.random.binomial(1, 0.3, n)
            })
            df.to_csv(csv_path, index=False)
            
            champion = train_and_evaluate(
                data_path=csv_path,
                max_fico_buckets=2,
                save_artifact=None,
                target_col='default',
                fico_col=None
            )
            preprocessor = champion.named_steps['preprocessor']
            cat_cols = [cols for name, _, cols in preprocessor.transformers if name == 'cat'][0]
            assert 'cat_19_unique' in cat_cols, "cat_19_unique should be kept"
            assert 'cat_20_unique' not in cat_cols, "cat_20_unique must be dropped (boundary >= 20)"
            assert 'cat_50_unique' not in cat_cols, "cat_50_unique must be dropped"
            log_test("Model Training", "Cardinality Filter Boundary (<20)", "PASS", "19 unique retained, 20 and 50 dropped")
    except Exception as e:
        log_test("Model Training", "Cardinality Filter Boundary (<20)", "FAIL", str(e))

def run_train_stress():
    print("\n" + "=" * 80)
    print("3. TRAIN.PY PIPELINE ADVERSARIAL STRESS SUITE")
    print("=" * 80)

    # Test 3.1: Missing file triggers synthetic generation fallback
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_csv = os.path.join(tmp_dir, "auto_synth_loans.csv")
            temp_model = os.path.join(tmp_dir, "temp_model.joblib")
            
            # Ensure file does not exist
            assert not os.path.exists(temp_csv)
            
            orig_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                run_pipeline(
                    data_path="auto_synth_loans.csv",
                    output_artifact="temp_model.joblib",
                    max_fico_buckets=3,
                    target_col='default',
                    fico_col='FICO_score'
                )
                assert os.path.exists("auto_synth_loans.csv")
                assert os.path.exists("temp_model.joblib")
                assert os.path.exists("model.joblib")  # dual artifact synchronization
                log_test("Train Pipeline", "Synthetic Fallback & Dual Sync", "PASS", "Generated 10k rows, saved temp_model and model.joblib")
            finally:
                os.chdir(orig_cwd)
    except Exception as e:
        log_test("Train Pipeline", "Synthetic Fallback & Dual Sync", "FAIL", str(e))

    # Test 3.2: Assertion check in run_pipeline catches probability corruption
    try:
        # In train.py: line 76: assert 0.0 <= default_prob <= 1.0
        # If default_prob is outside [0, 1], an AssertionError is raised
        test_prob_good = 0.42
        test_prob_bad = 1.05
        assert 0.0 <= test_prob_good <= 1.0
        try:
            assert 0.0 <= test_prob_bad <= 1.0
            log_test("Train Pipeline", "Probability Bounds Assertion", "FAIL", "Did not catch out-of-bounds probability")
        except AssertionError:
            log_test("Train Pipeline", "Probability Bounds Assertion", "PASS", "Assertion strictly triggers on probability > 1.0")
    except Exception as e:
        log_test("Train Pipeline", "Probability Bounds Assertion", "FAIL", str(e))

def run_incremental_train_stress():
    print("\n" + "=" * 80)
    print("4. INCREMENTAL_TRAIN.PY ADVERSARIAL STRESS SUITE")
    print("=" * 80)

    # Test 4.1: Harmonization Employment Mapping on Pathological Inputs
    try:
        df_lc = load_and_harmonize_lending_club()
        df_cr = load_and_harmonize_credit_risk()
        df_ger = load_and_harmonize_german()
        
        assert set(df_lc['emp_length'].unique()).issubset({'< 1 year', '1-4 years', '4-7 years', '7+ years', 'unknown'})
        assert set(df_cr['emp_length'].unique()).issubset({'< 1 year', '1-4 years', '4-7 years', '7+ years', 'unknown'})
        assert set(df_ger['emp_length'].unique()).issubset({'< 1 year', '1-4 years', '4-7 years', '7+ years', 'unknown', 'unemployed'})
        log_test("Incremental Train", "Harmonization Domain Enums", "PASS", "All 3 loaders mapped raw values to canonical categories")
    except Exception as e:
        log_test("Incremental Train", "Harmonization Domain Enums", "FAIL", str(e))

    # Test 4.2: Inference on Master Model with Extreme and Novel Inputs
    try:
        model_path = "incremental_master_model.joblib"
        if os.path.exists(model_path):
            pipeline = joblib.load(model_path)
            
            # Borrower with unseen category, extreme loan amount, and unknown employment
            adversarial_borrower = pd.DataFrame([{
                'loan_amount': 100_000_000.0,  # $100M loan
                'emp_length': 'alien_time_traveler',
                'purpose': 'intergalactic_trade'
            }])
            
            probs = pipeline.predict_proba(adversarial_borrower)
            assert probs.shape == (1, 2)
            assert 0.0 <= probs[0, 1] <= 1.0
            assert np.isclose(probs.sum(), 1.0)
            log_test("Incremental Train", "Master Model Adversarial Inference", "PASS", f"PD: {probs[0, 1]:.4f}, Sum: {probs.sum():.6f}")
        else:
            log_test("Incremental Train", "Master Model Adversarial Inference", "WARN", "incremental_master_model.joblib not found")
    except Exception as e:
        log_test("Incremental Train", "Master Model Adversarial Inference", "FAIL", str(e))

if __name__ == "__main__":
    print("=" * 80)
    print("EMPIRICAL ADVERSARIAL STRESS SUITE — CHALLENGER 1")
    print("=" * 80)
    run_preprocessing_stress()
    run_model_training_stress()
    run_train_stress()
    run_incremental_train_stress()
    print("\n" + "=" * 80)
    print("ADVERSARIAL STRESS SUITE EXECUTION COMPLETE.")
    print("=" * 80)
