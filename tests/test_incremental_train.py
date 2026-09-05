"""
Unit and integration tests for incremental_train.py (R1 requirements).
Covers:
- Dataset harmonization loaders (Lending Club, Generic Credit Risk, German Credit)
- Unified schema contract across all 3 domains
- Global 37-column preprocessor contract and category mapping
- Sequential XGBoost booster continuation tree accumulation (100 -> 200 -> 300 trees)
- Artifact persistence and multi-domain inference fidelity
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from incremental_train import (
    load_and_harmonize_lending_club,
    load_and_harmonize_credit_risk,
    load_and_harmonize_german,
    run_incremental_training
)


class TestHarmonizationLoaders:
    """Tests for dataset harmonization and schema standardization across data sources."""

    def test_lending_club_harmonization_contract(self):
        """Validates Lending Club loader: canonical columns, binary target, mapped employment."""
        df = load_and_harmonize_lending_club()
        expected_cols = {'loan_amount', 'emp_length', 'purpose', 'target'}
        assert set(df.columns) == expected_cols
        assert len(df) > 30000

        # Validate target
        unique_targets = set(df['target'].unique())
        assert unique_targets.issubset({0, 1})
        assert not df['target'].isna().any()

        # Validate employment mapping categories
        valid_emp_tiers = {'< 1 year', '1-4 years', '4-7 years', '7+ years', 'unknown'}
        emp_categories = set(df['emp_length'].unique())
        assert emp_categories.issubset(valid_emp_tiers)

        # Validate loan_amount is positive numeric
        assert pd.api.types.is_numeric_dtype(df['loan_amount'])
        assert (df['loan_amount'] > 0).all()

    def test_credit_risk_harmonization_contract(self):
        """Validates Generic Credit Risk loader: continuous employment binning into standard tiers."""
        df = load_and_harmonize_credit_risk()
        canonical_cols = {'loan_amount', 'purpose', 'target', 'emp_length'}
        assert set(df.columns) == canonical_cols
        assert len(df) > 30000

        # Check target
        unique_targets = set(df['target'].unique())
        assert unique_targets.issubset({0, 1})

        # Ensure person_emp_length was dropped
        assert 'person_emp_length' not in df.columns

        # Verify employment length was binned to standard strings
        valid_emp_tiers = {'< 1 year', '1-4 years', '4-7 years', '7+ years', 'unknown'}
        emp_categories = set(df['emp_length'].unique())
        assert emp_categories.issubset(valid_emp_tiers)

    def test_german_harmonization_contract(self):
        """Validates German Credit loader: integer code mapping and purpose string conversion."""
        df = load_and_harmonize_german()
        canonical_cols = {'loan_amount', 'purpose', 'target', 'emp_length'}
        assert set(df.columns) == canonical_cols
        assert len(df) == 1000

        # Check target
        unique_targets = set(df['target'].unique())
        assert unique_targets.issubset({0, 1})

        # Ensure beszeit was dropped and unemployed category is present
        assert 'beszeit' not in df.columns
        assert 'unemployed' in df['emp_length'].values
        assert pd.api.types.is_string_dtype(df['purpose'])

    def test_harmonized_datasets_shared_schema(self):
        """Confirms that all 3 harmonized DataFrames share identical canonical column sets."""
        df_lc = load_and_harmonize_lending_club()
        df_cr = load_and_harmonize_credit_risk()
        df_ger = load_and_harmonize_german()

        canonical_set = {'loan_amount', 'emp_length', 'purpose', 'target'}
        assert set(df_lc.columns) == canonical_set
        assert set(df_cr.columns) == canonical_set
        assert set(df_ger.columns) == canonical_set


class TestIncrementalPreprocessorContract:
    """Tests for the unified 37-column preprocessor contract."""

    @pytest.fixture(scope="class")
    def fitted_global_preprocessor(self):
        """Fits the global ColumnTransformer across all 3 harmonized datasets."""
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
        return preprocessor

    def test_preprocessor_produces_exactly_37_features(self, fitted_global_preprocessor):
        """Asserts that the global preprocessor transforms data into exactly 37 features."""
        preprocessor = fitted_global_preprocessor
        feature_names = preprocessor.get_feature_names_out()

        assert len(feature_names) == 37, f"Expected 37 features, got {len(feature_names)}"

        # Verify feature breakdown
        num_features = [f for f in feature_names if f.startswith('num__')]
        emp_features = [f for f in feature_names if 'emp_length' in f]
        purpose_features = [f for f in feature_names if 'purpose' in f]

        assert len(num_features) == 1
        assert len(emp_features) == 6
        assert len(purpose_features) == 30

    def test_unseen_category_robustness(self, fitted_global_preprocessor):
        """Unseen purpose or employment levels must produce all-zero one-hot vectors without crashing."""
        preprocessor = fitted_global_preprocessor
        novel_record = pd.DataFrame([{
            'loan_amount': 25000.0,
            'emp_length': '25_years_veteran',
            'purpose': 'exotic_cryptocurrency_mining'
        }])

        transformed = preprocessor.transform(novel_record)
        assert transformed.shape == (1, 37)
        assert not np.isnan(transformed).any()
        # All one-hot columns (indices 1..36) should be 0.0 because categories were unseen
        assert np.all(transformed[0, 1:] == 0.0)


class TestSequentialBoosterContinuation:
    """Tests for sequential XGBoost tree accumulation across multiple training stages."""

    def test_tree_accumulation_100_to_200_to_300(self):
        """
        Validates sequential booster continuation via xgb_model parameter:
        Stage 1: 100 trees -> Stage 2: 200 trees -> Stage 3: 300 trees.
        """
        # Create deterministic synthetic batches of 37 features
        np.random.seed(42)
        n = 300
        X1 = np.random.randn(n, 37)
        y1 = np.random.binomial(1, 0.2, n)
        X2 = np.random.randn(n, 37)
        y2 = np.random.binomial(1, 0.25, n)
        X3 = np.random.randn(n, 37)
        y3 = np.random.binomial(1, 0.3, n)

        # Stage 1: Initial fit
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='logloss')
        clf.fit(X1, y1)
        booster1 = clf.get_booster()
        assert len(booster1.get_dump()) == 100, f"Expected 100 trees after Stage 1, got {len(booster1.get_dump())}"

        # Stage 2: Continuation with xgb_model
        clf.fit(X2, y2, xgb_model=clf.get_booster())
        booster2 = clf.get_booster()
        assert len(booster2.get_dump()) == 200, f"Expected 200 trees after Stage 2, got {len(booster2.get_dump())}"

        # Stage 3: Continuation with xgb_model
        clf.fit(X3, y3, xgb_model=clf.get_booster())
        booster3 = clf.get_booster()
        assert len(booster3.get_dump()) == 300, f"Expected 300 trees after Stage 3, got {len(booster3.get_dump())}"


class TestIncrementalMasterModelArtifact:
    """Tests for incremental_master_model.joblib persistence, structure, and inference."""

    def test_run_incremental_training_execution_and_persistence(self):
        """Executes run_incremental_training and asserts master model artifact is saved."""
        run_incremental_training()
        out_file = 'models/incremental_master_model.joblib'
        assert os.path.exists(out_file), f"Expected artifact {out_file} to exist"

        pipeline = joblib.load(out_file)
        assert isinstance(pipeline, Pipeline)
        assert 'preprocessor' in pipeline.named_steps
        assert 'classifier' in pipeline.named_steps

        booster = pipeline.named_steps['classifier'].get_booster()
        assert len(booster.get_dump()) == 300, f"Expected 300 trees in master model, got {len(booster.get_dump())}"

    def test_incremental_master_model_inference_all_domains(self):
        """Verifies calibrated inference on borrower profiles from all 3 training domains."""
        out_file = 'models/incremental_master_model.joblib'
        pipeline = joblib.load(out_file)

        # 1. Lending Club borrower profile
        lc_borrower = pd.DataFrame([{
            'loan_amount': 12000.0,
            'emp_length': '4-7 years',
            'purpose': 'debt_consolidation'
        }])
        prob_lc = pipeline.predict_proba(lc_borrower)
        assert prob_lc.shape == (1, 2)
        assert 0.0 <= prob_lc[0, 1] <= 1.0
        assert np.isclose(prob_lc.sum(), 1.0, atol=1e-6)

        # 2. Credit Risk borrower profile
        cr_borrower = pd.DataFrame([{
            'loan_amount': 8500.0,
            'emp_length': '1-4 years',
            'purpose': 'VENTURE'
        }])
        prob_cr = pipeline.predict_proba(cr_borrower)
        assert prob_cr.shape == (1, 2)
        assert 0.0 <= prob_cr[0, 1] <= 1.0
        assert np.isclose(prob_cr.sum(), 1.0, atol=1e-6)

        # 3. German Credit borrower profile
        ger_borrower = pd.DataFrame([{
            'loan_amount': 3000.0,
            'emp_length': 'unemployed',
            'purpose': '1'
        }])
        prob_ger = pipeline.predict_proba(ger_borrower)
        assert prob_ger.shape == (1, 2)
        assert 0.0 <= prob_ger[0, 1] <= 1.0
        assert np.isclose(prob_ger.sum(), 1.0, atol=1e-6)
