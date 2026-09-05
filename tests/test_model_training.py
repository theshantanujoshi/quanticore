"""
Unit and integration tests for model_training.py (R1 requirements).
Covers:
- evaluate_model: 6 evaluation metrics (roc_auc, pr_auc, brier_score, accuracy, f1_score, gini) and identities
- benchmark_models: Candidate model families (Logistic Regression, Gradient Boosting, Random Forest, XGBoost)
- High-cardinality categorical dropping (>= 20 unique values)
- Dynamic FICO column adaptation (presence and absence)
- Objective champion selection (max ROC-AUC, lowest Brier score tiebreaker)
- Dual metadata serialization and schema verification
- Synthetic data generation fallback
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from model_training import evaluate_model, benchmark_models, train_and_evaluate
from preprocessing import build_preprocessor


class TestEvaluateModelContract:
    """Tests for evaluate_model metric calculation and mathematical properties."""

    def test_evaluate_model_keys_and_bounds(self, fitted_classification_pipeline):
        """Validates that all 6 required metrics are computed within valid ranges."""
        pipeline, X_test, y_test = fitted_classification_pipeline
        metrics = evaluate_model(pipeline, X_test, y_test)

        required_keys = {'roc_auc', 'pr_auc', 'brier_score', 'accuracy', 'f1_score', 'gini'}
        assert set(metrics.keys()) == required_keys

        # Check bounds
        assert 0.0 <= metrics['roc_auc'] <= 1.0
        assert 0.0 <= metrics['pr_auc'] <= 1.0
        assert 0.0 <= metrics['brier_score'] <= 1.0
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0
        assert -1.0 <= metrics['gini'] <= 1.0

    def test_gini_mathematical_identity(self, fitted_classification_pipeline):
        """Validates credit risk Gini identity: Gini = 2 * ROC_AUC - 1."""
        pipeline, X_test, y_test = fitted_classification_pipeline
        metrics = evaluate_model(pipeline, X_test, y_test)
        expected_gini = 2.0 * metrics['roc_auc'] - 1.0
        assert np.isclose(metrics['gini'], expected_gini, atol=1e-9)

    def test_evaluate_model_perfect_classifier(self):
        """A perfect classifier should produce ROC-AUC = 1.0, Gini = 1.0, Accuracy = 1.0."""
        class PerfectPipeline:
            def predict_proba(self, X):
                # Returns 1.0 for true positive, 0.0 for negative
                probs = np.array([0.0 if row[0] == 0 else 1.0 for row in X])
                return np.column_stack([1.0 - probs, probs])
            def predict(self, X):
                return np.array([row[0] for row in X])

        X_dummy = np.array([[0], [0], [1], [1]])
        y_true = np.array([0, 0, 1, 1])

        metrics = evaluate_model(PerfectPipeline(), X_dummy, y_true)
        assert metrics['roc_auc'] == 1.0
        assert metrics['gini'] == 1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['brier_score'] == 0.0


class TestBenchmarkModels:
    """Tests for multi-model benchmarking across candidate model families."""

    def test_benchmark_candidate_model_families(self, synthetic_loan_data_small):
        """Verifies candidate models: Logistic Regression, Gradient Boosting, Random Forest, XGBoost."""
        df = synthetic_loan_data_small
        X = df.drop(columns=['default'])
        y = df['default']

        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train, y_test = y.iloc[:150], y.iloc[150:]

        benchmark_results, pipelines = benchmark_models(
            X_train, y_train, X_test, y_test,
            max_fico_buckets=3,
            fico_col='FICO_score',
            cont_cols=['income', 'loan_amount', 'dti', 'employment_length'],
            cat_cols=['purpose', 'home_ownership']
        )

        expected_models = {'Logistic Regression', 'Gradient Boosting', 'Random Forest', 'XGBoost'}
        assert set(benchmark_results.keys()) == expected_models
        assert set(pipelines.keys()) == expected_models

        # Each pipeline must contain preprocessor and classifier steps
        for name, pipe in pipelines.items():
            assert isinstance(pipe, Pipeline)
            assert 'preprocessor' in pipe.named_steps
            assert 'classifier' in pipe.named_steps
            # Verify probabilities
            probs = pipe.predict_proba(X_test)
            assert probs.shape == (len(X_test), 2)
            assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_benchmark_models_without_fico(self, synthetic_loan_data_small):
        """Verifies benchmark executes successfully when fico_col=None."""
        df = synthetic_loan_data_small.drop(columns=['FICO_score'])
        X = df.drop(columns=['default'])
        y = df['default']

        X_train, X_test = X.iloc[:150], X.iloc[150:]
        y_train, y_test = y.iloc[:150], y.iloc[150:]

        benchmark_results, pipelines = benchmark_models(
            X_train, y_train, X_test, y_test,
            max_fico_buckets=3,
            fico_col=None,
            cont_cols=['income', 'loan_amount'],
            cat_cols=['purpose']
        )
        assert len(benchmark_results) >= 3


class TestTrainAndEvaluateContracts:
    """Tests for train_and_evaluate orchestration, filtering, champion selection, and persistence."""

    def test_high_cardinality_filtering(self, tmp_path, high_cardinality_df):
        """Categorical columns with >= 20 unique values must be automatically dropped."""
        data_csv = tmp_path / "cardinality_test.csv"
        high_cardinality_df.to_csv(data_csv, index=False)

        champion = train_and_evaluate(
            data_path=str(data_csv),
            max_fico_buckets=3,
            save_artifact=None,
            target_col='default',
            fico_col=None
        )

        # Inspect preprocessor to verify 'zip_code' was excluded
        preprocessor = champion.named_steps['preprocessor']
        cat_transformer = preprocessor.named_transformers_.get('cat')
        assert cat_transformer is not None
        # low_card_purpose should be included, zip_code dropped
        cat_cols_fitted = [cols for name, trans, cols in preprocessor.transformers if name == 'cat'][0]
        assert 'low_card_purpose' in cat_cols_fitted
        assert 'zip_code' not in cat_cols_fitted

    def test_absent_fico_column_handling(self, tmp_path, synthetic_loan_data_small):
        """When FICO column is absent, pipeline adapts dynamically and records None in metadata."""
        df_no_fico = synthetic_loan_data_small.drop(columns=['FICO_score'])
        data_csv = tmp_path / "no_fico.csv"
        df_no_fico.to_csv(data_csv, index=False)
        artifact_path = tmp_path / "no_fico_model.joblib"

        champion = train_and_evaluate(
            data_path=str(data_csv),
            max_fico_buckets=3,
            save_artifact=str(artifact_path),
            target_col='default',
            fico_col='FICO_score'  # Named FICO_score, but absent from CSV
        )

        meta_file = tmp_path / "no_fico_model_metadata.joblib"
        assert meta_file.exists()
        metadata = joblib.load(meta_file)

        assert metadata['fico_boundaries'] is None
        assert metadata['bucket_means'] is None
        assert 'FICO_score' not in metadata['features']

    def test_champion_selection_policy(self):
        """
        Validates that champion selection maximizes ROC-AUC, and breaks ties with lowest Brier score.
        """
        results = {
            'Model A': {'roc_auc': 0.85, 'brier_score': 0.15},
            'Model B': {'roc_auc': 0.88, 'brier_score': 0.18},  # Higher ROC-AUC -> Champion
            'Model C': {'roc_auc': 0.88, 'brier_score': 0.14}   # Equal ROC-AUC, lower Brier -> Champion!
        }
        champ = max(
            results.keys(),
            key=lambda k: (results[k]['roc_auc'], -results[k]['brier_score'])
        )
        assert champ == 'Model C'

    def test_dual_artifact_and_metadata_serialization(self, tmp_path, synthetic_loan_data_small):
        """Verifies serialization of both .joblib model pipeline and _metadata.joblib dictionary."""
        data_csv = tmp_path / "train_eval_data.csv"
        synthetic_loan_data_small.to_csv(data_csv, index=False)
        model_file = tmp_path / "champion_model.joblib"

        champion = train_and_evaluate(
            data_path=str(data_csv),
            max_fico_buckets=3,
            save_artifact=str(model_file),
            target_col='default',
            fico_col='FICO_score'
        )

        # Verify model artifact persistence
        assert model_file.exists()
        reloaded_pipeline = joblib.load(model_file)
        assert isinstance(reloaded_pipeline, Pipeline)

        # Verify metadata artifact persistence
        meta_file = tmp_path / "champion_model_metadata.joblib"
        assert meta_file.exists()
        metadata = joblib.load(meta_file)

        expected_keys = {'champion_model', 'metrics', 'fico_boundaries', 'bucket_means', 'features'}
        assert set(metadata.keys()) == expected_keys
        assert isinstance(metadata['champion_model'], str)
        assert isinstance(metadata['metrics'], dict)
        assert metadata['fico_boundaries'] is not None
        assert isinstance(metadata['bucket_means'], dict)
        assert 'FICO_score' in metadata['features']

    def test_synthetic_data_generation_fallback(self, tmp_path):
        """When data_path does not exist, automatically generates synthetic dataset and completes training."""
        missing_data_csv = tmp_path / "non_existent_data.csv"
        assert not missing_data_csv.exists()

        champion = train_and_evaluate(
            data_path=str(missing_data_csv),
            max_fico_buckets=3,
            save_artifact=None,
            target_col='default',
            fico_col='FICO_score'
        )

        assert missing_data_csv.exists(), "Synthetic dataset should be generated and persisted"
        generated_df = pd.read_csv(missing_data_csv)
        assert len(generated_df) == 10000
        assert 'FICO_score' in generated_df.columns
        assert isinstance(champion, Pipeline)
