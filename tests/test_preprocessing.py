"""
Unit and integration tests for preprocessing.py and fico_bucketing.py (R1 requirements).
Covers:
- FICO DP bucketer contracts (SSE minimization, NaN handling, boundaries, container polymorphism, edge cases)
- SimpleImputer (median for continuous, most_frequent for categorical)
- OneHotEncoder (drop='if_binary', handle_unknown='ignore')
- ColumnTransformer remainder='drop' isolation
- Preprocessor behavior when fico_col=None
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from fico_bucketing import FicoDPBucketer
from preprocessing import build_preprocessor


class TestFicoDPBucketerContracts:
    """Test suite for FicoDPBucketer contracts and mathematical properties."""

    def test_supervised_requirement_raises_error(self):
        """FicoDPBucketer requires target y to compute default rates for SSE minimization."""
        bucketer = FicoDPBucketer(max_buckets=5)
        X = np.array([550, 600, 650, 700])
        with pytest.raises(ValueError, match="supervised transformer and requires target y"):
            bucketer.fit(X, y=None)

    def test_invalid_max_buckets_raises_error(self):
        """max_buckets must be >= 1."""
        bucketer = FicoDPBucketer(max_buckets=0)
        X = np.array([550, 600, 650])
        y = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="max_buckets must be >= 1"):
            bucketer.fit(X, y)

    def test_length_mismatch_raises_error(self):
        """Length of X and y must be identical."""
        bucketer = FicoDPBucketer(max_buckets=3)
        X = np.array([550, 600, 650])
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="X and y must have equal length"):
            bucketer.fit(X, y)

    def test_unfitted_transform_raises_not_fitted(self):
        """Calling transform on unfitted bucketer raises NotFittedError."""
        bucketer = FicoDPBucketer(max_buckets=3)
        with pytest.raises(NotFittedError):
            bucketer.transform(np.array([600, 700]))

    def test_unfitted_get_feature_names_out_raises_not_fitted(self):
        """Calling get_feature_names_out on unfitted bucketer raises NotFittedError."""
        bucketer = FicoDPBucketer(max_buckets=3)
        with pytest.raises(NotFittedError):
            bucketer.get_feature_names_out()

    def test_empty_input_dataset(self):
        """Empty input arrays produce valid default boundaries and summary without crashing."""
        bucketer = FicoDPBucketer(max_buckets=3)
        bucketer.fit(np.array([]), np.array([]))
        assert bucketer.boundaries_ == [-np.inf, np.inf]
        assert len(bucketer.bucket_summary_) == 1
        assert bucketer.bucket_summary_[0]['count'] == 0

    def test_single_unique_fico_value(self):
        """Single unique FICO score produces a single bucket [-inf, inf)."""
        bucketer = FicoDPBucketer(max_buckets=5)
        X = np.array([720, 720, 720, 720])
        y = np.array([0, 1, 0, 0])
        bucketer.fit(X, y)
        assert bucketer.boundaries_ == [-np.inf, np.inf]
        assert bucketer.bucket_means_[0] == 0.25
        transformed = bucketer.transform(X)
        assert np.all(transformed == 0)

    def test_max_buckets_greater_than_unique_ficos(self):
        """When max_buckets exceeds unique FICO count, algorithm clamps to unique count."""
        bucketer = FicoDPBucketer(max_buckets=10)
        X = np.array([500, 600, 700])
        y = np.array([1, 0, 0])
        bucketer.fit(X, y)
        # Should create at most 3 buckets (len(boundaries_) <= 4)
        assert len(bucketer.boundaries_) <= 4
        assert bucketer.boundaries_[0] == -np.inf
        assert bucketer.boundaries_[-1] == np.inf

    def test_boundary_determination_and_sse_minimization(self):
        """
        Validates optimal partitioning on bimodal default distribution:
        Subprime [500..550] with 90% default, Prime [700..750] with 10% default.
        Optimal split must strictly lie between subprime and prime groups.
        """
        np.random.seed(42)
        n = 200
        ficos_low = np.random.randint(500, 560, n // 2)
        defaults_low = np.random.binomial(1, 0.9, n // 2)
        ficos_high = np.random.randint(700, 760, n // 2)
        defaults_high = np.random.binomial(1, 0.1, n // 2)

        X = np.concatenate([ficos_low, ficos_high])
        y = np.concatenate([defaults_low, defaults_high])

        bucketer = FicoDPBucketer(max_buckets=2)
        bucketer.fit(X, y)

        assert len(bucketer.boundaries_) == 3
        assert bucketer.boundaries_[0] == -np.inf
        assert bucketer.boundaries_[2] == np.inf
        
        split = bucketer.boundaries_[1]
        assert 550 <= split <= 700, f"Split cutoff {split} should separate low and high risk groups"
        assert bucketer.bucket_means_[0] > bucketer.bucket_means_[1]

    def test_nan_handling_during_fit_and_transform(self):
        """NaNs during fit are handled via groupby; NaNs during transform produce NaN bucket indices."""
        bucketer = FicoDPBucketer(max_buckets=3)
        X = np.array([500.0, 600.0, np.nan, 700.0, 800.0, 500.0])
        y = np.array([1, 0, 0, 0, 0, 1])
        bucketer.fit(X, y)
        
        # Boundaries must be valid floats
        assert bucketer.boundaries_[0] == -np.inf
        assert bucketer.boundaries_[-1] == np.inf

        # Transform with NaN values
        X_test = np.array([550.0, np.nan, 750.0])
        transformed = bucketer.transform(X_test)
        assert not np.isnan(transformed[0, 0])
        assert np.isnan(transformed[1, 0])
        assert not np.isnan(transformed[2, 0])

    def test_container_polymorphism(self):
        """FicoDPBucketer must produce identical results across array, Series, DataFrame, and list inputs."""
        ficos = [550, 600, 650, 700, 750]
        defaults = [1, 1, 0, 0, 0]

        b_list = FicoDPBucketer(max_buckets=2).fit(ficos, defaults)
        b_1d = FicoDPBucketer(max_buckets=2).fit(np.array(ficos), np.array(defaults))
        b_2d = FicoDPBucketer(max_buckets=2).fit(np.array(ficos).reshape(-1, 1), defaults)
        b_series = FicoDPBucketer(max_buckets=2).fit(pd.Series(ficos), pd.Series(defaults))
        b_df = FicoDPBucketer(max_buckets=2).fit(pd.DataFrame({'FICO': ficos}), defaults)

        assert b_list.boundaries_ == b_1d.boundaries_ == b_2d.boundaries_ == b_series.boundaries_ == b_df.boundaries_

        # Verify transform polymorphism
        t_1d = b_1d.transform(np.array([570, 720]))
        t_df = b_1d.transform(pd.DataFrame({'FICO': [570, 720]}))
        np.testing.assert_array_equal(t_1d, t_df)

    def test_extreme_and_unseen_score_mapping(self):
        """Scores beyond US credit range (-500, 1500) fall into [-inf, b1) and [bk-1, inf)."""
        bucketer = FicoDPBucketer(max_buckets=3)
        X = np.array([500, 600, 650, 700, 800])
        y = np.array([1, 1, 0, 0, 0])
        bucketer.fit(X, y)

        test_scores = np.array([-500, 300, 850, 1500])
        res = bucketer.transform(test_scores)
        assert res[0, 0] == 0
        assert res[-1, 0] == len(bucketer.boundaries_) - 2

    def test_bucket_summary_structure(self):
        """bucket_summary_ must provide dictionary records containing all reporting fields."""
        bucketer = FicoDPBucketer(max_buckets=3)
        X = np.array([500, 600, 700, 800])
        y = np.array([1, 1, 0, 0])
        bucketer.fit(X, y)

        assert hasattr(bucketer, 'bucket_summary_')
        for item in bucketer.bucket_summary_:
            assert 'bucket' in item
            assert 'range' in item
            assert 'count' in item
            assert 'defaults' in item
            assert 'default_rate' in item
            assert 0.0 <= item['default_rate'] <= 1.0


class TestBuildPreprocessorContracts:
    """Test suite for build_preprocessor ColumnTransformer contracts."""

    def test_default_structure(self):
        """Default preprocessor should include 'fico' and 'cont' transformers and remainder='drop'."""
        prep = build_preprocessor()
        transformer_names = [name for name, _, _ in prep.transformers]
        assert 'fico' in transformer_names
        assert 'cont' in transformer_names
        assert 'cat' not in transformer_names  # cat_cols is None/empty by default
        assert prep.remainder == 'drop'

    def test_preprocessor_with_no_fico(self, synthetic_loan_data_small):
        """When fico_col=None, the 'fico' transformer pipeline must be omitted."""
        prep = build_preprocessor(
            fico_col=None,
            cont_cols=['income', 'loan_amount'],
            cat_cols=['purpose']
        )
        transformer_names = [name for name, _, _ in prep.transformers]
        assert 'fico' not in transformer_names
        assert 'cont' in transformer_names
        assert 'cat' in transformer_names

        df = synthetic_loan_data_small.drop(columns=['FICO_score'])
        X = df.drop(columns=['default'])
        y = df['default']
        
        # Fit and transform should succeed without FICO column
        prep.fit(X, y)
        out = prep.transform(X)
        assert isinstance(out, np.ndarray)
        assert out.shape[0] == len(df)

    def test_continuous_median_imputation(self):
        """Continuous pipeline must impute missing values using median."""
        df_train = pd.DataFrame({
            'income': [10000.0, 20000.0, 30000.0, 100000.0],  # median = 25000
            'loan_amount': [1000.0, 2000.0, 3000.0, 4000.0],
            'default': [0, 0, 1, 1]
        })
        prep = build_preprocessor(
            fico_col=None,
            cont_cols=['income', 'loan_amount'],
            cat_cols=[]
        )
        prep.fit(df_train.drop(columns=['default']), df_train['default'])

        # Transform with NaN in income
        df_test = pd.DataFrame({
            'income': [np.nan],
            'loan_amount': [2500.0]
        })
        out = prep.transform(df_test)
        assert not np.isnan(out).any(), "Output should have zero NaNs after median imputation"

    def test_continuous_standard_scaling(self):
        """Continuous pipeline must standardize features to zero mean and unit variance on training set."""
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'income': np.random.normal(loc=50000, scale=10000, size=n),
            'loan_amount': np.random.normal(loc=15000, scale=3000, size=n),
            'default': np.random.choice([0, 1], n)
        })
        prep = build_preprocessor(
            fico_col=None,
            cont_cols=['income', 'loan_amount'],
            cat_cols=[]
        )
        prep.fit(df.drop(columns=['default']), df['default'])
        X_trans = prep.transform(df.drop(columns=['default']))

        means = np.mean(X_trans, axis=0)
        stds = np.std(X_trans, axis=0)
        np.testing.assert_allclose(means, [0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(stds, [1.0, 1.0], atol=1e-7)

    def test_continuous_zero_variance_scaling(self):
        """Columns with zero variance (all identical values) scale safely to zero without crashing."""
        df = pd.DataFrame({
            'constant_col': [100.0, 100.0, 100.0, 100.0],
            'loan_amount': [1000.0, 2000.0, 3000.0, 4000.0],
            'default': [0, 1, 0, 1]
        })
        prep = build_preprocessor(
            fico_col=None,
            cont_cols=['constant_col', 'loan_amount'],
            cat_cols=[]
        )
        prep.fit(df.drop(columns=['default']), df['default'])
        X_trans = prep.transform(df.drop(columns=['default']))
        assert not np.isnan(X_trans).any()
        np.testing.assert_allclose(X_trans[:, 0], 0.0)

    def test_categorical_most_frequent_imputation(self):
        """Categorical pipeline must impute missing values using most frequent value (mode)."""
        df_train = pd.DataFrame({
            'purpose': ['debt_consolidation', 'debt_consolidation', 'credit_card', 'home_improvement'],
            'default': [0, 0, 1, 1]
        })
        prep = build_preprocessor(
            fico_col=None,
            cont_cols=[],
            cat_cols=['purpose']
        )
        prep.fit(df_train.drop(columns=['default']), df_train['default'])

        # Transform with NaN in purpose
        df_test = pd.DataFrame({'purpose': [np.nan]})
        out = prep.transform(df_test)
        assert out.shape == (1, 3)  # 3 categories
        # The mode ('debt_consolidation') should be encoded as active (1.0)
        mode_idx = list(prep.named_transformers_['cat'].named_steps['ohe'].categories_[0]).index('debt_consolidation')
        assert out[0, mode_idx] == 1.0

    def test_categorical_binary_drop_if_binary(self, binary_categorical_df):
        """Binary categorical column must be encoded as a single column when drop='if_binary'."""
        df = binary_categorical_df
        prep = build_preprocessor(
            fico_col=None,
            cont_cols=['income', 'loan_amount'],
            cat_cols=['is_employed']
        )
        prep.fit(df.drop(columns=['default']), df['default'])
        out = prep.transform(df.drop(columns=['default']))
        # 2 continuous + 1 binary categorical = 3 columns
        assert out.shape == (len(df), 3)

    def test_categorical_multiclass_one_hot_encoding(self):
        """Multiclass categorical feature with K categories produces K columns."""
        df = pd.DataFrame({
            'tier': ['A', 'B', 'C', 'D'],
            'default': [0, 0, 1, 1]
        })
        prep = build_preprocessor(
            fico_col=None,
            cont_cols=[],
            cat_cols=['tier']
        )
        prep.fit(df.drop(columns=['default']), df['default'])
        out = prep.transform(df.drop(columns=['default']))
        assert out.shape == (4, 4)

    def test_unseen_categories_handled_gracefully(self):
        """Unseen categories during inference are ignored (all zeros) without error via handle_unknown='ignore'."""
        df_train = pd.DataFrame({
            'purpose': ['debt_consolidation', 'credit_card'],
            'default': [0, 1]
        })
        prep = build_preprocessor(
            fico_col=None,
            cont_cols=[],
            cat_cols=['purpose']
        )
        prep.fit(df_train.drop(columns=['default']), df_train['default'])

        # New, unseen category string
        df_test = pd.DataFrame({'purpose': ['space_travel']})
        out = prep.transform(df_test)
        assert out.shape == (1, 1)  # Binary feature dropped 1 category -> 1 column
        assert out[0, 0] == 0.0, "Unseen category should produce all zeros"

    def test_remainder_drop_discards_unmodeled_columns(self):
        """remainder='drop' must discard extraneous columns (IDs, text, timestamps)."""
        df = pd.DataFrame({
            'FICO_score': [650, 700, 750, 600],
            'income': [50000, 60000, 70000, 40000],
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003', 'CUST_004'],
            'ssn': ['000-11-2222', '000-11-3333', '000-11-4444', '000-11-5555'],
            'timestamp': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04'],
            'default': [0, 0, 0, 1]
        })
        prep = build_preprocessor(
            max_fico_buckets=2,
            fico_col='FICO_score',
            cont_cols=['income'],
            cat_cols=[]
        )
        prep.fit(df.drop(columns=['default']), df['default'])
        out = prep.transform(df.drop(columns=['default']))
        
        # Output should contain only FICO one-hot columns (2) + income (1) = 3 columns
        assert out.shape == (4, 3)

    def test_preprocessor_with_nan_in_fico_during_transform(self):
        """NaN in FICO score at transform time maps to all-zero one-hot vector via handle_unknown='ignore'."""
        df_train = pd.DataFrame({
            'FICO_score': [600, 650, 700, 750],
            'income': [50000, 60000, 70000, 80000],
            'default': [1, 1, 0, 0]
        })
        prep = build_preprocessor(
            max_fico_buckets=2,
            fico_col='FICO_score',
            cont_cols=['income'],
            cat_cols=[]
        )
        prep.fit(df_train.drop(columns=['default']), df_train['default'])

        df_test = pd.DataFrame({
            'FICO_score': [np.nan],
            'income': [55000]
        })
        out = prep.transform(df_test)
        assert out.shape == (1, 3)
        # First 2 cols are FICO one-hot (should be all zeros for NaN), 3rd is scaled income
        np.testing.assert_array_equal(out[0, :2], [0.0, 0.0])
        assert not np.isnan(out[0, 2])
