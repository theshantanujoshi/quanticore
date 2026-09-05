import os
import sys
import itertools
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Set up paths
site_pkg = r"D:\Lib\site-packages"
proj_dir = r"D:\quantproj"
if site_pkg not in sys.path:
    sys.path.insert(0, site_pkg)
if proj_dir not in sys.path:
    sys.path.insert(0, proj_dir)

from fico_bucketing import FicoDPBucketer
from preprocessing import build_preprocessor


# =====================================================================
# 1. ORACLE: Brute-force optimal partitioner for verifying DP optimality
# =====================================================================
def brute_force_optimal_cost_and_boundaries(ficos, defaults, k_target):
    """
    Exhaustively searches all C(n-1, k_target-1) valid contiguous partitions
    of sorted unique FICO scores to find the exact global minimum SSE cost.
    """
    df = pd.DataFrame({'FICO': ficos, 'default': defaults})
    grouped = df.groupby('FICO', observed=False).agg(
        total=('default', 'count'),
        defaults=('default', 'sum')
    ).reset_index().sort_values('FICO')
    
    unique_ficos = grouped['FICO'].values
    totals = grouped['total'].values
    defs = grouped['defaults'].values
    n = len(unique_ficos)
    
    if n <= k_target:
        k_target = min(k_target, n)
    if k_target <= 1:
        tot = np.sum(totals)
        d = np.sum(defs)
        cost = float(d - (d**2 / tot)) if tot > 0 else 0.0
        return cost, [-np.inf, np.inf]
        
    cum_total = np.zeros(n + 1, dtype=np.int64)
    cum_defs = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        cum_total[i+1] = cum_total[i] + totals[i]
        cum_defs[i+1] = cum_defs[i] + defs[i]
        
    def seg_cost(m, i):
        tot = cum_total[i] - cum_total[m]
        d = cum_defs[i] - cum_defs[m]
        if tot == 0:
            return 0.0
        return float(d - (d ** 2 / tot))
        
    best_cost = np.inf
    best_boundaries = None
    
    # Choose k_target - 1 split points among range(1, n)
    for splits in itertools.combinations(range(1, n), k_target - 1):
        idx_splits = [0] + list(splits) + [n]
        total_cost = 0.0
        for b in range(len(idx_splits) - 1):
            total_cost += seg_cost(idx_splits[b], idx_splits[b+1])
            
        if total_cost < best_cost - 1e-12:
            best_cost = total_cost
            boundaries = [-np.inf] + [float(unique_ficos[s]) for s in splits] + [np.inf]
            best_boundaries = boundaries
            
    return best_cost, best_boundaries


# =====================================================================
# 2. EDGE CASES: Targets (All defaults, all non-defaults, rare events)
# =====================================================================
class TestTargetEdgeCases:
    
    def test_all_defaults_y1(self):
        """All loans defaulted (y=1). Within-bucket variance is zero everywhere."""
        ficos = np.array([350, 420, 510, 600, 720, 800])
        defaults = np.ones_like(ficos)
        
        bucketer = FicoDPBucketer(max_buckets=3)
        bucketer.fit(ficos, defaults)
        
        assert len(bucketer.boundaries_) == 4
        assert bucketer.boundaries_[0] == -np.inf
        assert bucketer.boundaries_[-1] == np.inf
        
        # All bucket means should be 1.0
        for b_idx, mean_rate in bucketer.bucket_means_.items():
            assert mean_rate == 1.0, f"Bucket {b_idx} expected mean 1.0, got {mean_rate}"
            
        # Transform should produce valid integer buckets
        out = bucketer.transform(ficos)
        assert out.shape == (6, 1)
        assert not np.isnan(out).any()
        assert set(out.ravel()).issubset({0, 1, 2})

    def test_all_non_defaults_y0(self):
        """No loans defaulted (y=0). Within-bucket variance is zero everywhere."""
        ficos = np.array([350, 420, 510, 600, 720, 800])
        defaults = np.zeros_like(ficos)
        
        bucketer = FicoDPBucketer(max_buckets=3)
        bucketer.fit(ficos, defaults)
        
        assert len(bucketer.boundaries_) == 4
        for b_idx, mean_rate in bucketer.bucket_means_.items():
            assert mean_rate == 0.0, f"Bucket {b_idx} expected mean 0.0, got {mean_rate}"
            
        out = bucketer.transform(ficos)
        assert out.shape == (6, 1)
        assert not np.isnan(out).any()
        assert set(out.ravel()).issubset({0, 1, 2})

    def test_single_default_among_many_non_defaults(self):
        """Extreme class imbalance: 1 default, 99 non-defaults."""
        np.random.seed(42)
        ficos = np.linspace(300, 850, 100).astype(int)
        defaults = np.zeros(100, dtype=int)
        defaults[5] = 1  # FICO score around 327 defaults
        
        bucketer = FicoDPBucketer(max_buckets=3)
        bucketer.fit(ficos, defaults)
        
        assert len(bucketer.boundaries_) == 4
        out = bucketer.transform(ficos)
        assert out.shape == (100, 1)
        assert not np.isnan(out).any()
        # The default must be captured in exactly one bucket
        summary_defs = sum(s['defaults'] for s in bucketer.bucket_summary_)
        assert summary_defs == 1


# =====================================================================
# 3. EDGE CASES: FICO Scores (Identical, Negatives, Extreme ranges)
# =====================================================================
class TestFicoScoreEdgeCases:
    
    def test_identical_ficos_same_score(self):
        """All observations have the exact same FICO score."""
        ficos = np.array([620, 620, 620, 620, 620])
        defaults = np.array([1, 0, 1, 0, 0])
        
        bucketer = FicoDPBucketer(max_buckets=4)
        bucketer.fit(ficos, defaults)
        
        assert bucketer.boundaries_ == [-np.inf, np.inf]
        assert len(bucketer.bucket_means_) == 1
        assert pytest.approx(bucketer.bucket_means_[0], 1e-4) == 0.4
        
        # Test transform on same and novel scores
        out = bucketer.transform([620, 500, 800])
        assert out.shape == (3, 1)
        assert (out == 0).all()

    def test_negative_and_zero_fico_scores(self):
        """Unconventional/synthetic features where scores are negative or zero."""
        ficos = np.array([-300, -150, 0, 150, 300])
        defaults = np.array([1, 1, 1, 0, 0])
        
        bucketer = FicoDPBucketer(max_buckets=2)
        bucketer.fit(ficos, defaults)
        
        assert len(bucketer.boundaries_) == 3
        # Boundary should isolate the high-risk negative/zero scores
        assert bucketer.boundaries_[1] in [-150.0, 0.0, 150.0]
        
        out = bucketer.transform([-500, -200, 0, 200, 500])
        assert out.shape == (5, 1)
        assert not np.isnan(out).any()

    def test_extreme_fico_values_300_850_1000(self):
        """Tests standard US credit spectrum [300, 850] plus out-of-range 1000."""
        ficos = np.array([300, 450, 600, 750, 850, 1000])
        defaults = np.array([1, 1, 1, 0, 0, 0])
        
        bucketer = FicoDPBucketer(max_buckets=3)
        bucketer.fit(ficos, defaults)
        
        assert len(bucketer.boundaries_) == 4
        # Transform extreme inputs: very low (-100), standard (500), very high (1500)
        test_pts = np.array([-100, 300, 850, 1000, 1500])
        out = bucketer.transform(test_pts)
        assert out.shape == (5, 1)
        assert not np.isnan(out).any()

    def test_duplicate_ficos_with_mixed_defaults(self):
        """Multiple observations per distinct FICO score."""
        ficos = np.array([500, 500, 500, 600, 600, 700, 700, 700])
        defaults = np.array([1, 1, 0, 1, 0, 0, 0, 0])
        
        bucketer = FicoDPBucketer(max_buckets=3)
        bucketer.fit(ficos, defaults)
        
        assert len(bucketer.boundaries_) == 4
        assert bucketer.boundaries_ == [-np.inf, 600.0, 700.0, np.inf]
        assert bucketer.bucket_means_[0] > bucketer.bucket_means_[2]


# =====================================================================
# 4. SMALL DATASETS & BOUNDARY CONDITIONS (n < max_buckets)
# =====================================================================
class TestSmallDatasetBoundaryConditions:
    
    def test_n_equals_1_single_observation(self):
        """Single borrower record."""
        ficos = np.array([650])
        defaults = np.array([0])
        
        bucketer = FicoDPBucketer(max_buckets=5)
        bucketer.fit(ficos, defaults)
        
        assert bucketer.boundaries_ == [-np.inf, np.inf]
        assert len(bucketer.bucket_means_) == 1
        assert bucketer.bucket_means_[0] == 0.0
        
        out = bucketer.transform([650, 700])
        assert out.shape == (2, 1)
        assert (out == 0).all()

    def test_n_equals_2_with_max_buckets_5(self):
        """2 unique observations with max_buckets=5: should yield exactly 2 buckets."""
        ficos = np.array([550, 750])
        defaults = np.array([1, 0])
        
        bucketer = FicoDPBucketer(max_buckets=5)
        bucketer.fit(ficos, defaults)
        
        assert len(bucketer.boundaries_) == 3  # [-inf, 750.0, inf]
        assert bucketer.boundaries_ == [-np.inf, 750.0, np.inf]
        assert bucketer.bucket_means_[0] == 1.0
        assert bucketer.bucket_means_[1] == 0.0

    def test_n_equals_5_with_max_buckets_5(self):
        """n == max_buckets: each unique score should occupy its own bucket."""
        ficos = np.array([500, 600, 650, 700, 800])
        defaults = np.array([1, 1, 0, 0, 0])
        
        bucketer = FicoDPBucketer(max_buckets=5)
        bucketer.fit(ficos, defaults)
        
        # 5 buckets requires 6 boundary markers: [-inf, 600, 650, 700, 800, inf]
        assert len(bucketer.boundaries_) == 6
        assert bucketer.boundaries_ == [-np.inf, 600.0, 650.0, 700.0, 800.0, np.inf]

    def test_empty_dataset(self):
        """Empty input arrays."""
        bucketer = FicoDPBucketer(max_buckets=5)
        bucketer.fit([], [])
        
        assert bucketer.boundaries_ == [-np.inf, np.inf]
        assert bucketer.bucket_means_ == {0: 0.0}
        out = bucketer.transform([])
        assert out.shape == (0, 1)


# =====================================================================
# 5. INPUT CONTAINERS, SHAPES, AND MEMORY LAYOUTS
# =====================================================================
class TestContainerTypesAndMemoryLayouts:
    
    @pytest.fixture
    def sample_data(self):
        ficos = [520, 580, 640, 710, 790, 820]
        defaults = [1, 1, 1, 0, 0, 0]
        return ficos, defaults

    def test_1d_vs_2d_numpy_arrays(self, sample_data):
        ficos, defaults = sample_data
        
        b1 = FicoDPBucketer(max_buckets=3).fit(np.array(ficos), defaults)
        b2 = FicoDPBucketer(max_buckets=3).fit(np.array(ficos).reshape(-1, 1), defaults)
        
        assert b1.boundaries_ == b2.boundaries_
        assert b1.bucket_means_ == b2.bucket_means_
        
        # Test transforms
        t1 = b1.transform(np.array([550, 750]))
        t2 = b2.transform(np.array([[550], [750]]))
        assert np.array_equal(t1, t2)

    def test_pandas_series_and_dataframes(self, sample_data):
        ficos, defaults = sample_data
        
        s_ficos = pd.Series(ficos, index=[10, 20, 30, 40, 50, 60])
        s_defaults = pd.Series(defaults, index=[10, 20, 30, 40, 50, 60])
        df_ficos = pd.DataFrame({'FICO_score': ficos})
        
        b_series = FicoDPBucketer(max_buckets=3).fit(s_ficos, s_defaults)
        b_df = FicoDPBucketer(max_buckets=3).fit(df_ficos, defaults)
        
        assert b_series.boundaries_ == b_df.boundaries_
        
        t_s = b_series.transform(s_ficos)
        t_df = b_df.transform(df_ficos)
        assert np.array_equal(t_s, t_df)

    def test_non_contiguous_and_strided_arrays(self, sample_data):
        """Test Fortran-order, strided slices, and non-contiguous memory layouts."""
        ficos, defaults = sample_data
        
        # Truly non-contiguous Fortran slice (2D array sliced column-wise)
        f_2d = np.asfortranarray(np.column_stack([ficos, np.zeros_like(ficos)]))
        f_arr = f_2d[:, 0]
        b_f = FicoDPBucketer(max_buckets=3).fit(f_arr, defaults)
        
        # Strided slice (every 2nd element)
        interleaved_ficos = np.array([520, -99, 580, -99, 640, -99, 710, -99, 790, -99, 820, -99])[::2]
        interleaved_defaults = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0])[::2]
        assert not interleaved_ficos.flags.contiguous
        
        b_strided = FicoDPBucketer(max_buckets=3).fit(interleaved_ficos, interleaved_defaults)
        assert b_f.boundaries_ == b_strided.boundaries_
        
        # Transform non-contiguous test array
        test_strided = np.array([550, 0, 750, 0])[::2]
        out = b_strided.transform(test_strided)
        assert out.shape == (2, 1)

    def test_lists_of_ints_and_floats(self):
        """Python native lists of ints and floats."""
        b_int = FicoDPBucketer(max_buckets=2).fit([500, 600, 700], [1, 1, 0])
        b_float = FicoDPBucketer(max_buckets=2).fit([500.0, 600.0, 700.0], [1, 1, 0])
        assert b_int.boundaries_ == b_float.boundaries_


# =====================================================================
# 6. MATHEMATICAL OPTIMALITY: DP vs BRUTE-FORCE ORACLE
# =====================================================================
class TestMathematicalOptimalityAgainstOracle:
    
    @pytest.mark.parametrize("seed, n_samples, max_k", [
        (101, 15, 2),
        (202, 20, 3),
        (303, 25, 4),
        (404, 30, 4),
        (505, 35, 5),
    ])
    def test_dp_exact_optimality_matches_brute_force(self, seed, n_samples, max_k):
        """
        Rigorously compares DP algorithm cost against an exhaustive brute-force oracle.
        Both must achieve the exact same global minimum SSE cost.
        """
        np.random.seed(seed)
        ficos = np.random.choice(np.arange(400, 800, 10), size=n_samples, replace=True)
        # Prob defaults decreases with FICO
        prob_default = 1.0 / (1.0 + np.exp((ficos - 600) / 50.0))
        defaults = np.random.binomial(1, prob_default)
        
        # 1. Compute via DP Bucketer
        bucketer = FicoDPBucketer(max_buckets=max_k)
        bucketer.fit(ficos, defaults)
        
        # Calculate empirical DP total cost (sum of SSW across assigned buckets)
        assigned = pd.cut(ficos, bins=bucketer.boundaries_, labels=False, right=False)
        df = pd.DataFrame({'bucket': assigned, 'default': defaults})
        dp_total_cost = 0.0
        for b_idx, group in df.groupby('bucket', observed=False):
            n_b = len(group)
            d_b = group['default'].sum()
            dp_total_cost += float(d_b - (d_b ** 2 / n_b))
            
        # 2. Compute via Brute Force Oracle
        oracle_cost, oracle_boundaries = brute_force_optimal_cost_and_boundaries(
            ficos, defaults, min(max_k, len(np.unique(ficos)))
        )
        
        # Verify that DP cost matches global minimum within floating point precision
        assert pytest.approx(dp_total_cost, abs=1e-8) == oracle_cost, (
            f"DP cost {dp_total_cost} did not match brute force minimum {oracle_cost}"
        )


# =====================================================================
# 7. MONOTONICITY OF DISCOVERED DEFAULT RATES
# =====================================================================
class TestMonotonicityProperties:
    
    def test_strictly_monotonic_under_standard_risk_gradient(self):
        """
        Under realistic monotonic credit risk distributions (higher FICO -> lower default),
        discovered bucket default rates must be strictly non-increasing.
        """
        np.random.seed(42)
        n_samples = 3000
        ficos = np.random.randint(350, 850, size=n_samples)
        # Logit risk gradient
        logits = (620.0 - ficos) / 60.0
        p_defaults = 1.0 / (1.0 + np.exp(-logits))
        defaults = np.random.binomial(1, p_defaults)
        
        bucketer = FicoDPBucketer(max_buckets=5)
        bucketer.fit(ficos, defaults)
        
        rates = [bucketer.bucket_means_[k] for k in range(len(bucketer.bucket_means_))]
        
        # Check that default rates decrease monotonically as FICO increases
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i+1], (
                f"Monotonicity violated between bucket {i} (rate {rates[i]:.4f}) "
                f"and bucket {i+1} (rate {rates[i+1]:.4f})"
            )

    def test_behavior_under_inverted_risk_distribution(self):
        """
        When data is deliberately inverted (higher FICO -> higher default),
        the bucketer still minimizes SSE and discovers monotonically INCREASING rates.
        """
        np.random.seed(99)
        ficos = np.random.randint(400, 800, size=2000)
        p_defaults = (ficos - 400.0) / 400.0  # higher FICO = higher default
        defaults = np.random.binomial(1, p_defaults)
        
        bucketer = FicoDPBucketer(max_buckets=4)
        bucketer.fit(ficos, defaults)
        
        rates = [bucketer.bucket_means_[k] for k in range(len(bucketer.bucket_means_))]
        for i in range(len(rates) - 1):
            assert rates[i] <= rates[i+1], (
                f"Expected increasing rates, but bucket {i} ({rates[i]:.4f}) > bucket {i+1} ({rates[i+1]:.4f})"
            )


# =====================================================================
# 8. PIPELINE INTEGRATION: ColumnTransformer, OneHotEncoder, Out-of-bounds
# =====================================================================
class TestPipelineIntegration:
    
    def test_columntransformer_and_onehotencoder_pipeline(self):
        """Tests full preprocessing pipeline integration with downstream Logistic Regression."""
        df = pd.DataFrame({
            'FICO_score': [450, 520, 610, 680, 740, 800, 820, 590, 630, 710],
            'income': [30000, 40000, 55000, 70000, 95000, 110000, 130000, 48000, 62000, 85000],
            'loan_amount': [15000, 12000, 10000, 15000, 20000, 25000, 18000, 14000, 11000, 16000],
            'dti': [0.45, 0.40, 0.30, 0.25, 0.18, 0.12, 0.10, 0.35, 0.28, 0.20],
            'employment_length': [1, 2, 4, 6, 8, 10, 12, 3, 5, 7],
            'default': [1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        })
        
        X = df.drop(columns=['default'])
        y = df['default']
        
        preprocessor = build_preprocessor(max_fico_buckets=3)
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        pipe.fit(X, y)
        
        # Test inference with out-of-distribution FICO scores
        out_of_bounds_test = pd.DataFrame({
            'FICO_score': [250, 950],  # 250 < min(450), 950 > max(820)
            'income': [25000, 150000],
            'loan_amount': [10000, 30000],
            'dti': [0.50, 0.08],
            'employment_length': [0, 15]
        })
        
        preds_proba = pipe.predict_proba(out_of_bounds_test)
        assert preds_proba.shape == (2, 2)
        assert not np.isnan(preds_proba).any()
        # Extreme low FICO (250) must have higher PD than extreme high FICO (950)
        assert preds_proba[0, 1] > preds_proba[1, 1], (
            f"Low FICO PD ({preds_proba[0, 1]}) should exceed High FICO PD ({preds_proba[1, 1]})"
        )

    def test_pipeline_nan_and_inf_handling(self):
        """Pipeline handles NaN, +Inf, -Inf during inference gracefully via handle_unknown='ignore'."""
        df = pd.DataFrame({
            'FICO_score': [500, 600, 700, 800],
            'income': [40000, 50000, 60000, 70000],
            'loan_amount': [10000, 10000, 10000, 10000],
            'dti': [0.3, 0.3, 0.2, 0.2],
            'employment_length': [2, 3, 4, 5],
            'default': [1, 1, 0, 0]
        })
        preprocessor = build_preprocessor(max_fico_buckets=2)
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])
        pipe.fit(df.drop(columns=['default']), df['default'])

        # Inference on NaN and Inf
        inf_sample = pd.DataFrame({
            'FICO_score': [np.nan, np.inf, -np.inf, 1e10],
            'income': [50000, 50000, 50000, 50000],
            'loan_amount': [10000, 10000, 10000, 10000],
            'dti': [0.25, 0.25, 0.25, 0.25],
            'employment_length': [3, 3, 3, 3]
        })
        probs = pipe.predict_proba(inf_sample)
        assert probs.shape == (4, 2)
        assert not np.isnan(probs).any()
        assert np.all((probs >= 0.0) & (probs <= 1.0))


# =====================================================================
# 9. ADVERSARIAL INVARIANCE & ROBUSTNESS TESTS
# =====================================================================
class TestAdversarialInvarianceAndRobustness:

    def test_max_buckets_1_with_multiple_samples(self):
        """max_buckets=1 creates exactly 1 global bucket [-inf, inf]."""
        ficos = np.array([450, 550, 650, 750])
        defaults = np.array([1, 1, 0, 0])
        b = FicoDPBucketer(max_buckets=1).fit(ficos, defaults)
        assert b.boundaries_ == [-np.inf, np.inf]
        assert len(b.bucket_means_) == 1
        assert b.bucket_means_[0] == 0.5
        out = b.transform([400, 600, 800])
        assert out.shape == (3, 1)
        assert (out == 0).all()

    def test_shuffled_inputs_order_invariance(self):
        """DP bucketer boundaries are invariant to input permutation/shuffling."""
        ficos = np.array([400, 500, 600, 700, 800])
        defaults = np.array([1, 1, 0, 0, 0])
        
        b_sorted = FicoDPBucketer(max_buckets=3).fit(ficos, defaults)
        
        # Shuffle inputs
        perm = [3, 0, 4, 1, 2]
        b_shuffled = FicoDPBucketer(max_buckets=3).fit(ficos[perm], defaults[perm])
        
        assert b_sorted.boundaries_ == b_shuffled.boundaries_
        assert b_sorted.bucket_means_ == b_shuffled.bucket_means_

    def test_target_polymorphism_bool_and_2d(self):
        """y can be boolean array or 2D column vector."""
        ficos = [500, 600, 700]
        y_bool = [True, True, False]
        y_2d = np.array([1, 1, 0]).reshape(-1, 1)
        
        b_bool = FicoDPBucketer(max_buckets=2).fit(ficos, y_bool)
        b_2d = FicoDPBucketer(max_buckets=2).fit(ficos, y_2d)
        
        assert b_bool.boundaries_ == b_2d.boundaries_
        assert b_bool.bucket_means_ == b_2d.bucket_means_

    def test_incompatible_feature_dimension_fit_raises(self):
        """Passing multi-column feature array to fit() raises ValueError."""
        X_multi = np.array([[500, 1000], [600, 2000], [700, 3000]])
        y = [1, 0, 0]
        with pytest.raises(ValueError):
            FicoDPBucketer(max_buckets=2).fit(X_multi, y)

    def test_large_scale_stress_performance(self):
        """Stress tests scalability: 50,000 samples fit and transform runs in < 2 seconds."""
        import time
        np.random.seed(42)
        ficos = np.random.randint(300, 850, 50000)
        y = np.random.binomial(1, 0.35, 50000)
        
        t0 = time.time()
        b = FicoDPBucketer(max_buckets=5).fit(ficos, y)
        t_fit = time.time() - t0
        
        t0 = time.time()
        out = b.transform(ficos[:10000])
        t_transform = time.time() - t0
        
        assert t_fit < 2.0, f"Fit took too long: {t_fit:.3f}s"
        assert t_transform < 0.5, f"Transform took too long: {t_transform:.3f}s"
        assert len(b.boundaries_) == 6

