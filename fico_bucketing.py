import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class FicoDPBucketer(BaseEstimator, TransformerMixin):
    """
    Dynamic Programming FICO Bucketer.
    
    Partitions continuous FICO credit scores into K optimal contiguous buckets
    by directly minimizing the total within-bucket default variance (SSE):
        Cost = D - D^2 / N = N * p * (1 - p)
    where N is the sample count, D is defaults count, and p = D / N.
    
    Evaluates subsegment costs in O(1) time using precomputed prefix sums,
    yielding an overall time complexity of O(K * n^2) where n is the number
    of unique sorted FICO scores (n <= 551 for standard credit scores).
    """
    def __init__(self, max_buckets=5):
        self.max_buckets = max_buckets
        
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("FicoDPBucketer is a supervised transformer and requires target y.")
            
        if self.max_buckets < 1:
            raise ValueError(f"max_buckets must be >= 1, got {self.max_buckets}")
            
        # Support 1D arrays, 2D arrays (N, 1), pandas Series, and DataFrames
        X_arr = np.asarray(X)
        self.n_features_in_ = 1 if X_arr.ndim == 1 else X_arr.shape[1]
        X_val = X_arr.ravel()
        y_val = np.asarray(y).ravel()
        
        if len(X_val) != len(y_val):
            raise ValueError(f"X and y must have equal length, got len(X)={len(X_val)}, len(y)={len(y_val)}")
            
        if len(X_val) == 0:
            self.boundaries_ = [-np.inf, np.inf]
            self.bucket_means_ = {0: 0.0}
            self.bucket_summary_ = [{
                'bucket': 0,
                'range': '[-inf, inf)',
                'count': 0,
                'defaults': 0,
                'default_rate': 0.0
            }]
            return self
            
        # Group by unique FICO scores
        df = pd.DataFrame({'FICO': X_val, 'default': y_val})
        grouped = df.groupby('FICO', observed=False).agg(
            total=('default', 'count'),
            defaults=('default', 'sum')
        ).reset_index().sort_values('FICO')
        
        unique_ficos = grouped['FICO'].values
        totals = grouped['total'].values
        defaults = grouped['defaults'].values
        n = len(unique_ficos)
        
        # Handle edge case where n <= 1
        if n <= 1:
            self.boundaries_ = [-np.inf, np.inf]
            mean_rate = float(y_val.mean()) if len(y_val) > 0 else 0.0
            self.bucket_means_ = {0: mean_rate}
            self.bucket_summary_ = [{
                'bucket': 0,
                'range': '[-inf, inf)',
                'count': len(y_val),
                'defaults': int(y_val.sum()),
                'default_rate': round(mean_rate, 4)
            }]
            return self
            
        # Handle edge cases gracefully: do not collapse prematurely to a single bucket
        k_target = min(self.max_buckets, n)
        if k_target <= 1:
            self.boundaries_ = [-np.inf, np.inf]
            mean_rate = float(y_val.mean())
            self.bucket_means_ = {0: mean_rate}
            self.bucket_summary_ = [{
                'bucket': 0,
                'range': '[-inf, inf)',
                'count': len(y_val),
                'defaults': int(y_val.sum()),
                'default_rate': round(mean_rate, 4)
            }]
            return self
            
        # Precompute prefix sums for O(1) cost evaluation
        cum_total = np.zeros(n + 1, dtype=np.int64)
        cum_defaults = np.zeros(n + 1, dtype=np.int64)
        
        for i in range(n):
            cum_total[i+1] = cum_total[i] + totals[i]
            cum_defaults[i+1] = cum_defaults[i] + defaults[i]
            
        def cost(m, i):
            """Cost of bucket from unique index m to i-1: SSW = D - D^2 / N"""
            tot = cum_total[i] - cum_total[m]
            dfaults = cum_defaults[i] - cum_defaults[m]
            if tot == 0:
                return 0.0
            return float(dfaults - (dfaults ** 2 / tot))
            
        # dp[k][i] = min cost to partition first i unique elements into k buckets
        dp = np.full((k_target + 1, n + 1), np.inf)
        path = np.zeros((k_target + 1, n + 1), dtype=int)
        
        # Base case k=1
        for i in range(1, n + 1):
            dp[1][i] = cost(0, i)
            path[1][i] = 0
            
        # DP transitions
        for k in range(2, k_target + 1):
            for i in range(k, n + 1):
                min_cost = np.inf
                best_m = -1
                for m in range(k - 1, i):
                    current_cost = dp[k-1][m] + cost(m, i)
                    if current_cost < min_cost:
                        min_cost = current_cost
                        best_m = m
                dp[k][i] = min_cost
                path[k][i] = best_m
                
        # Reconstruct optimal partition boundaries
        curr_i = n
        curr_k = k_target
        indices = []
        while curr_k > 1:
            best_m = path[curr_k][curr_i]
            indices.append(best_m)
            curr_i = best_m
            curr_k -= 1
            
        indices.reverse()
        
        # Convert indices to FICO score cutoffs
        boundaries = [-np.inf]
        for idx in indices:
            boundaries.append(float(unique_ficos[idx]))
        boundaries.append(np.inf)
        self.boundaries_ = boundaries
        
        # Calculate mean default rate and empirical summary for each bucket
        assigned_buckets = pd.cut(X_val, bins=self.boundaries_, labels=False, right=False)
        bucket_df = pd.DataFrame({'bucket': assigned_buckets, 'default': y_val})
        stats = bucket_df.groupby('bucket', observed=False)['default'].agg(['count', 'sum', 'mean']).reset_index()
        
        self.bucket_means_ = {}
        self.bucket_summary_ = []
        num_buckets = len(self.boundaries_) - 1
        for b_idx in range(num_buckets):
            match = stats[stats['bucket'] == b_idx]
            if len(match) > 0:
                mean_rate = float(match['mean'].iloc[0])
                cnt = int(match['count'].iloc[0])
                defs = int(match['sum'].iloc[0])
            else:
                mean_rate = 0.0
                cnt = 0
                defs = 0
            self.bucket_means_[b_idx] = mean_rate
            self.bucket_summary_.append({
                'bucket': b_idx,
                'range': f"[{self.boundaries_[b_idx]}, {self.boundaries_[b_idx+1]})",
                'count': cnt,
                'defaults': defs,
                'default_rate': round(mean_rate, 4)
            })
            
        return self
        
    def transform(self, X):
        check_is_fitted(self, 'boundaries_')
        
        # Support 1D arrays, 2D arrays, Series, DataFrames
        X_val = np.asarray(X).ravel()
        buckets = pd.cut(X_val, bins=self.boundaries_, labels=False, right=False)
        return np.asarray(buckets).reshape(-1, 1)
        
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'boundaries_')
        return np.array(['fico_bucket'], dtype=object)

if __name__ == "__main__":
    np.random.seed(42)
    ficos = np.random.randint(300, 850, 1000)
    defaults = np.random.binomial(1, p=np.where(ficos < 600, 0.4, 0.1))
    
    bucketer = FicoDPBucketer(max_buckets=3)
    bucketer.fit(ficos, defaults)
    print("Optimal FICO Boundaries:", bucketer.boundaries_)
    print("Bucket Means:", bucketer.bucket_means_)
    for item in bucketer.bucket_summary_:
        print(" ", item)
