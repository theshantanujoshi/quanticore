from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from fico_bucketing import FicoDPBucketer

def build_preprocessor(max_fico_buckets=5, fico_col='FICO_score', cont_cols=None, cat_cols=None):
    """
    Builds a scikit-learn ColumnTransformer that applies:
    - DP Bucketing and OneHotEncoding to the FICO score feature
    - Median Imputation and Standard Scaling to continuous features
    - Most-Frequent Imputation and OneHotEncoding to categorical features
    - Drops any unmodeled metadata/ID columns (remainder='drop')
    """
    if cont_cols is None:
        cont_cols = ['income', 'loan_amount', 'dti', 'employment_length']
    if cat_cols is None:
        cat_cols = []
        
    transformers = []
    
    # FICO pipeline: DP optimal discretization + one-hot encoding
    if fico_col is not None:
        fico_pipeline = Pipeline([
            ('bucketer', FicoDPBucketer(max_buckets=max_fico_buckets)),
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        transformers.append(('fico', fico_pipeline, [fico_col]))
    
    # Continuous features pipeline: median imputation + standard scaling
    if cont_cols:
        continuous_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('cont', continuous_pipeline, cont_cols))
        
    # Categorical features pipeline: mode imputation + one-hot encoding
    if cat_cols:
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary'))
        ])
        transformers.append(('cat', categorical_pipeline, cat_cols))
    
    # Combine into ColumnTransformer with strict remainder='drop'
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    return preprocessor
