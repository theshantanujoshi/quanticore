import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    classification_report
)
from preprocessing import build_preprocessor
from data_generator import generate_loan_data

def evaluate_model(pipeline, X_test, y_test):
    """
    Computes comprehensive classification, calibration, and discrimination metrics.
    """
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)
    
    auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    gini = 2.0 * auc - 1.0
    
    return {
        'roc_auc': auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'accuracy': acc,
        'f1_score': f1,
        'gini': gini
    }

def benchmark_models(X_train, y_train, X_test, y_test, max_fico_buckets=5, fico_col='FICO_score', cont_cols=None, cat_cols=None):
    """
    Evaluates candidate models across identical training and testing splits.
    Compares Logistic Regression, XGBoost, Gradient Boosting, and Random Forest.
    """
    candidate_classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }
    
    # Check for XGBoost availability
    try:
        from xgboost import XGBClassifier
        candidate_classifiers['XGBoost'] = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
    except ImportError:
        pass
        
    benchmark_results = {}
    pipelines = {}
    
    print("\n" + "=" * 80)
    print("MULTI-MODEL BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"{'Model Family':22s} | {'ROC-AUC':8s} | {'PR-AUC':8s} | {'Brier':8s} | {'Accuracy':8s} | {'F1':8s}")
    print("-" * 80)
    
    for name, clf in candidate_classifiers.items():
        pipe = Pipeline([
            ('preprocessor', build_preprocessor(
                max_fico_buckets=max_fico_buckets,
                fico_col=fico_col,
                cont_cols=cont_cols,
                cat_cols=cat_cols
            )),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)
        
        benchmark_results[name] = metrics
        pipelines[name] = pipe
        
        print(f"{name:22s} | {metrics['roc_auc']:.4f}   | {metrics['pr_auc']:.4f}   | {metrics['brier_score']:.4f}  | {metrics['accuracy']:.4f}   | {metrics['f1_score']:.4f}")
        
    print("-" * 80)
    return benchmark_results, pipelines

def train_and_evaluate(data_path="loan_data.csv", max_fico_buckets=5, save_artifact=None, target_col='default', fico_col='FICO_score'):
    """
    End-to-end model training, multi-model benchmark comparison, champion selection,
    and optional artifact serialization.
    """
    if not os.path.exists(data_path):
        print(f"Data file '{data_path}' not found. Generating 10,000 synthetic loan records...")
        df = generate_loan_data(10000, random_state=42)
        df.to_csv(data_path, index=False)
    else:
        print(f"Loading loan data from '{data_path}'...")
        df = pd.read_csv(data_path)
        
    print(f"Loaded {len(df)} records. Empirical portfolio default rate: {df[target_col].mean():.2%}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Dynamically detect columns for real-world datasets
    features = list(X.columns)
    if fico_col in features:
        features.remove(fico_col)
    else:
        fico_col = None # Disable FICO bucketing if not found
        
    num_cols = X[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Filter high-cardinality categorical columns to prevent One-Hot Encoding memory explosions
    all_cat_cols = X[features].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    cat_cols = []
    for col in all_cat_cols:
        if X[col].nunique() < 20:
            cat_cols.append(col)
        else:
            print(f"Dropping high-cardinality categorical column: {col} ({X[col].nunique()} unique values)")
            
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # Benchmark multiple models
    benchmark_results, pipelines = benchmark_models(
        X_train, y_train, X_test, y_test, 
        max_fico_buckets=max_fico_buckets,
        fico_col=fico_col,
        cont_cols=num_cols,
        cat_cols=cat_cols
    )
    
    # Champion selection based on ROC-AUC (and lower Brier score)
    champion_name = max(
        benchmark_results.keys(),
        key=lambda k: (benchmark_results[k]['roc_auc'], -benchmark_results[k]['brier_score'])
    )
    champion_pipeline = pipelines[champion_name]
    champ_metrics = benchmark_results[champion_name]
    
    print(f"\n[CHAMPION] MODEL SELECTED: {champion_name}")
    print(f"  ROC-AUC:     {champ_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:      {champ_metrics['pr_auc']:.4f}")
    print(f"  Gini Index:  {champ_metrics['gini']:.4f}")
    print(f"  Brier Score: {champ_metrics['brier_score']:.4f} (Calibrated default probability)")
    print(f"  Accuracy:    {champ_metrics['accuracy']:.4f}")
    print(f"  F1 Score:    {champ_metrics['f1_score']:.4f}")
    
    # Extract optimal FICO boundaries from champion pipeline if FICO column exists
    if fico_col:
        bucketer = champion_pipeline.named_steps['preprocessor'].named_transformers_['fico'].named_steps['bucketer']
        print(f"\nOptimal FICO Cutoffs: {bucketer.boundaries_}")
        if hasattr(bucketer, 'bucket_summary_'):
            print("\nDiscovered FICO Risk Tiers:")
            for tier in bucketer.bucket_summary_:
                print(f"  Tier {tier['bucket']}: {tier['range']:20s} | Count: {tier['count']:5d} | Defaults: {tier['defaults']:4d} | Default Rate: {tier['default_rate']:.2%}")
                
        fico_boundaries = bucketer.boundaries_
        bucket_means = bucketer.bucket_means_
    else:
        fico_boundaries = None
        bucket_means = None
            
    # Classification Report
    y_test_preds = champion_pipeline.predict(X_test)
    print("\nChampion Classification Report (Threshold = 0.50):")
    print(classification_report(y_test, y_test_preds, target_names=['Non-Default (0)', 'Default (1)']))
    
    # Persist artifact if requested
    if save_artifact:
        print(f"Serializing champion model artifact to '{save_artifact}'...")
        joblib.dump(champion_pipeline, save_artifact)
        metadata = {
            'champion_model': champion_name,
            'metrics': champ_metrics,
            'fico_boundaries': fico_boundaries,
            'bucket_means': bucket_means,
            'features': list(X.columns)
        }
        meta_file = save_artifact.replace('.joblib', '_metadata.joblib')
        joblib.dump(metadata, meta_file)
        print(f"Artifact persisted successfully to '{save_artifact}' and metadata to '{meta_file}'.")
        
    return champion_pipeline

if __name__ == "__main__":
    train_and_evaluate("loan_data.csv", max_fico_buckets=5, save_artifact="model_artifact.joblib")
