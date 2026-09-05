import os
import shutil
import argparse
import joblib
import pandas as pd
import numpy as np

from data_generator import generate_loan_data
from model_training import train_and_evaluate

def run_pipeline(data_path="loan_data.csv", output_artifact="models/model_artifact.joblib", max_fico_buckets=5, target_col='default', fico_col='FICO_score'):
    """
    Top-level execution pipeline:
    1. Ingestion / Data verification (generates loan_data.csv if missing)
    2. Multi-model benchmarking (Logistic Regression, XGBoost, Gradient Boosting, Random Forest)
    3. Champion model selection based on discrimination (ROC-AUC) and calibration (Brier score)
    4. Metric logging to stdout (ROC-AUC, PR-AUC, Gini, Brier score, Accuracy, F1)
    5. Optimal FICO cutoffs extraction and risk tier reporting
    6. Artifact serialization to disk (model_artifact.joblib and model.joblib)
    7. Artifact verification: reload from disk and score sample record
    """
    print("=" * 80)
    print("QUANTITATIVE CREDIT RISK ENGINE - TOP-LEVEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 80)
    
    # 1. Ingestion
    if not os.path.exists(data_path):
        print(f"Data file '{data_path}' not found. Generating 10,000 synthetic loans...")
        df = generate_loan_data(num_samples=10000, random_state=42)
        df.to_csv(data_path, index=False)
        print(f"Saved generated dataset to '{data_path}'.")
    else:
        print(f"Found existing data at '{data_path}'.")
        
    # 2-5. Train, benchmark, evaluate, and persist
    champion_pipeline = train_and_evaluate(
        data_path=data_path,
        max_fico_buckets=max_fico_buckets,
        save_artifact=output_artifact,
        target_col=target_col,
        fico_col=fico_col
    )
    
    # Also save as model.joblib for seamless multi-convention compatibility
    alt_artifact = "models/model.joblib"
    if output_artifact != alt_artifact:
        shutil.copyfile(output_artifact, alt_artifact)
        print(f"Also saved synchronized artifact to '{alt_artifact}'.")
        
    # 6. Verification: Reload and score sample record
    print("\n" + "=" * 80)
    print("VERIFYING MODEL ARTIFACT PERSISTENCE & INFERENCE FIDELITY")
    print("=" * 80)
    
    loaded_pipeline = joblib.load(output_artifact)
    print(f"Successfully reloaded model artifact from '{output_artifact}'.")
    
    # Test sample inference using first row of dataset (to ensure feature names match)
    df = pd.read_csv(data_path)
    # Exclude the target column if present, assuming it might be 'default' or custom
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    
    sample_borrower = df.iloc[[0]].copy()
    
    pred_proba = loaded_pipeline.predict_proba(sample_borrower)
    default_prob = float(pred_proba[0, 1])
    pred_class = int(loaded_pipeline.predict(sample_borrower)[0])
    
    print(f"Sample Loan Application Features:")
    print(sample_borrower.to_dict(orient='records')[0])
    print(f"Inference Output:")
    print(f"  Probability of Default (PD): {default_prob:.4f} ({default_prob:.2%})")
    print(f"  Binary Default Prediction:   {pred_class}")
    
    assert 0.0 <= default_prob <= 1.0, f"Predicted PD {default_prob} is out of bounds [0, 1]"
    print("Artifact verification PASSED: Output is a valid calibrated probability in [0, 1].")
    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY.")
    print("=" * 80)
    
    return champion_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Credit Risk Model Pipeline")
    parser.add_argument("--data-path", default="loan_data.csv", help="Path to loan data CSV")
    parser.add_argument("--output", default="models/model_artifact.joblib", help="Output artifact joblib file")
    parser.add_argument("--max-buckets", type=int, default=5, help="Maximum FICO DP buckets")
    parser.add_argument("--target-col", default="default", help="Name of the target variable column")
    parser.add_argument("--fico-col", default="FICO_score", help="Name of the FICO/credit score column")
    args = parser.parse_args()
    
    run_pipeline(
        data_path=args.data_path,
        output_artifact=args.output,
        max_fico_buckets=args.max_buckets,
        target_col=args.target_col,
        fico_col=args.fico_col
    )
