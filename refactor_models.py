import os

replacements = {
    'train.py': [
        ('output_artifact="model_artifact.joblib"', 'output_artifact="models/model_artifact.joblib"'),
        ('alt_artifact = "model.joblib"', 'alt_artifact = "models/model.joblib"'),
        ('default="model_artifact.joblib"', 'default="models/model_artifact.joblib"')
    ],
    'incremental_train.py': [
        ("out_file = 'incremental_master_model.joblib'", "out_file = 'models/incremental_master_model.joblib'")
    ],
    'tests/test_incremental_train.py': [
        ("out_file = 'incremental_master_model.joblib'", "out_file = 'models/incremental_master_model.joblib'")
    ],
    'tests/test_model_inference.py': [
        ('"model_artifact.joblib",', '"models/model_artifact.joblib",'),
        ('"model.joblib",', '"models/model.joblib",'),
        ('"model_german.joblib",', '"models/model_german.joblib",'),
        ('"model_credit_risk.joblib",', '"models/model_credit_risk.joblib",'),
        ('"model_loan.joblib",', '"models/model_loan.joblib",'),
        ('"incremental_master_model.joblib"', '"models/incremental_master_model.joblib"'),
        ('"model_artifact_metadata.joblib",', '"models/model_artifact_metadata.joblib",'),
        ('"model_german_metadata.joblib",', '"models/model_german_metadata.joblib",'),
        ('"model_credit_risk_metadata.joblib",', '"models/model_credit_risk_metadata.joblib",'),
        ('"model_loan_metadata.joblib"', '"models/model_loan_metadata.joblib"'),
        ('if artifact_name == "model_artifact.joblib":', 'if artifact_name == "models/model_artifact.joblib":'),
        ('elif artifact_name == "model.joblib":', 'elif artifact_name == "models/model.joblib":'),
        ('elif artifact_name == "model_german.joblib":', 'elif artifact_name == "models/model_german.joblib":'),
        ('elif artifact_name == "model_credit_risk.joblib":', 'elif artifact_name == "models/model_credit_risk.joblib":'),
        ('elif artifact_name == "model_loan.joblib":', 'elif artifact_name == "models/model_loan.joblib":'),
        ('elif artifact_name == "incremental_master_model.joblib":', 'elif artifact_name == "models/incremental_master_model.joblib":'),
        ("for meta_name in ['model_artifact_metadata.joblib', 'model_credit_risk_metadata.joblib']:", "for meta_name in ['models/model_artifact_metadata.joblib', 'models/model_credit_risk_metadata.joblib']:"),
        ('model = joblib.load("model_artifact.joblib")', 'model = joblib.load("models/model_artifact.joblib")'),
        ('df = get_sample_inference_data("model_artifact.joblib", n_samples', 'df = get_sample_inference_data("models/model_artifact.joblib", n_samples'),
        ('model = joblib.load("incremental_master_model.joblib")', 'model = joblib.load("models/incremental_master_model.joblib")'),
        ('m1 = joblib.load("model_artifact.joblib")', 'm1 = joblib.load("models/model_artifact.joblib")'),
        ('m2 = joblib.load("model.joblib")', 'm2 = joblib.load("models/model.joblib")'),
        ('df1 = get_sample_inference_data("model_artifact.joblib", n_samples=2)', 'df1 = get_sample_inference_data("models/model_artifact.joblib", n_samples=2)'),
        ('df2 = get_sample_inference_data("model.joblib", n_samples=2)', 'df2 = get_sample_inference_data("models/model.joblib", n_samples=2)')
    ],
    'api.py': [
        ('os.path.join(CURRENT_DIR, "model_artifact.joblib")', 'os.path.join(CURRENT_DIR, "models", "model_artifact.joblib")'),
        ('"model_artifact.joblib",', '"models/model_artifact.joblib",'),
        ('os.path.join(CURRENT_DIR, "model.joblib")', 'os.path.join(CURRENT_DIR, "models", "model.joblib")'),
        ('"model.joblib"', '"models/model.joblib"'),
        ('os.path.join(CURRENT_DIR, "model_artifact_metadata.joblib")', 'os.path.join(CURRENT_DIR, "models", "model_artifact_metadata.joblib")'),
        ('"model_artifact_metadata.joblib",', '"models/model_artifact_metadata.joblib",'),
    ]
}

for file_path, changes in replacements.items():
    if not os.path.exists(file_path): continue
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in changes:
        content = content.replace(old, new)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated {file_path}")
