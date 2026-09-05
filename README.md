# Quantitative Credit Risk Engine 

Welcome to the **Quantitative Credit Risk Engine**! This project is a highly specialized, production-grade machine learning pipeline designed for modern financial underwriting. It takes raw financial profiles and transforms them into strictly calibrated probabilities of default, utilizing dynamic FICO bucketing and adversarial-tested gradient boosting.

##  Key Features

- **Dynamic FICO Bucketing:** Employs dynamic programming to group credit scores into statistically optimal risk tiers, mimicking real-world bank underwriting (Prime, Subprime, etc.).
- **Multi-Dataset Support:** Successfully harmonizes and trains on diverse global datasets, including Lending Club, the German Credit Market, and generic credit risk profiles.
- **Incremental Learning Pipeline:** Uses XGBoost's `xgb_model` continuation to sequentially learn across multiple distinct datasets without forgetting previous patterns—yielding a massive, generalized "super model".
- **Calibrated Default Probabilities:** Evaluates against the Brier Score to ensure predictions aren't just arbitrary scores, but true mathematical probabilities of default (PD).
- **Bulletproof Test Suite:** Contains a dedicated 108-test `pytest` suite (covering 280 assertions) that rigorously verifies algorithmic contracts, dataset compatibility, and adversarial robustness.

##  Project Structure

```text
quantproj/
├── data/                       # CSV datasets (Lending Club, German, Generic)
├── docs/                       # Project documentation (PRD, HLD, Architecture diagrams)
├── tests/                      # Massive pytest suite (108 tests) for 100% reliability
├── models/                     # (Ignored) Your generated .joblib model artifacts
├── api.py                      # (If applicable) Core API endpoints
├── data_generator.py           # Synthetic data generation fallback
├── fico_bucketing.py           # Custom DP FICO Bucketer Transformer
├── model_training.py           # Multi-model benchmarking (RF, XGBoost, LR)
├── preprocessing.py            # Data imputation and categorical encoding
├── train.py                    # End-to-end CLI training orchestrator
└── incremental_train.py        # Harmonization and sequential XGBoost updating
```

##  Getting Started

### 1. Environment Setup
The project requires Python and standard data science libraries.
```bash
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```

### 2. Training a Model
You can dynamically train a model on any dataset using the CLI orchestrator:
```bash
python train.py --data-path data/credit_risk_dataset.csv --target-col loan_status --fico-col cb_person_cred_hist_length --output model_credit_risk.joblib
```

### 3. Incremental Super-Model
To run the specialized incremental pipeline that learns sequentially across all datasets:
```bash
python incremental_train.py
```

### 4. Running the Test Suite
Ensure the integrity of the entire engine by running the adversarial test suite:
```bash
python -m pytest
```

##  Security & Integrity
This repository enforces strict data hygiene. Buffer files, pycache, massive datasets, and binary model artifacts are ignored via `.gitignore` to keep the codebase lightweight and highly secure.

---
*Built with precision for quantitative finance.*
