# Original User Request

## Initial Request — 2026-09-05T08:19:40Z

Develop a production-ready quantitative credit risk model that predicts the probability of default, designed as a polished portfolio piece for recruiters and hiring managers. The project involves robust data preprocessing, an optimized dynamic programming FICO score bucketer, and advanced predictive modeling to complete the pipeline.

Working directory: D:\quantproj
Integrity mode: development

## Requirements

### R1. End-to-End Model Pipeline
Fix the existing codebase and ensure the pipeline runs end-to-end (from data generation to model evaluation). The team should evaluate multiple models (e.g., Logistic Regression, XGBoost) and select the one that yields the best results. The pipeline must correctly utilize the custom dynamic programming FICO bucketer.

### R2. Serving API
Build a deployment-ready serving API to demonstrate how the model would be queried in production. It should accept customer and loan features and return a calibrated probability of default.

### R3. Comprehensive Reporting
Generate a detailed Markdown report summarizing the data analysis, the FICO bucketing logic, model selection, and final performance metrics. The report must be polished and targeted at recruiters/hiring managers.

## Acceptance Criteria

### E2E Pipeline
- [ ] `pytest test_pipeline.py` (and any new test files) passes with 0 failures.
- [ ] A top-level script (e.g., `train.py` or `run.py`) executes without errors and successfully saves a trained model artifact to disk.
- [ ] The execution outputs evaluation metrics (including ROC-AUC).

### Serving API
- [ ] A test script (e.g., `test_api.py`) can successfully send a request to the API with sample features and receive a valid JSON response containing a probability (float between 0 and 1).

### Documentation
- [ ] A `REPORT.md` (or similar) file is created.
- [ ] The report explicitly states the final model's ROC-AUC score and explains the optimal FICO boundaries chosen by the bucketer.

## Follow-up — 2026-09-05T12:15:43Z

# Teamwork Project Prompt - Draft

> Status: Step 4-6 - Drafting requirements and acceptance criteria
> Goal: Craft prompt   get user approval   delegate to teamwork_preview
> Requested team: The full team

Write a comprehensive test suite for the quantitative credit risk engine, verifying data processing, model training pipelines, the incremental learning script, and the generated model artifacts. Use a full team of agents.

Working directory: D:\quantproj
Integrity mode: development

## Requirements

### R1. Comprehensive Test Coverage
Implement tests that verify the core logic of `preprocessing.py`, `model_training.py`, `train.py`, and `incremental_train.py`. The tests should ensure that categorical bucketing, imputation, and model serialization work as expected.

### R2. Dataset Compatibility Tests
Include integration tests that verify the pipeline can successfully process and train models on the three different dataset formats (Lending Club, German Credit, and generic credit risk).

### R3. Model Inference Checks
Ensure tests verify that the saved `.joblib` model artifacts can be loaded and produce valid probability scores (between 0.0 and 1.0) on sample data.

## Acceptance Criteria

### Test Execution
- [ ] A `pytest` test suite is created.
- [ ] All tests pass successfully when `pytest` is run from the command line.
- [ ] No hardcoded dependencies on absolute paths; tests must use relative paths to the `data/` directory.

---
*Next: Please review this draft. If it looks good, say "go" or "launch" and I will delegate it to the team!*
