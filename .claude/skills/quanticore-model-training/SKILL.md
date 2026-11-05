---
name: quanticore-model-training
description: "Build or evaluate a Quanticore-style preprocessing and classifier benchmark, select a champion, and verify its serialized artifact. Use for training code, model metrics, or feature-contract changes. Not for a separate continued XGBoost path or serving API; use those skills."
---

# Model training

Tie a champion claim to the exact data, split, feature schema, and selection rule that produced it.

## When to use

- Build or change the target project's preprocessing, model benchmark, or training entry point.
- Compare candidate models or inspect an artifact from the main training path.

## When not to use

- Three-dataset continuation: use `quanticore-incremental-training`.
- API request and response behavior: use `quanticore-serving`.

## Inputs to gather

- Data path, target and FICO columns, sample counts, target balance, and split.
- Requested artifact path and existing artifacts to preserve.
- Feature list, metrics, calibration requirement, and candidate models.

## Workflow

1. Use the `quanticore-data-intake` skill to confirm the target and features. If the target project generates synthetic loans when input is missing, ensure that fallback is intended before invoking training.
2. Trace `build_preprocessor`: supervised FICO buckets, numeric median imputation and scaling, categorical mode imputation and one-hot encoding. Check fit/transform consistency and scoring-time feature availability.
3. Inspect or establish the split and candidates. Logistic regression, gradient boosting, random forest, and XGBoost are reasonable candidates when available. All candidates must see the same held-out rows.
4. Evaluate discrimination, precision-recall, Brier loss, and classification metrics with class balance and uncertainty in view. State the champion selection rule before choosing; ROC-AUC with lower Brier as a tie-breaker is one option. A Brier value is a loss, not proof of calibration.
5. If training is requested, choose an isolated output path unless replacing an artifact is explicitly in scope. Record the command, dataset hash or identity, seed/split, metrics, feature schema, chosen model, and produced metadata.
6. Reload the saved pipeline and score representative rows. Check that the target serving API can accept its feature contract before claiming the artifact is ready for serving.

## Guardrails

- Do not use the held-out test set to tune the bucketer or candidates and then call it untouched validation.
- Do not silently replace a current model artifact or report a synthetic-data score as real-world performance.
- Keep training and inference schemas explicit; `incremental_master_model.joblib` is a separate artifact with different features.
- Do not discard unrelated work or weaken a test to pass.

## Verification

- Run the target project's focused model and training tests, then relevant integration tests. Add them for a new implementation.
- For changed serialization, reload a freshly written artifact and verify finite probabilities in `[0,1]` for the intended schema.
- Inspect metrics from the actual run, not a stale report or artifact name.

## Report

Return data and target, split, candidate comparison, selection rule, artifact paths and feature contract, commands and results, and limits of calibration or external validity.

## Optional delegation

The `quanticore_model_trainer` agent can own an assigned training path. This skill also works without delegation.
