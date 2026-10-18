---
name: quanticore-incremental-training
description: "Build or inspect a three-dataset harmonization and XGBoost continuation path for credit risk. Use for incremental training, Lending Club/German/generic schema mapping, continuation tests, or its separate artifact. Not for the main classifier benchmark or API serving contract."
---

# Incremental training

Maintain one harmonized feature meaning across all three datasets and verify what continued boosting actually changes.

## When to use

- Build or change a loader or incremental training entry point in the target project.
- Investigate a separate incremental artifact or continuation order.

## When not to use

- Main benchmark: use `quanticore-model-training`.
- Serving the main five-feature model: use `quanticore-serving`.

## Inputs to gather

- Availability and provenance of prepared Lending Club, generic credit risk, and German Credit datasets, if those are the requested sources.
- Source-to-target mappings and binary default meaning for each dataset.
- Shared feature schema, category encodings, continuation order, and output path.

## Workflow

1. Use `quanticore-data-intake` to inspect each prepared source. Verify labels and mapping before concatenation. Check how missing employment and purpose values are represented.
2. Trace or implement the source loaders. Harmonize loan amount, employment tenure, purpose, and target with source-specific mappings. Find any semantic mismatch before fitting.
3. Inspect or build the shared preprocessor. If it fits category dimensions across sources before staged XGBoost fits, distinguish shared unsupervised feature mapping from label leakage; any holdout evaluation requires a split design that avoids fitting evaluation data.
4. For staged `XGBClassifier.fit` calls using `xgb_model=get_booster()`, note that continuation appends trees; it does not prove the model retains performance on previous datasets. Measure each dataset before and after stages if retention is a requirement.
5. If running the pipeline, verify data files, compute cost, output directory, and destination artifact. Use an isolated copy or path for experiments and record source versions and training order.
6. Reload the final pipeline and score the harmonized schema. Do not pass this artifact to a main-model API without an explicit schema adapter and tests.

## Guardrails

- Do not claim "no forgetting" from continuation alone.
- Do not mix target polarity across datasets or call cross-dataset evaluation representative without provenance and split evidence.
- Do not overwrite an existing artifact during exploratory checks or discard unrelated work.

## Verification

- Run focused incremental and dataset compatibility tests in the target project, adding them when building a new path.
- Confirm each loader's output columns and binary labels on actual data, or mark checks as fixture-only.
- After a training change, reload and score the produced artifact with the harmonized feature schema.

## Report

Return source mappings, dataset availability, training sequence, feature contract, artifact path, measured performance if any, tests, and unverified retention or generalization claims.

## Optional delegation

The `quanticore_incremental_trainer` agent can handle an assigned continuation path. This skill is complete for one agent.
