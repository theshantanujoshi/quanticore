---
name: quanticore-data-intake
description: "Inspect Quanticore loan datasets, feature schemas, target meanings, and preparation paths before modeling. Use for new data, dataset compatibility, label mapping, missing values, or leakage questions. Not for model selection or API scoring; use the training or serving skill."
---

# Data intake

Establish what each row and binary label means before fitting a model. Work from the available source and data rather than treating file names as a schema.

## When to use

- A dataset is added, prepared, harmonized, or questioned.
- Training fails because columns or target values do not match expectations.

## When not to use

- Model comparison: use the `quanticore-model-training` skill.
- Scoring request validation: use the `quanticore-serving` skill.

## Inputs to gather

- Dataset path, provenance, license or usage limits, row count, columns, and types.
- Target column and which value means default; FICO column if present.
- Missing values, duplicate or ID fields, time ordering, and sensitive attributes.
- Whether the source is synthetic, Lending Club, German credit, or generic credit risk.

## Workflow

1. Identify the data path from the task and check availability without printing borrower records or secrets. Inspect headers, types, label counts, and a minimal safe sample only when needed.
2. Compare the schema with the target project's generator, preparation, training, and harmonization code if present. If building from scratch, define those contracts explicitly. Do not silently substitute generated data for a requested real dataset.
3. Confirm target semantics. For German Credit, verify how `kredit` maps to default; for Lending Club, identify terminal statuses before mapping. Check the actual prepared file before training; a reversal invalidates every metric.
4. Check that features are known at scoring time. Identify identifiers, post-outcome fields, duplicates, missingness, outliers, and categorical cardinality. State whether a time or group split is needed for the requested use case.
5. For a main pipeline, define numeric and categorical feature handling. For incremental training, verify each source loader and a shared `loan_amount`, `emp_length`, `purpose` feature schema, or document a justified alternative.
6. Return a compact schema and target mapping. If source data are unavailable, distinguish code expectations from observed data properties.

## Guardrails

- Never upload or expose row-level applicant data to answer a schema question.
- Do not overwrite source or prepared data as part of inspection. Preparation writes derived CSV files only when authorized by the task.
- Do not infer target semantics from a model's output shape or dataset name.
- Do not discard unrelated work or weaken tests to fit an unexpected dataset.

## Verification

- Confirm the target has exactly the intended classes and enough examples for the requested split.
- Check that reported columns and counts come from the actual file, or label them as code expectations.
- Run the target project's focused dataset compatibility tests where available, then report the command and result. For new implementations, add tests that prove the target mapping and schema.

## Report

Return dataset identity and provenance, schema, default-label mapping, missingness and leakage concerns, compatibility with each requested pipeline, checks run, and unverified assumptions.

## Optional delegation

The `quanticore_data_steward` agent can inspect an assigned dataset and return a schema handoff. The skill works in one session.
