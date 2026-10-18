---
name: quanticore-verification
description: "Verify Quanticore changes against tests, serialized artifacts, model metrics, and documented claims. Use before declaring a data, bucketing, training, incremental, or serving task complete. Not for choosing a new algorithm or designing a feature; use the relevant specialist skill."
---

# Verification

Match each claim to an observed check at the same scope. Use `../quanticore-workflow/references/evidence.md` for the evidence contract.

## When to use

- Validate a completed Quanticore change or investigate a failing check.
- Audit reported scores, model artifacts, or API readiness claims.

## When not to use

- Plan a new model: use `quanticore-model-training`.
- Implement a specialized stage: use its matching skill, then return here.

## Inputs to gather

- Exact change and acceptance criteria; worktree diff and revision.
- Test environment, available datasets, artifact paths, and claimed metrics.
- Required inference schema and expected API behavior.

## Workflow

1. Convert each requested outcome into a specific source, test, artifact, or runtime check. Include labels, leakage, serialization, probability bounds, and API behavior where applicable.
2. Inspect the current diff and target repository test layout. Run focused tests first, then its discovered full test command when the environment supports it. Record failures and skips; do not present a subset as the entire suite.
3. For a model artifact, verify provenance and metadata, load it from a trusted path, and score rows matching its documented feature schema. Check shape, finite probabilities, and bounds. Avoid executing unknown joblib files.
4. For metrics, identify dataset, target mapping, split, model version, and computation. Compare project documentation claims only with matching runs; stale numbers remain unverified.
5. For API changes, exercise loaded and unloaded behavior and compare single and batch outputs to the same model where possible.
6. State residual risks. A passing unit suite does not establish real-world calibration, fairness, compliance, absence of leakage, or production deployment.

## Guardrails

- Never weaken a test, alter a metric, or replace an artifact to produce a green report.
- Never run expensive retraining or overwrite data solely for verification when focused checks suffice.
- Do not discard unrelated work or present fixture-only checks as real-data validation.

## Verification

- Confirm every reported command actually ran and record exit status.
- Recheck the changed files and `git diff --check` after fixes.
- If a required check cannot run, record the precise missing dependency, data, or artifact and the next check needed.

## Report

Return status (`PASSED`, `PARTIAL`, `FAILED`, or `BLOCKED`), scope, checks with results, artifact and metric evidence, unverified requirements, and actionable follow-up.

## Optional delegation

The `quanticore_verifier` agent can run an independent bounded check. A single agent can perform the whole workflow.
