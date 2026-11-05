---
name: quanticore-serving
description: "Build or verify a FastAPI credit-risk scoring contract, artifact loading, single and batch predictions, and health behavior. Use for serving code, inference errors, request validation, or model-version reporting. Not for training or selecting the model; use the training skill."
---

# Serving

Verify the loaded model and the request schema together. A response shape alone does not prove useful credit risk scores.

## When to use

- Build, edit, or review scoring endpoints or model loading in a target project.
- Diagnose inconsistent single/batch inference or 503 behavior.

## When not to use

- Train or select an artifact: use `quanticore-model-training`.
- Analyze source datasets: use `quanticore-data-intake`.

## Inputs to gather

- Intended artifact and metadata paths, model feature schema, and version.
- Example request and expected status/response; whether model is loaded.
- Deployment environment, data sensitivity, and allowed network exposure.

## Workflow

1. Inspect or define request and response models, artifact loading, and health, single-score, and batch-score routes. A main-model feature contract can use FICO score, income, loan amount, DTI, and employment length; verify the trained artifact before fixing the schema. Document any accepted aliases.
2. Verify which artifact path the loader actually chooses, whether metadata matches the pipeline, and whether custom classes can deserialize. Prefer explicit configuration and inspect the loaded artifact rather than trusting the health text.
3. Compare single and batch DataFrame construction and response ordering. Check finite probabilities within `[0,1]`, class consistency, FICO boundary labels, ID passthrough, and empty batches.
4. Check validation failures, missing artifact 503s, and computation failures. Avoid returning sensitive exception text or applicant features in logs and responses.
5. Distinguish technical risk tiers and the model's binary prediction from an authorized underwriting decision. Thresholds in API code are application behavior, not lending policy.
6. For a change, exercise the endpoint with a known test artifact and one representative request; do not expose the service publicly just to test it.

## Guardrails

- Do not deserialize untrusted joblib artifacts; they can execute code during loading.
- Do not disclose applicant PII or raw loan data in logs, examples, or reports.
- Do not claim an API is production ready from `/health` or a synthetic request alone.
- Preserve existing model artifacts and unrelated work.

## Verification

- Run the target project's API and inference tests. Add loaded, unloaded, and batch cases for a new implementation.
- Test loaded and unloaded states, single and batch responses, schema errors, and probability bounds.
- Report the actual artifact path and whether live inference was exercised.

## Report

Return routes and schemas affected, artifact identity, behavior observed, commands and results, and security or deployment limits.

## Optional delegation

The `quanticore_serving_specialist` agent can review assigned API paths. The skill also works in one session.
