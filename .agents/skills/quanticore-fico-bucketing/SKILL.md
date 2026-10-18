---
name: quanticore-fico-bucketing
description: "Implement or assess Quanticore's supervised dynamic-programming FICO bucketer and its learned risk partitions. Use for cutoff behavior, bucket quality, edge cases, or leakage in preprocessing. Not for model benchmarking or API response design; use those specialist skills."
---

# FICO bucketing

Keep FICO boundaries tied to training labels and verify that the fitted transform uses the same interval convention in every path.

## When to use

- Build or review a supervised FICO bucketer such as `FicoDPBucketer` in the target project.
- Explain boundaries, bucket statistics, or integration with preprocessing.

## When not to use

- Compare classifiers: use the `quanticore-model-training` skill.
- Review API tier text: use the `quanticore-serving` skill.

## Inputs to gather

- Score range and type, binary target meaning, sample and unique-score counts.
- Requested maximum bucket count and success measure.
- Whether boundaries are fit within the training split or cross-validation fold.

## Workflow

1. Inspect the target bucketer and tests, or define them if absent. A useful objective is within-bucket binary-target variance: `D - D²/N`; prefix sums support dynamic programming over sorted unique scores. Record the chosen objective rather than assuming one.
2. Check `max_buckets`, length and shape handling, empty and single-score inputs, missing/non-finite scores, and labels. Do not claim behavior for inputs that tests or source do not support.
3. Verify reconstructed boundaries and half-open intervals `[low, high)`. A score equal to a cutoff belongs to the bucket starting at that cutoff. Compare transformed assignments with `bucket_summary_` counts and rates.
4. Trace use through the target preprocessing and training code. The bucketer is supervised and must fit after the split, inside the training pipeline. Investigate whether any transformation loses `y` or fits on held-out rows.
5. If changing the objective or bucket count, compare results on the same split and report the data, objective, minimum bucket sizes, and observed stability. Do not describe cutoffs as universal FICO policy.

## Guardrails

- Never fit supervised cutoffs on evaluation labels and report those results as held-out performance.
- Never equate an empirical bucket default rate with a calibrated individual probability.
- Preserve serialized pipeline compatibility or explicitly version artifact changes.
- Do not discard unrelated work or weaken adversarial tests.

## Verification

- Run focused bucketer and preprocessing tests using the target project's test command. Add boundary and leakage checks for a new implementation.
- Check boundary ordering, count conservation, cutoff equality, and post-serialization transform when an artifact is affected.
- Report dataset and split for any displayed cutoff values.

## Report

Return objective, actual boundaries and interval convention if measured, bucket counts/rates, leakage assessment, tests and results, and any unsupported cases.

## Optional delegation

The `quanticore_bucketing_specialist` agent can analyze assigned algorithm or test paths. The skill works in one session.
