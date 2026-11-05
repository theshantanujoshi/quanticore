# Quanticore capability coverage

This kit contains agent instructions, not executable credit-risk software. Use
the skills in a target checkout to build or maintain the capabilities below.
Discover the target project's paths before invoking commands.

| Capability | Skill | Agent |
| --- | --- | --- |
| Full pipeline coordination | `quanticore-workflow` | `quanticore_coordinator` |
| Data provenance, preparation, and target mapping | `quanticore-data-intake` | `quanticore_data_steward` |
| Supervised FICO partitioning | `quanticore-fico-bucketing` | `quanticore_bucketing_specialist` |
| Preprocessing and classifier benchmark | `quanticore-model-training` | `quanticore_model_trainer` |
| Multi-dataset continued training | `quanticore-incremental-training` | `quanticore_incremental_trainer` |
| Single and batch scoring | `quanticore-serving` | `quanticore_serving_specialist` |
| Tests, artifact and claim checks | `quanticore-verification` | `quanticore_verifier` |

Do not describe a capability as implemented until target source and runtime
checks prove it. The main and incremental models have distinct feature
contracts unless the target project explicitly makes them compatible.
