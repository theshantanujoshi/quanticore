# Quanticore kit test record

The canonical kit contains seven skills, seven Codex agents, and generated
profiles for Claude, Cursor, Gemini, GitHub Copilot, OpenCode, and Kiro.

## Kit checks

| Check | Result |
| --- | --- |
| `python scripts/generate-harness-adapters.py --check` | Passed; 56 adapters current |
| `python scripts/validate-quanticore-kit.py` | Passed; frontmatter, sections, links, one-to-one mapping, adapter consistency |
| `python scripts/test-install-quanticore.py` | Passed; three installation tests |
| `npm.cmd --prefix cli test` | Passed; five tests including repeat install, harness selection, conflict refusal, and symlink path refusal |
| `npm.cmd pack --dry-run --json --cache D:\quantproj\cli\.npm-cache` from `cli/` | Passed; 77 package entries including all seven canonical skills and agent profiles |
| `git diff --check` | Passed |

## Suite-only boundary

This repository contains no training code, datasets, model artifacts, API, or
application tests. The kit checks above exercise the skill and agent mappings,
generated adapters, installer behavior, and package contents. Model and API
behavior must be tested in a target checkout after the suite is used there.
