# Quanticore kit test record

The canonical kit contains seven Agent Skills and seven optional Codex agents, with generated
profiles for Claude, Cursor, Gemini, GitHub Copilot, OpenCode, and Kiro.

## Kit checks

| Check | Result |
| --- | --- |
| `D:\python.exe scripts/generate-harness-adapters.py --check` | Passed; 60 generated files current |
| `D:\python.exe scripts/validate-quanticore-kit.py` | Passed; frontmatter fields, links, one-to-one mapping, adapter consistency |
| `python scripts/test-install-quanticore.py` | Passed; three installation tests |
| `npm.cmd --prefix cli test` | Passed; five tests including repeat install, harness selection, conflict refusal, and symlink path refusal |
| `npm.cmd pack --dry-run --json --cache D:\quantproj\cli\.npm-cache` from `cli/` | Passed; 81 package entries, including complete Claude and Kiro skill directories |
| `git diff --check` | Passed |

`D:\python.exe` was used because the `python` launcher on this machine reports
`No pyvenv.cfg file`. The optional skill-creator `quick_validate.py` could not
run because PyYAML is not installed in that interpreter; the repository validator
and installer tests passed.

## Suite-only boundary

This repository contains no training code, datasets, model artifacts, API, or
application tests. The kit checks above exercise the skill and agent mappings,
generated adapters, installer behavior, and package contents. Model and API
behavior must be tested in a target checkout after the suite is used there.
