# Quanticore agent suite

[![npm version](https://img.shields.io/npm/v/quanticore-agent-suite)](https://www.npmjs.com/package/quanticore-agent-suite) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Quanticore is a portable set of seven skills and seven matching agents for building and reviewing a quantitative credit risk pipeline. This repository contains **instructions and installation tools only**. It does not include loan datasets, trained models, training code, a scoring API, or a frontend.

The [workflow skill](.agents/skills/quanticore-workflow/SKILL.md) coordinates data intake, supervised FICO bucketing, model training, three-dataset continuation, API serving, and verification. Each specialist skill also works alone. The suite uses the same canonical skill, generated adapter, and conflict-safe installer layout as Codelore, with [Indium-AgentKit](https://github.com/Indium-AI-Labs/indium-agentkit) authoring guidance.

## Use the suite

Requires Node.js 20 or later. Install it into an existing checkout where you want to create or maintain a credit risk engine:

```powershell
npx quanticore-agent-suite init D:\path\to\target-repo --harness codex
```

From this source checkout, `python scripts/install-quanticore.py D:\path\to\target-repo --harness codex` provides the same files without npm.

Then ask your coding agent, for example: `Use the quanticore-workflow skill to build and verify a credit-risk training and serving pipeline in this checkout.` For a focused task, name `quanticore-fico-bucketing`, `quanticore-model-training`, or another specialist. A single agent can follow every skill; delegation is optional.

The [CLI guide](cli/README.md) covers other harnesses. The [authoring guide](docs/quanticore-kit-authoring.md) explains canonical files and validation. See the [test record](docs/quanticore-kit-test.md) for the latest suite checks.

## What is included

- Seven skills covering data intake, FICO bucketing, model training, incremental training, serving, verification, and the end-to-end workflow.
- Seven matching agent profiles. Codex profiles live alongside the canonical skills; generated adapters support Aider, Amp, Antigravity, Claude, Cline, GitHub Copilot, Cursor, Devin, Gemini, Kiro, OpenCode, and Roo.
- A conflict-safe installer that copies the skills and selected agent profiles into an existing checkout. Run `npx quanticore-agent-suite --help` for the current harness list and command syntax.

## Contribute and get help

See [CONTRIBUTING.md](CONTRIBUTING.md) for the authoring workflow and local checks. Use [GitHub Issues](https://github.com/theshantanujoshi/quanticore/issues) for bugs and feature requests.

## License

Quanticore is available under the [MIT License](LICENSE).
