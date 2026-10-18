# Authoring the Quanticore kit

The canonical seven skills live in `.agents/skills`. Each has one matching Codex agent in `.codex/agents`. Harness profiles and Claude/Kiro skill shims are generated from those sources, following the local Codelore framework and the [Indium-AgentKit authoring standard](https://github.com/Indium-AI-Labs/indium-agentkit).

Edit canonical files first. Skills must have `name` and quoted `description` frontmatter, **When to use**, **Inputs to gather**, **Workflow**, and **Report** sections. Keep procedures usable by one agent and tie model claims to data, split, artifact, tests, and source evidence. The workflow references contain capability coverage and the shared evidence contract. Agent profiles point to exactly one canonical skill and add a bounded role, scope, procedure, and result format.

The generator creates profiles for Claude, Cursor, Gemini, GitHub Copilot, OpenCode, and Kiro. Codex reads the canonical skill and `.codex/agents` profile. The Python and Node installers preserve existing differing files.

Run each command from the repository root:

```powershell
python scripts/generate-harness-adapters.py
python scripts/validate-quanticore-kit.py
python scripts/test-install-quanticore.py
npm --prefix cli test
git diff --check
```

The validator checks frontmatter, required sections, links, one-to-one coverage, and generated adapter consistency. Installation tests check repeat installs and conflict refusal. These checks do not by themselves prove discovery by every installed harness version. This repository contains no credit risk runtime or model test suite; use the installed skills to build and test one in a target checkout.
