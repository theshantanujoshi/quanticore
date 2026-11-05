# Authoring the Quanticore kit

The canonical seven skills live in `.agents/skills` and follow the [Agent Skills specification](https://agentskills.io/specification). Each has one optional matching Codex agent in `.codex/agents`. Harness profiles and complete Claude/Kiro skill directories are generated from those sources. Every generated skill directory includes its own `SKILL.md` and relative references, so it works without reading `.agents/skills`.

Edit canonical files first. Skills require a lowercase hyphenated `name` matching the directory and a nonempty `description` explaining the capability and when to use it. Optional frontmatter fields follow the specification. Use sections that help the task; the format does not require particular headings or a minimum length. Keep procedures usable by one agent and tie model claims to data, split, artifact, tests, and source evidence. The workflow references contain capability coverage and the shared evidence contract. Agent profiles point to exactly one canonical skill and add a bounded role, scope, procedure, and result format.

The generator creates profiles for Claude, Cursor, Gemini, GitHub Copilot, OpenCode, and Kiro. Codex reads the canonical skill and `.codex/agents` profile. The Python and Node installers preserve existing differing files.

Run each command from the repository root:

```powershell
python scripts/generate-harness-adapters.py
python scripts/validate-quanticore-kit.py
python scripts/test-install-quanticore.py
npm --prefix cli test
git diff --check
```

The validator checks frontmatter fields, links, one-to-one agent coverage, and generated adapter consistency. For independent format validation, run `skills-ref validate .agents/skills/<skill-name>` when the [reference library](https://github.com/agentskills/agentskills/tree/main/skills-ref) is installed. Installation tests check repeat installs and conflict refusal. These checks do not by themselves prove discovery by every installed harness version. This repository contains no credit risk runtime or model test suite; use the installed skills to build and test one in a target checkout.
