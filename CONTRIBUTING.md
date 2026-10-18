# Contributing to Quanticore

Quanticore publishes agent instructions and an installer. The canonical seven skills are in `.agents/skills`, and their matching Codex agents are in `.codex/agents`. See the [authoring guide](docs/quanticore-kit-authoring.md) before editing them; generated harness adapters should be updated through the generator.

## Check a change

From the repository root, with Python 3 and Node.js 20 or later installed, run:

```powershell
python scripts/generate-harness-adapters.py
python scripts/validate-quanticore-kit.py
python scripts/test-install-quanticore.py
npm --prefix cli test
git diff --check
```

The validator checks skill structure and generated profiles. The installer tests cover repeat installation and conflict refusal. Describe any behavior you could not verify in your pull request.

Use [GitHub Issues](https://github.com/theshantanujoshi/quanticore/issues) to report bugs or suggest additions. Include the target harness, command, expected result, and actual result when reporting an installer problem. Do not attach private loan data, model artifacts, or credentials.
