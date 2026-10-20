# Quanticore agent suite CLI

This package installs Quanticore skills and matching agent profiles into an existing checkout. It does not train a model or score loans by itself. This repository contains no credit risk engine; use the skills in a target checkout to build or maintain one.

## Install from npm

From a target checkout, run:

```powershell
npx quanticore-agent-suite
```

To choose a target or limit profiles to one harness:

```powershell
npx quanticore-agent-suite init D:\path\to\target-repo --harness codex
```

Open that checkout in your coding agent and ask it to use `quanticore-workflow`
or a specialist such as `quanticore-fico-bucketing`. The package installs
instructions; it does not include datasets, model artifacts, or a scoring API.

## Local source use

Requires Node.js 20 or later. From the Quanticore repository root:

```powershell
node cli/scripts/prepare-payload.mjs
node cli/bin/quanticore.mjs init D:\path\to\target-repo --harness codex
```

Run `node cli/bin/quanticore.mjs --help` for available harnesses. With no arguments, the command installs into the current directory. Canonical `.agents/skills` files are always included; `--harness` selects native profiles. The installer is repeatable and refuses differing existing files or symbolic-link paths.

## Build and test

```powershell
python scripts/generate-harness-adapters.py
python scripts/validate-quanticore-kit.py
python scripts/test-install-quanticore.py
npm --prefix cli test
```

`prepack` refreshes the package payload from canonical skills and generated profiles. The package has no runtime dependencies. See `docs/quanticore-kit-authoring.md` for the authoring contract.

## Release setup

The [release workflow](../.github/workflows/cli-release.yml) runs when a GitHub release is published from a `v` tag matching `cli/package.json`. It checks Node.js 20 and 24, packs the tagged commit once, and publishes that tarball with provenance. The `npm-release` GitHub environment requires a reviewer and a `QUANTICORE` secret containing an npm publish token.
