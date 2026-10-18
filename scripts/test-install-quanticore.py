"""Smoke tests for installing Quanticore into a separate checkout."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install-quanticore.py"


class InstallQuanticoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.target = Path(self.temporary.name).resolve()
        self.assertTrue(self.target.is_relative_to(ROOT.resolve()))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def install(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(INSTALLER), str(self.target), *args],
            capture_output=True, text=True, check=False,
        )

    def test_all_harnesses_and_idempotence(self) -> None:
        first = self.install()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        for path in (
            ".agents/skills/quanticore-workflow/SKILL.md",
            ".codex/agents/quanticore_coordinator.toml",
            ".claude/agents/quanticore_coordinator.md",
            ".claude/skills/quanticore-workflow/SKILL.md",
            ".cursor/agents/quanticore_coordinator.md",
            ".gemini/agents/quanticore_coordinator.md",
            ".github/agents/quanticore_coordinator.agent.md",
            ".opencode/agents/quanticore_coordinator.md",
            ".kiro/agents/quanticore_coordinator.json",
            ".kiro/skills/quanticore-workflow/SKILL.md",
        ):
            self.assertTrue((self.target / path).is_file(), path)
        second = self.install()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("Installed 0 files", second.stdout)

        for profile in (self.target / ".kiro/agents").glob("*.json"):
            data = json.loads(profile.read_text(encoding="utf-8"))
            self.assertEqual(data["name"], profile.stem)
            self.assertTrue(data["prompt"])
            for resource in data["resources"]:
                self.assertTrue(resource.startswith("skill://"), resource)
                self.assertTrue((self.target / resource.removeprefix("skill://")).is_file(), resource)

    def test_conflict_refuses_all_writes(self) -> None:
        self.assertEqual(self.install("--harness", "claude").returncode, 0)
        conflict = self.target / ".claude/agents/quanticore_coordinator.md"
        conflict.write_text("locally modified\n", encoding="utf-8")
        result = self.install()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Refusing to overwrite", result.stdout)
        self.assertEqual(conflict.read_text(encoding="utf-8"), "locally modified\n")
        self.assertFalse((self.target / ".codex/agents/quanticore_coordinator.toml").exists())

    def test_skill_only_harness_selection(self) -> None:
        result = self.install("--harness", "aider")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.target / ".agents/skills/quanticore-workflow/SKILL.md").is_file())
        self.assertFalse((self.target / ".claude/agents/quanticore_coordinator.md").exists())


if __name__ == "__main__":
    unittest.main()
