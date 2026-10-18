"""Install Quanticore's project skills and agents into another local checkout.

Usage: python scripts/install-quanticore.py TARGET [--harness NAME ...]
Existing files with different content are never overwritten.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "codex": [".codex/agents"],
    "claude": [".claude/skills", ".claude/agents"],
    "cursor": [".cursor/agents"],
    "gemini": [".gemini/agents"],
    "copilot": [".github/agents"],
    "opencode": [".opencode/agents"],
    "kiro": [".kiro/skills", ".kiro/agents"],
    "cline": [".claude/skills"],
    "aider": [],
    "amp": [],
    "devin": [],
    "antigravity": [],
    "roo": [],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="existing target repository directory")
    parser.add_argument("--harness", action="append", choices=sorted(PROFILES),
                        help="install only for this harness; repeat to select several")
    args = parser.parse_args()
    target = args.target.resolve()
    if not target.is_dir():
        parser.error(f"target is not a directory: {target}")
    if target == ROOT:
        parser.error("target is the Quanticore source repository")
    harnesses = args.harness or list(PROFILES)
    directories = {".agents/skills"}
    for harness in harnesses:
        directories.update(PROFILES[harness])
    sources = [file for directory in sorted(directories)
               for file in (ROOT / directory).rglob("*") if file.is_file()]
    if not sources:
        parser.error("no Quanticore files found")
    for file in sources:
        destination = target / file.relative_to(ROOT)
        if not destination.resolve().is_relative_to(target):
            parser.error(f"destination escapes target: {destination}")
    conflicts = [file.relative_to(ROOT) for file in sources
                 if (target / file.relative_to(ROOT)).exists()
                 and (target / file.relative_to(ROOT)).read_bytes() != file.read_bytes()]
    if conflicts:
        print("Refusing to overwrite differing files:")
        for path in conflicts:
            print(f"  {path.as_posix()}")
        return 1
    copied = 0
    for file in sources:
        destination = target / file.relative_to(ROOT)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, destination)
        copied += 1
    print(f"Installed {copied} files into {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
