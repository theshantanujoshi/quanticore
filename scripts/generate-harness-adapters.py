"""Generate native agent profiles and Claude skill shims from Quanticore sources.

Usage: python scripts/generate-harness-adapters.py [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / ".agents" / "skills"
CODEX_AGENTS = ROOT / ".codex" / "agents"
PROFILE_DIRS = {
    "claude": ROOT / ".claude" / "agents",
    "cursor": ROOT / ".cursor" / "agents",
    "gemini": ROOT / ".gemini" / "agents",
    "github": ROOT / ".github" / "agents",
    "opencode": ROOT / ".opencode" / "agents",
    "kiro": ROOT / ".kiro" / "agents",
}
SKILL_PATH = re.compile(r"\.agents/skills/([a-z0-9-]+)/SKILL\.md")


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def adapter_content(agent: dict[str, str], harness: str, skill_name: str) -> str:
    name = agent["name"]
    description = agent["description"]
    frontmatter = ["---", f"name: {yaml_string(name)}", f"description: {yaml_string(description)}"]
    if harness in {"claude", "cursor"}:
        frontmatter.extend(["skills:", f"  - {skill_name}"] if harness == "claude" else [])
    if harness == "opencode":
        frontmatter.append("mode: subagent")
    frontmatter.append("---")
    instructions = agent["developer_instructions"].strip()
    return "\n".join(frontmatter) + "\n\n" + instructions + "\n"


def claude_skill_content(skill_name: str, description: str) -> str:
    return (
        "---\n"
        f"name: {yaml_string(skill_name)}\n"
        f"description: {yaml_string(description)}\n"
        "---\n\n"
        f"Read `.agents/skills/{skill_name}/SKILL.md` from the repository root "
        "and follow that canonical skill. Resolve its relative reference links "
        f"from `.agents/skills/{skill_name}/`.\n"
    )


def kiro_agent_content(agent: dict[str, str], skill_name: str) -> str:
    return json.dumps({
        "name": agent["name"],
        "description": agent["description"],
        "resources": [f"skill://.kiro/skills/{skill_name}/SKILL.md"],
        "prompt": agent["developer_instructions"].strip(),
    }, ensure_ascii=False, indent=2) + "\n"


def skill_metadata(path: Path) -> tuple[str, str]:
    content = path.read_text(encoding="utf-8-sig")
    if not content.startswith("---\n"):
        raise ValueError(f"Missing YAML frontmatter: {path}")
    end = content.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"Unclosed YAML frontmatter: {path}")
    header = content[4:end]
    fields = dict(re.findall(r"^(name|description):\s*(.+)$", header, re.MULTILINE))
    name = fields.get("name", "").strip().strip('"')
    description = fields.get("description", "").strip().strip('"')
    if not name or not description or name != path.parent.name:
        raise ValueError(f"Invalid skill name or description: {path}")
    return name, description


def expected_files() -> dict[Path, str]:
    result: dict[Path, str] = {}
    skills = {
        name: description
        for name, description in (
            skill_metadata(path) for path in sorted(SKILLS.glob("*/SKILL.md"))
        )
    }
    if not skills:
        raise ValueError("No source skills found")
    for agent_path in sorted(CODEX_AGENTS.glob("*.toml")):
        agent = tomllib.loads(agent_path.read_text(encoding="utf-8-sig"))
        name = agent["name"]
        if agent_path.stem != name:
            raise ValueError(f"Agent filename/name mismatch: {agent_path}")
        paths = SKILL_PATH.findall(agent["developer_instructions"])
        if len(paths) != 1 or paths[0] not in skills:
            raise ValueError(f"Expected one source skill in {agent_path}")
        skill_name = paths[0]
        for harness, directory in PROFILE_DIRS.items():
            if harness == "kiro":
                result[directory / f"{name}.json"] = kiro_agent_content(agent, skill_name)
            else:
                suffix = ".agent.md" if harness == "github" else ".md"
                result[directory / f"{name}{suffix}"] = adapter_content(agent, harness, skill_name)
    if len(result) != len(PROFILE_DIRS) * len(skills):
        raise ValueError("Expected one agent per skill and harness")
    for name, description in skills.items():
        result[ROOT / ".claude" / "skills" / name / "SKILL.md"] = claude_skill_content(name, description)
        result[ROOT / ".kiro" / "skills" / name / "SKILL.md"] = claude_skill_content(name, description)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated files are missing or stale")
    args = parser.parse_args()
    expected = expected_files()
    stale = []
    for path, content in expected.items():
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            if args.check:
                stale.append(path.relative_to(ROOT).as_posix())
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8", newline="\n")
    if stale:
        print("Missing or stale adapters:\n" + "\n".join(stale), file=sys.stderr)
        return 1
    action = "verified" if args.check else "generated"
    print(f"{action} {len(expected)} adapters for {len(PROFILE_DIRS)} agent harnesses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
