"""Validate Quanticore's canonical skill/agent suite and generated adapters."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / ".agents" / "skills"
AGENTS = ROOT / ".codex" / "agents"
SKILL_REF = re.compile(r"\.agents/skills/([a-z0-9-]+)/SKILL\.md")
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
REQUIRED_SECTIONS = ("## When to use", "## Inputs to gather", "## Workflow", "## Report")


def validate() -> list[str]:
    errors: list[str] = []
    skills: dict[str, Path] = {}
    for path in sorted(SKILLS.glob("*/SKILL.md")):
        content = path.read_text(encoding="utf-8-sig")
        relative = path.relative_to(ROOT).as_posix()
        if not content.startswith("---\n") or "\n---\n" not in content[4:]:
            errors.append(f"{relative}: missing frontmatter")
            continue
        header, body = content[4:].split("\n---\n", 1)
        fields = dict(re.findall(r"^(name|description):\s*(.+)$", header, re.MULTILINE))
        if set(re.findall(r"^([A-Za-z_-]+):", header, re.MULTILINE)) != {"name", "description"}:
            errors.append(f"{relative}: frontmatter must contain only name and description")
        if fields.get("name") != path.parent.name:
            errors.append(f"{relative}: name does not match directory")
        description = fields.get("description", "").strip()
        if not (description.startswith('"') and description.endswith('"')):
            errors.append(f"{relative}: description must be quoted")
        if len(description.strip('"')) < 80 or "not for" not in description.lower():
            errors.append(f"{relative}: description is too vague")
        if len(content.splitlines()) < 35:
            errors.append(f"{relative}: skill needs substantive procedure and checks")
        for section in REQUIRED_SECTIONS:
            if section not in body:
                errors.append(f"{relative}: missing {section}")
        for raw_link in LINK.findall(body):
            target = raw_link.split("#", 1)[0]
            if target and not target.startswith(("http://", "https://")):
                resolved = (path.parent / target).resolve()
                if not resolved.is_relative_to(ROOT.resolve()) or not resolved.exists():
                    errors.append(f"{relative}: broken link {raw_link}")
        skills[path.parent.name] = path

    mapped: set[str] = set()
    agent_paths = sorted(AGENTS.glob("*.toml"))
    for path in agent_paths:
        relative = path.relative_to(ROOT).as_posix()
        try:
            agent = tomllib.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{relative}: {exc}")
            continue
        if agent.get("name") != path.stem:
            errors.append(f"{relative}: name does not match filename")
        references = SKILL_REF.findall(agent.get("developer_instructions", ""))
        if len(references) != 1 or references[0] not in skills:
            errors.append(f"{relative}: expected exactly one existing canonical skill")
        else:
            if references[0] in mapped:
                errors.append(f"{relative}: duplicate mapping for {references[0]}")
            mapped.add(references[0])
        if len(agent.get("developer_instructions", "").strip()) < 250:
            errors.append(f"{relative}: agent instructions need scope and output guidance")

    if not skills or mapped != set(skills):
        errors.append(f"Skill/agent coverage differs: skills={len(skills)}, mapped={len(mapped)}")
    if not agent_paths:
        errors.append("No Codex agent profiles found")

    generated = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate-harness-adapters.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if generated.returncode:
        errors.append("Generated adapters are stale: " + (generated.stderr or generated.stdout).strip())
    return errors


if __name__ == "__main__":
    problems = validate()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(1)
    print("Validated Quanticore skills, agent mappings, links, and generated adapters")
