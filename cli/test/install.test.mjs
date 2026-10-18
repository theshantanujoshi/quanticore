import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, stat, rm, symlink } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { install } from '../lib/install.mjs';

const packageRoot = fileURLToPath(new URL('..', import.meta.url));
const payloadRoot = path.join(packageRoot, 'payload');

async function withTarget(fn) {
  const target = await mkdtemp(path.join(packageRoot, '.tmp-'));
  try { await fn(target); }
  finally { await rm(target, { recursive: true, force: true }); }
}

test('installs all profiles and is idempotent', async () => withTarget(async target => {
  const first = await install({ target, harnesses: [], payloadRoot });
  assert.ok(first.copied > 11);
  for (const file of [
    '.agents/skills/quanticore-workflow/SKILL.md',
    '.codex/agents/quanticore_coordinator.toml',
    '.claude/agents/quanticore_coordinator.md',
    '.kiro/skills/quanticore-workflow/SKILL.md',
  ]) assert.ok((await stat(path.join(target, file))).isFile());
  const second = await install({ target, harnesses: [], payloadRoot });
  assert.equal(second.copied, 0);
}));

test('bare CLI command installs into the current directory', async () => withTarget(async target => {
  const output = execFileSync(process.execPath, [path.join(packageRoot, 'bin/quanticore.mjs')], {
    cwd: target, encoding: 'utf8',
  });
  assert.match(output, /Installed \d+ files/);
  assert.ok((await stat(path.join(target, '.agents/skills/quanticore-workflow/SKILL.md'))).isFile());
}));

test('selected harness limits profiles while keeping canonical skills', async () => withTarget(async target => {
  await install({ target, harnesses: ['codex'], payloadRoot });
  assert.ok((await stat(path.join(target, '.codex/agents/quanticore_coordinator.toml'))).isFile());
  assert.ok((await stat(path.join(target, '.agents/skills/quanticore-workflow/SKILL.md'))).isFile());
  await assert.rejects(stat(path.join(target, '.claude/agents/quanticore_coordinator.md')), { code: 'ENOENT' });
}));

test('conflict prevents all writes', async () => withTarget(async target => {
  await install({ target, harnesses: ['claude'], payloadRoot });
  const conflict = path.join(target, '.claude/agents/quanticore_coordinator.md');
  await writeFile(conflict, 'local change');
  await assert.rejects(install({ target, harnesses: [], payloadRoot }), /Refusing to overwrite/);
  assert.equal(await readFile(conflict, 'utf8'), 'local change');
  await assert.rejects(stat(path.join(target, '.codex/agents/quanticore_coordinator.toml')), { code: 'ENOENT' });
}));

test('rejects symbolic links in install path', async t => withTarget(async target => {
  const outside = await mkdtemp(path.join(packageRoot, '.tmp-'));
  try {
    try { await symlink(outside, path.join(target, '.agents'), 'junction'); }
    catch (error) { if (error.code === 'EPERM') { t.skip('symlinks unavailable'); return; } throw error; }
    await assert.rejects(install({ target, harnesses: ['aider'], payloadRoot }), /symbolic links/);
    await assert.rejects(stat(path.join(outside, 'skills/quanticore-workflow/SKILL.md')), { code: 'ENOENT' });
  } finally { await rm(outside, { recursive: true, force: true }); }
}));
