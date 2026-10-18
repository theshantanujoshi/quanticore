import { cp, mkdir, rm, stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const packageRoot = fileURLToPath(new URL('..', import.meta.url));
const repoRoot = path.resolve(packageRoot, '..');
const payloadRoot = path.join(packageRoot, 'payload');
const directories = [
  '.agents/skills', '.codex/agents', '.claude/skills', '.claude/agents',
  '.cursor/agents', '.gemini/agents', '.github/agents', '.opencode/agents',
  '.kiro/skills', '.kiro/agents',
];

await rm(payloadRoot, { recursive: true, force: true });
for (const directory of directories) {
  const source = path.join(repoRoot, directory);
  if (!(await stat(source)).isDirectory()) throw new Error(`Missing source directory: ${source}`);
  const destination = path.join(payloadRoot, directory);
  await mkdir(path.dirname(destination), { recursive: true });
  await cp(source, destination, { recursive: true, force: false });
}
console.log(`Prepared Quanticore payload in ${payloadRoot}`);
