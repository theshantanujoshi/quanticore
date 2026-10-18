import { readdir, readFile, lstat, mkdir, copyFile, realpath } from 'node:fs/promises';
import path from 'node:path';

export const PROFILES = {
  codex: ['.codex/agents'],
  claude: ['.claude/skills', '.claude/agents'],
  cursor: ['.cursor/agents'],
  gemini: ['.gemini/agents'],
  copilot: ['.github/agents'],
  opencode: ['.opencode/agents'],
  kiro: ['.kiro/skills', '.kiro/agents'],
  cline: ['.claude/skills'],
  aider: [], amp: [], devin: [], antigravity: [], roo: [],
};

async function listFiles(root, relative = '') {
  const entries = await readdir(path.join(root, relative), { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const child = path.join(relative, entry.name);
    if (entry.isDirectory()) files.push(...await listFiles(root, child));
    else if (entry.isFile()) files.push(child);
    else throw new Error(`Unexpected payload entry: ${child}`);
  }
  return files;
}

async function existingPath(file) {
  try { return await lstat(file); }
  catch (error) { if (error.code === 'ENOENT') return null; throw error; }
}

export async function install({ target, harnesses, payloadRoot }) {
  const targetRoot = await realpath(target);
  const targetState = await lstat(targetRoot);
  if (!targetState.isDirectory()) throw new Error(`Target is not a directory: ${target}`);
  const selected = harnesses.length ? harnesses : Object.keys(PROFILES);
  for (const harness of selected) {
    if (!(harness in PROFILES)) throw new Error(`Unknown harness: ${harness}`);
  }
  const directories = new Set(['.agents/skills']);
  for (const harness of selected) for (const directory of PROFILES[harness]) directories.add(directory);
  const files = (await Promise.all([...directories].map(directory => listFiles(payloadRoot, directory)))).flat().sort();
  if (!files.length) throw new Error('Quanticore payload is missing. Run npm pack from the cli directory.');

  const conflicts = [];
  for (const relative of files) {
    const destination = path.join(targetRoot, relative);
    // Reject links in any existing path component before reading or writing a file.
    let cursor = targetRoot;
    for (const part of relative.split(path.sep)) {
      cursor = path.join(cursor, part);
      const state = await existingPath(cursor);
      if (state?.isSymbolicLink()) { conflicts.push(relative); break; }
    }
    if (conflicts.at(-1) === relative) continue;
    const state = await existingPath(destination);
    if (state) {
      if (!state.isFile()) { conflicts.push(relative); continue; }
      const [current, incoming] = await Promise.all([
        readFile(destination), readFile(path.join(payloadRoot, relative)),
      ]);
      if (!current.equals(incoming)) conflicts.push(relative);
    }
  }
  if (conflicts.length) {
    const error = new Error(`Refusing to overwrite differing files or symbolic links:\n${conflicts.map(file => `  ${file}`).join('\n')}`);
    error.code = 'CONFLICT';
    throw error;
  }

  let copied = 0;
  for (const relative of files) {
    const destination = path.join(targetRoot, relative);
    if (await existingPath(destination)) continue;
    await mkdir(path.dirname(destination), { recursive: true });
    await copyFile(path.join(payloadRoot, relative), destination);
    copied++;
  }
  return { copied, total: files.length, target: targetRoot };
}
