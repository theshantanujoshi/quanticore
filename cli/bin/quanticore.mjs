#!/usr/bin/env node
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { install, PROFILES } from '../lib/install.mjs';

const usage = `Quanticore agent suite

Usage: quanticore-agent-suite [init [target] [--harness NAME ...]]

Install Quanticore skills and agent profiles into an existing local checkout.
Target defaults to the current directory. Supported harnesses:
${Object.keys(PROFILES).sort().join(', ')}

Examples:
  npx quanticore-agent-suite
  npx quanticore-agent-suite init ../my-repo --harness codex
`;

const args = process.argv.slice(2);
if (args.includes('--help') || args.includes('-h')) {
  process.stdout.write(usage);
  process.exit(0);
}
if (args[0] && args[0] !== 'init') {
  process.stderr.write(`Unknown command: ${args[0]}\n\n${usage}`);
  process.exit(2);
}

let target;
const harnesses = [];
for (let index = 1; index < args.length; index++) {
  const arg = args[index];
  if (arg === '--harness') {
    const value = args[++index];
    if (!value || value.startsWith('-')) {
      process.stderr.write('--harness requires a name\n');
      process.exit(2);
    }
    harnesses.push(value);
  } else if (arg.startsWith('-') || target) {
    process.stderr.write(`Unexpected argument: ${arg}\n`);
    process.exit(2);
  } else target = arg;
}

const packageRoot = fileURLToPath(new URL('..', import.meta.url));
try {
  const result = await install({
    target: path.resolve(target ?? '.'),
    harnesses,
    payloadRoot: path.join(packageRoot, 'payload'),
  });
  process.stdout.write(`Installed ${result.copied} files into ${result.target} (${result.total} selected files)\n`);
  process.stdout.write('Open the checkout in your coding agent and ask it to use quanticore-workflow.\n');
} catch (error) {
  process.stderr.write(`${error.message}\n`);
  process.exit(error.code === 'CONFLICT' ? 1 : 2);
}
