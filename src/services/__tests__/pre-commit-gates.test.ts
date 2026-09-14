import { execFileSync, spawnSync } from 'node:child_process';
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';

const root = path.resolve(import.meta.dirname, '../../..');
const fixtures: string[] = [];

function fixture(python = true) {
  const cwd = mkdtempSync(path.join(tmpdir(), 'devskyy-pre-commit-'));
  fixtures.push(cwd);
  const git = (...args: string[]) => execFileSync('git', args, { cwd, encoding: 'utf8' });
  git('init', '-q');
  const write = (file: string, contents: string) => {
    mkdirSync(path.dirname(path.join(cwd, file)), { recursive: true });
    writeFileSync(path.join(cwd, file), contents);
  };
  write(python ? 'staged.py' : 'staged.txt', 'fixture\n');
  git('add', '.');
  mkdirSync(path.join(cwd, 'tests/unit'), { recursive: true });
  for (const tool of ['npx', 'mypy', 'pytest']) {
    write(
      `bin/${tool}`,
      `#!/bin/sh\nprintf '%s\\n' '${tool}' >> "$TRACE"\nif [ "$FAIL_STAGE" = '${tool}' ]; then exit 23; fi\n`
    );
    chmodSync(path.join(cwd, `bin/${tool}`), 0o755);
  }
  write('scripts/freshness-guard.sh', '#!/bin/sh\nprintf "%s\\n" freshness >> "$TRACE"\n');
  copyFileSync(path.join(root, '.husky/pre-commit'), path.join(cwd, 'pre-commit'));
  const run = (failStage = '') => {
    const trace = path.join(cwd, 'trace');
    const result = spawnSync('sh', ['pre-commit'], {
      cwd,
      encoding: 'utf8',
      env: {
        ...process.env,
        PATH: `${path.join(cwd, 'bin')}:${process.env.PATH}`,
        TRACE: trace,
        FAIL_STAGE: failStage,
      },
    });
    return { status: result.status, stages: readFileSync(trace, 'utf8').trim().split('\n') };
  };
  return { cwd, git, write, run };
}

afterEach(() => {
  for (const cwd of fixtures.splice(0)) rmSync(cwd, { recursive: true, force: true });
});

describe('executable pre-commit gates (offline fake tools)', () => {
  it.each([
    ['npx', ['npx']],
    ['mypy', ['npx', 'mypy']],
    ['pytest', ['npx', 'mypy', 'pytest']],
  ])('stops immediately after %s fails', (stage, stages) => {
    expect(fixture().run(stage as string)).toEqual({ status: 23, stages });
  });

  it('runs all gates after successful Python checks', () => {
    expect(fixture().run()).toEqual({ status: 0, stages: ['npx', 'mypy', 'pytest', 'freshness'] });
  });

  it('keeps Python gates conditional on staged Python files', () => {
    expect(fixture(false).run()).toEqual({ status: 0, stages: ['npx', 'freshness'] });
  });

  it('allows normal staging of only the maintained build source', () => {
    const repo = fixture(false);
    copyFileSync(path.join(root, '.gitignore'), path.join(repo.cwd, '.gitignore'));
    const source = 'skyyrose/build/tool-calling.js';
    repo.write(source, 'export {};\n');
    repo.git('add', '--', source);
    expect(repo.git('ls-files', '--', source).trim()).toBe(source);
    for (const file of ['skyyrose/build/generated.js', 'skyyrose/build/nested/output.js', 'build/output.js']) {
      repo.write(file, 'generated\n');
      expect(repo.git('check-ignore', '--', file).trim()).toBe(file);
    }
  });
});
