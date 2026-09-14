import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import postcss from 'postcss';
import * as prettier from 'prettier';
import { expect, test } from 'vitest';

const root = process.cwd();
const source = resolve(root, 'wordpress-theme/skyyrose-flagship/assets/css/design-tokens.css');
const generator = resolve(root, 'wordpress-theme/skyyrose-flagship/data/gen-design-tokens.py');

function declarations(css: string): string[] {
  const values: string[] = [];
  postcss.parse(css).walkDecls(decl => {
    const value = decl.value
      .replace(/#[0-9a-f]{3,8}\b/gi, hex => {
        const digits = hex.slice(1).toLowerCase();
        return `#${digits.length <= 4 ? [...digits].map(char => char.repeat(2)).join('') : digits}`;
      })
      .replace(/(['"])([^'"]*)\1/g, '$2');
    values.push(`${decl.parent?.type === 'rule' ? decl.parent.selector : ''}|${decl.prop}|${value}`);
  });
  return values;
}

test('token generation survives source formatting with identical CSS values', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'skyyrose-token-format-'));
  const target = join(directory, 'design-tokens.css');
  try {
    const before = readFileSync(source, 'utf8');
    writeFileSync(target, before);
    execFileSync('python3', [generator], {
      cwd: root,
      env: { ...process.env, SKYYROSE_TOKENS_CSS: target },
    });
    const generated = readFileSync(target, 'utf8');
    const options = await prettier.resolveConfig(source);
    const formatted = await prettier.format(generated, { ...options, filepath: source });
    expect(formatted).toBe(generated);
    expect(declarations(generated)).toEqual(declarations(before));
    writeFileSync(target, formatted);
    execFileSync('python3', [generator], {
      cwd: root,
      env: { ...process.env, SKYYROSE_TOKENS_CSS: target },
    });
    expect(readFileSync(target, 'utf8')).toBe(formatted);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
