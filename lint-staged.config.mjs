/** @type {import('lint-staged').Configuration} */
export default {
  // Python: normalize imports, apply safe lint fixes, format, then reject only
  // findings that cannot be fixed automatically. lint-staged appends paths.
  '*.py': ['isort', 'ruff check --fix', 'black', 'ruff check'],

  // Prettier-supported source and content languages. Shell and TOML support
  // comes from the explicitly pinned plugins in .prettierrc.js.
  '*.{js,jsx,ts,tsx,mjs,cjs,json,jsonc,yaml,yml,md,mdx,css,scss,less,html,htm,graphql,gql,sh,bash,zsh,toml,sql,xml,svg,ipynb}':
    'prettier --write --ignore-unknown',
  '{Dockerfile,**/Dockerfile,.husky/*}': 'prettier --write --ignore-unknown',

  // Root application JS/TS: apply ESLint fixes after Prettier.
  'src/**/*.{ts,tsx,js,jsx,mjs,cjs}': 'eslint --fix --no-error-on-unmatched-pattern',

  // Frontend JS/TS: apply ESLint fixes on staged files only.
  // Do NOT use --max-warnings 0 (242 existing warnings would block every commit)
  // ESLint exits non-zero on errors, zero on warnings-only -- this is correct behavior
  // Must run from frontend/ dir -- root node_modules/eslint has ajv crash (ESLint v9 + @eslint/eslintrc)
  // lint-staged uses execa (no shell) so we use bash -c to enable cd && chain
  'frontend/**/*.{ts,tsx,js,jsx,mjs}': files => {
    const relPaths = files.map(f => f.replace(/^.*\/frontend\//, '').replace(/'/g, "'\\''"));
    const quoted = relPaths.map(p => `'${p}'`).join(' ');
    return `bash -c 'cd frontend && npx eslint --fix ${quoted}'`;
  },

  // Frontend TypeScript type check: whole-project (function prevents file arg appending)
  // tsc ignores tsconfig.json when given individual file arguments on CLI
  'frontend/**/*.{ts,tsx}': () => 'tsc --noEmit --project frontend/tsconfig.json',

  // WordPress PHP: PHPCBF applies every safe WPCS fix before php -l validates
  // syntax. The formatter wrapper accepts PHPCBF's "changes applied" status.
  'wordpress-theme/**/*.php': ['bash scripts/php-format.sh', 'bash scripts/php-lint.sh'],
};
