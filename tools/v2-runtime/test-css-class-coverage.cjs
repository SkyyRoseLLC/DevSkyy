const test = require('node:test');
const assert = require('node:assert/strict');

const coverage = import('./css-class-coverage.mjs');

test('matches exact ordinary class selectors and rejects longer class tokens', async () => {
  const { covered } = await coverage;
  assert.equal(covered('.sr2-hero{display:block}', 'sr2-hero'), true);
  for (const suffix of ['-child', '_child', '2', 'x']) {
    assert.equal(covered(`.sr2-hero${suffix}{}`, 'sr2-hero'), false);
  }
});

test('regex metacharacters in delivered class tokens are matched literally', async () => {
  const { covered } = await coverage;
  for (const token of [
    'sr2-x.y',
    'sr2-x+y',
    'sr2-x*y',
    'sr2-x?y',
    'sr2-x(y)',
    'sr2-x[y]',
    'sr2-x{2}',
    'sr2-x|y',
    'sr2-x^y',
    'sr2-x$y',
  ]) {
    assert.equal(covered(`.${token}{color:red}`, token), true, token);
    assert.equal(covered('.sr2-xxy{color:red}', token), false, token);
  }
});

test('backslashes cannot become regex escapes or make the expression invalid', async () => {
  const { covered } = await coverage;
  for (const token of [String.raw`sr2-\d`, String.raw`sr2-\b`, 'sr2-end\\']) {
    assert.equal(covered(`.${token}{}`, token), true, token);
    assert.equal(covered('.sr2-7{} .sr2-end{}', token), false, token);
  }
});
