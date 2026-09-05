/** Existing runtime CSS is the Phase 2 primitive authority; sync only mapped editor settings. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const theme = path.resolve(__dirname, '../../wordpress-theme/skyyrose-flagship-2');
const css = fs.readFileSync(path.join(theme, 'assets/css/design-tokens.css'), 'utf8');
const root = css.match(/:root\s*\{([^}]+)\}/)[1];
const tokens = Object.fromEntries([...root.matchAll(/--sr2-([\w-]+):\s*([^;]+);/g)].map(m => [m[1], m[2].trim()]));
const token = name => { assert(tokens[name], `Missing token ${name}`); return tokens[name]; };
const file = path.join(theme, 'theme.json');
const current = JSON.parse(fs.readFileSync(file, 'utf8'));
const next = structuredClone(current);
const colors = {'rose-gold':'rose','signature-gold':'gold','black-rose-silver':'silver','love-hurts-crimson':'crimson'};
for (const [slug, key] of Object.entries(colors)) {
  const entry = next.settings.color.palette.find(p => p.slug === slug); assert(entry, slug); entry.color = token(key).toUpperCase();
}
const spacing = {'2-xs':'space-1','xs':'space-2','small':'space-4','medium':'space-6','large':'space-10','xl':'space-16','2-xl':'space-24'};
for (const [slug,key] of Object.entries(spacing)) {
  const entry=next.settings.spacing.spacingSizes.find(p=>p.slug===slug); assert(entry,slug); entry.size=token(key);
}
const fonts = {'archivo':'font-display','hanken-grotesk':'font-body','anton':'font-ui','cinzel':'font-caps'};
for (const [slug,key] of Object.entries(fonts)) {
  const entry=next.settings.typography.fontFamilies.find(p=>p.slug===slug);assert(entry,slug);entry.fontFamily=token(key);
}
Object.assign(next.settings.custom.motion, {'house-ease':token('ease'),fast:token('fast'),standard:token('normal'),slow:token('slow')});
next.settings.custom.layers={guide:token('layer-guide'),header:token('layer-header'),skip:token('layer-skip')};
if (process.argv.includes('--write')) {
  if (JSON.stringify(current)!==JSON.stringify(next)) fs.writeFileSync(file,JSON.stringify(next,null,2)+'\n');
} else {
  assert.deepEqual(current,next,'Editor tokens drift from runtime contract; run build:tokens');
}
console.log('PASS runtime/editor collection colors, spacing, typography, motion and layer contract');
