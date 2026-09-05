const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(__dirname + '/../../wordpress-theme/skyyrose-flagship-2/assets/js/theme.js', 'utf8');
function fixture() {
  const classes = () => { const set = new Set(); return { add: x => set.add(x), contains: x => set.has(x), toggle: (x, on) => on ? set.add(x) : set.delete(x) }; };
  const doc = { handlers: {}, activeElement: null, addEventListener(name, fn) { this.handlers[name] = fn; } };
  const element = () => ({ classList: classes(), attrs: {}, handlers: {}, inert: false, addEventListener(name, fn) { this.handlers[name] = fn; }, setAttribute(k,v) { this.attrs[k] = v; }, getAttribute(k) { return this.attrs[k]; }, focus() { doc.activeElement = this; }, getClientRects() { return [1]; } });
  const root=element(), body=element(), button=element(), first=element(), last=element(), main=element(), footer=element(); footer.inert=true;
  const menu=element(); menu.querySelector=()=>first;menu.querySelectorAll=()=>[];
  const header=element();header.querySelectorAll=()=>[button,first,last];
  const dialogs=[element(),element()]; dialogs.forEach(d=>{d.open=false;d.close=()=>{d.open=false;};});
  Object.assign(doc,{documentElement:root,body,querySelector:s=>({'[data-site-header]':header,'[data-sr2-menu]':button,'[data-sr2-nav]':menu}[s]),querySelectorAll:s=>s==='dialog'?dialogs:s==='dialog[open]'?dialogs.filter(d=>d.open):s.startsWith('main,')?[main,footer]:[]});
  const start=source.indexOf('  const root ='); const end=source.indexOf('  if (header) {');
  assert(start>0&&end>start);
  vm.runInNewContext(source.slice(start,end),{document:doc,window:{matchMedia:()=>({matches:false})},navigator:{},getComputedStyle:()=>({visibility:'visible'})},{codeGeneration:{strings:false,wasm:false}});
  return {doc,body,button,first,last,main,footer,dialogs};
}
test('menu owns focus, preserves prior inert state and restores Escape focus',()=>{
 const f=fixture();f.button.handlers.click();assert.equal(f.main.inert,true);assert.equal(f.doc.activeElement,f.first);assert(f.body.classList.contains('sr2-nav-open'));
 f.last.focus();let prevented=false;f.doc.handlers.keydown({key:'Tab',shiftKey:false,preventDefault(){prevented=true;}});assert(prevented);assert.equal(f.doc.activeElement,f.button);
 f.doc.handlers.keydown({key:'Escape',preventDefault(){}});assert.equal(f.main.inert,false);assert.equal(f.footer.inert,true);assert.equal(f.doc.activeElement,f.button);
});
test('opening a dialog closes menu and other dialog without altering native modal API',()=>{
 const f=fixture();f.button.handlers.click();f.dialogs[0].open=true;f.dialogs[1].handlers.beforetoggle({newState:'open'});assert.equal(f.dialogs[0].open,false);assert.equal(f.main.inert,false);assert.equal(f.button.getAttribute('aria-expanded'),'false');
});
