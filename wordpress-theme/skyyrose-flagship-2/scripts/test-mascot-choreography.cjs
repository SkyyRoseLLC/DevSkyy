'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../assets/js/mascot.js'), 'utf8');
// Execute the real entrance function with a deterministic clock and DOM boundary.
const entrance = source.slice(source.indexOf('\tfunction walkOn('), source.indexOf('\tfunction walkOff('));
function boot(reduced = false) {
  const tasks = [], emitted = [], css = {};
  const context = {
    state: 'dormant', prefersReducedMotion: reduced, greetTimer: null,
    window: { skyyRoseMascot3D: { getActionDuration: () => 2.8, getCurrentAction: () => context.currentAction } },
    currentAction: 'Skyy_Idle', document: { hidden: false },
    mascotEl: { style: { setProperty: (k, v) => { css[k] = v; } }, setAttribute() {}, classList: { add() {}, remove() {} } },
    sessionStorage: { getItem: () => null, setItem() {} }, SESSION_KEY_GREETED: 'greeted',
    SCRIPTS: { default: { greeting: { text: 'Hello', chips: [] } } }, context: 'default', mascotConfig: {},
    emitSkyy: name => { emitted.push(name); if (name === 'wave') context.currentAction = 'Skyy_Wave'; },
    isDismissedThisSession: () => false, proactiveSpeak: () => emitted.push('speech'), recordProactiveAppearance() {},
    setTimeout: (fn, ms) => { tasks.push({ fn, ms }); return tasks.length; },
  };
  vm.createContext(context); vm.runInContext(entrance, context);
  return { context, tasks, emitted, css };
}
const greeting = boot(); greeting.context.walkOn(true);
assert.equal(greeting.css['--skyy-walk-duration'], '2200ms');
greeting.tasks.shift().fn();
assert.deepEqual(greeting.emitted, ['walking-in', 'idle', 'wave']);
assert.equal(greeting.tasks[0].ms, 2950, 'Speech must leave time for the full 2.8-second gesture');
greeting.tasks.shift().fn();
assert.ok(!greeting.emitted.includes('speech'), 'A wave still playing cannot be interrupted by speech');
greeting.context.currentAction = 'Skyy_Idle'; greeting.context.document.hidden = true;
greeting.tasks.shift().fn();
assert.ok(!greeting.emitted.includes('speech'), 'Hidden pages must defer proactive speech');
greeting.context.document.hidden = false; greeting.tasks.shift().fn();
assert.equal(greeting.emitted.at(-1), 'speech');
const cancelled = boot(); cancelled.context.walkOn(true); cancelled.context.state = 'exiting'; cancelled.tasks.shift().fn();
assert.deepEqual(cancelled.emitted, ['walking-in'], 'Dismissal during entrance must not resurrect the character');
const reduced = boot(true); reduced.context.walkOn(false);
assert.equal(reduced.tasks[0].ms, 0);
console.log('PASS: entrance timing, complete wave before speech, hidden-page deferral, cancellation, reduced motion.');

// Cross-module regression: retain the pending wave through a slow GLB load.
const runtimeBoot = require('./test-mascot-performance.cjs').boot;
const slowRuntime = runtimeBoot({ delayed: true });
const slow = boot(); slow.context.window = slowRuntime.window;
slow.context.emitSkyy = name => { slow.emitted.push(name); slowRuntime.fire('skyy:' + name); };
slow.context.walkOn(true); slow.tasks.shift().fn(); slow.tasks.shift().fn();
assert.ok(!slow.emitted.includes('speech'), 'Speech cannot replace a wave pending model readiness');
slowRuntime.stats.completeLoad();
assert.equal(slowRuntime.window.skyyRoseMascot3D.getCurrentAction(), 'Skyy_Wave');
slow.tasks.shift().fn();
assert.ok(!slow.emitted.includes('speech'), 'The delayed wave must finish before speech');
slowRuntime.stats.finished({ action: slowRuntime.stats.actions.Skyy_Wave });
slow.tasks.shift().fn(); assert.equal(slow.emitted.at(-1), 'speech');
const failedRuntime = runtimeBoot({ delayed: true });
const failed = boot(); failed.context.window = failedRuntime.window;
failed.context.emitSkyy = name => failedRuntime.fire('skyy:' + name);
failed.context.walkOn(true); failed.tasks.shift().fn(); failedRuntime.stats.failLoad(new Error('offline'));
failed.tasks.shift().fn(); assert.equal(failed.emitted.at(-1), 'speech', 'Failed load must retain static greeting');
assert.equal(failedRuntime.window.skyyRoseMascot3D.isLoading(), false);
failedRuntime.stats.completeLoad();
assert.equal(failedRuntime.window.skyyRoseMascot3D.isReady(), false, 'Late load cannot revive failed rendering');
console.log('PASS: combined slow-load Wave -> Idle -> Speech, failed-load fallback and late-load guard.');

const incompleteRuntime = runtimeBoot({ missingClip: 'Exit' });
const incomplete = boot(); incomplete.context.window = incompleteRuntime.window;
incomplete.context.emitSkyy = name => incompleteRuntime.fire('skyy:' + name);
incomplete.context.walkOn(true); incomplete.tasks.shift().fn(); incomplete.tasks.shift().fn();
assert.equal(incomplete.emitted.at(-1), 'speech', 'Incomplete animation set must use the static greeting');
assert.equal(incompleteRuntime.window.skyyRoseMascot3D.getCurrentAction(), null);
console.log('PASS: incomplete animation set cannot leave a nonplaying wave blocking speech.');
