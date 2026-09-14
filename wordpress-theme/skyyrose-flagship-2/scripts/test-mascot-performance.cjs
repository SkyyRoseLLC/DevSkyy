'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../assets/js/' + (process.argv[2] || 'skyy-3d.js')), 'utf8');

function boot({ narrow = false, memory = 8, reduced = false, dismissed = false, delayed = false, side = "right", missingClip = null } = {}) {
  const events = new Map();
  const stats = { renders: 0, updates: 0, url: null, loop: null, dpr: null, actions: {}, resets: 0 };
  const canvas = { style: {}, closest: () => null, getAttribute: () => side };
  const doc = {
    hidden: false,
    getElementById: () => canvas,
    querySelector: () => null,
    createElement: () => ({ getContext: () => ({}) }),
    addEventListener(name, fn) { events.set(name, fn); },
    dispatchEvent(event) { events.get(event.type)?.(event); },
  };
  class Vector { constructor() { this.x = 0; this.y = 1; this.z = 0; } set() {} }
  class Box {
    constructor() { this.min = { y: 0 }; }
    setFromObject() { return this; }
    getSize() { return new Vector(); }
    getCenter() { return new Vector(); }
  }
  class Renderer {
    constructor() { this.shadowMap = {}; }
    setPixelRatio(value) { stats.dpr = value; }
    setSize() {}
    setAnimationLoop(fn) { stats.loop = fn; }
    render() { stats.renders++; }
  }
  class Light { constructor() { this.position = new Vector(); } }
  class Camera extends Light { lookAt() {} }
  class Mixer {
    addEventListener(name, handler) { stats[name] = handler; }
    clipAction(clip) {
      const action = { getClip: () => clip, setLoop(mode) { this.mode = mode; return this; } };
      stats.actions[clip.name] = action;
      for (const name of ['play', 'reset', 'fadeIn', 'fadeOut', 'setEffectiveTimeScale', 'setEffectiveWeight']) {
        action[name] = () => action;
      }
      action.reset = () => { stats.resets++; return action; };
      return action;
    }
    update(delta) { stats.updates += delta; }
  }
  class Loader {
    setDRACOLoader() {}
    load(url, ready, progress, fail) {
      stats.failLoad = fail;
      stats.url = url;
      stats.completeLoad = () => ready({
        scene: { rotation: { y: 0 }, scale: { setScalar() {} }, position: { x: 0, y: 0, z: 0 }, traverse() {} },
        animations: ['Idle', 'Walk', 'Wave', 'Talk', 'Joy', 'Exit'].filter(name => name !== missingClip).map(name => ({ name: 'Skyy_' + name, duration: name === 'Wave' ? 2.8 : 1.1 })),
      });
      if (!delayed) stats.completeLoad();
    }
  }
  const window = {
    devicePixelRatio: 3,
    SKYY_3D_CONFIG: { modelUrl: 'desktop.glb', mobileModelUrl: 'mobile.glb', startVisible: true },
    matchMedia: query => ({ matches: query.includes('max-width') ? narrow : reduced }),
    addEventListener() {},
    THREE: {
      LoopOnce: 2200, LoopRepeat: 2201, WebGLRenderer: Renderer, PerspectiveCamera: Camera, AmbientLight: Light,
      DirectionalLight: Light, Vector3: Vector, Box3: Box, AnimationMixer: Mixer,
      Scene: class { add() {} }, Clock: class { getDelta() { return 1 / 120; } },
    },
    THREE_GLTFLoader: Loader,
    THREE_DRACOLoader: class { setDecoderPath() {} },
  };
  vm.runInNewContext(source, {
    window, document: doc, navigator: { deviceMemory: memory },
    sessionStorage: { getItem: () => dismissed ? '1' : null },
    CustomEvent: class { constructor(type) { this.type = type; } },
    setTimeout, clearTimeout, console,
  });
  return { stats, doc, fire: name => events.get(name)?.(), window };
}
const desktop = boot();
assert.equal(desktop.stats.url, 'desktop.glb');
assert.equal(desktop.stats.dpr, 2);
for (let i = 0; i < 120; i++) desktop.stats.loop();
assert.equal(desktop.stats.renders, 30, '120 callbacks should produce 30 GPU renders');
assert.ok(Math.abs(desktop.stats.updates - 1) < 0.001, 'Animation must still advance one real second');
desktop.doc.hidden = true; desktop.fire('visibilitychange');
assert.equal(desktop.stats.loop, null, 'Hidden tab must stop its animation loop');
desktop.doc.hidden = false; desktop.fire('visibilitychange');
assert.equal(typeof desktop.stats.loop, 'function');
desktop.fire('skyy:hidden'); desktop.fire('visibilitychange');
assert.equal(desktop.stats.loop, null, 'A dismissed mascot must not restart on tab focus');
const mobile = boot({ narrow: true });
assert.equal(mobile.stats.url, 'mobile.glb');
assert.equal(mobile.stats.dpr, 1.5);
assert.equal(boot({ memory: 4 }).stats.url, 'mobile.glb');
const stationary = boot({ reduced: true });
assert.equal(stationary.stats.loop, null, 'Reduced motion must not run an animation loop');
assert.equal(stationary.stats.renders, 1);
assert.equal(boot({ dismissed: true }).stats.url, null, 'Dismissed sessions must not fetch the GLB');
const greeting = boot();
greeting.fire('skyy:wave');
assert.equal(greeting.window.skyyRoseMascot3D.getCurrentAction(), 'Skyy_Wave');
assert.equal(greeting.window.skyyRoseMascot3D.getActionDuration('skyy_wave'), 2.8);
assert.equal(greeting.stats.actions.Skyy_Wave.mode, 2200, 'Wave must play exactly once');
const resets = greeting.stats.resets; greeting.fire('skyy:wave');
assert.equal(greeting.stats.resets, resets + 1, 'Replaying a gesture must restart it');
greeting.stats.finished({ action: greeting.stats.actions.Skyy_Walk });
assert.equal(greeting.window.skyyRoseMascot3D.getCurrentAction(), 'Skyy_Wave', 'Stale action completion must not interrupt the wave');
greeting.stats.finished({ action: greeting.stats.actions.Skyy_Wave });
assert.equal(greeting.window.skyyRoseMascot3D.getCurrentAction(), 'Skyy_Idle', 'Only animation completion returns to idle');
const pending = boot({ delayed: true }); pending.fire('skyy:wave'); pending.stats.completeLoad();
assert.equal(pending.window.skyyRoseMascot3D.getCurrentAction(), 'Skyy_Wave');
assert.equal(pending.stats.actions.Skyy_Wave.mode, 2200, 'Gesture requested during download must retain one-shot playback');
console.log('PASS: one-shot completion, replay, pending gesture, stale events; model tiers, DPR, 30 fps pacing, elapsed animation time, tab visibility, dismissal, reduced motion.');

module.exports = { boot };
