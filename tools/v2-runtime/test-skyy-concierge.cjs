/** Focused DOM/state contract tests; rendering and WebGL require browser QA. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const theme = path.resolve(__dirname, '../../wordpress-theme/skyyrose-flagship-2');
const { harness, Element } = require('./fixtures/skyy-dom-harness.cjs');
function text(el) {
  return (el.textContent || '') + el.children.map(text).join(' ');
}
function links(el) {
  return el.children.flatMap(child => [...(child.tagName === 'a' ? [child] : []), ...links(child)]);
}
const guide = {
  greeting: 'Welcome to the house.',
  products: [
    { name: 'The Signature Hoodie', sku: 'SG-005', url: '/product/signature-hoodie/', collection: 'Signature' },
  ],
  intents: [
    { id: 'hi', patterns: ['hi'], answer: 'Hello.' },
    {
      id: 'shipping',
      patterns: ['shipping'],
      answer: 'Read the current shipping policy.',
      link: '/shipping/',
      label: 'Shipping information',
    },
    { id: 'unsafe', patterns: ['unsafe'], answer: '<img src=x onerror=alert(1)>', link: 'https://outside.invalid/' },
  ],
  pages: { contact: { label: 'Contact', url: '/contact/' } },
};
test('native dialog opens on invitation, closes and returns focus; modified links remain native', () => {
  const h = harness();
  h.window.SKYY_GUIDE_DATA = guide;
  h.run('mascot.js');
  assert.equal(h.click(h.ids['skyyrose-mascot-recall'], { ctrlKey: true }).defaultPrevented, false);
  h.click(h.ids['skyyrose-mascot-recall']);
  assert.equal(h.ids['skyy-ask-dialog'].open, true);
  assert.equal(h.ids['skyyrose-mascot'].dataset.state, 'walking-in');
  assert.equal(h.document.activeElement, h.ids['skyy-ask-input']);
  h.click(h.ids['skyy-ask-cancel']);
  assert.equal(h.ids['skyyrose-mascot'].dataset.state, 'hidden');
  assert.equal(h.document.activeElement, h.ids['skyyrose-mascot-recall']);
  assert.equal(h.timers.size, 0);
});
test('real SKU discovery supplies exact native URL; phrase boundaries avoid hi matching shipping', () => {
  const h = harness();
  h.window.SKYY_GUIDE_DATA = guide;
  h.run('mascot.js');
  h.window.skyyRoseConcierge.open();
  h.submit('Find SG-005');
  assert.equal(links(h.ids['skyy-conversation']).at(-1).href, 'http://localhost:8899/product/signature-hoodie/');
  assert.match(text(h.ids['skyy-conversation']), /The Signature Hoodie · SG-005/);
  h.submit('shipping');
  assert.match(text(h.ids['skyy-conversation'].children.at(-1)), /Read the current shipping policy/);
  assert.doesNotMatch(text(h.ids['skyy-conversation'].children.at(-1)), /Hello/);
});
test('answer text cannot become markup and external links are excluded; transcript is bounded', () => {
  const h = harness();
  h.window.SKYY_GUIDE_DATA = guide;
  h.run('mascot.js');
  h.window.skyyRoseConcierge.open();
  h.submit('unsafe');
  const answer = h.ids['skyy-conversation'].children.at(-1);
  assert.match(text(answer), /<img src=x/);
  assert.equal(
    answer.children.some(el => el.tagName === 'img'),
    false
  );
  assert.equal(links(answer).length, 0);
  for (let i = 0; i < 30; i++) h.submit('unknown');
  assert.equal(h.ids['skyy-conversation'].children.length, 20);
  assert.equal(h.timers.size, 1);
});
test('loader waits for invitation, deduplicates requests, and failure preserves contact fallback', async () => {
  const h = harness();
  h.window.SKYY_LOADER_CONFIG = { mascotUrl: '/mascot.js', skyy3dUrl: '/skyy-3d.js' };
  h.run('mascot-loader.js');
  assert.equal(h.document.head.children.length, 0);
  h.click(h.ids['skyyrose-mascot-recall']);
  h.click(h.ids['skyyrose-mascot-recall']);
  assert.equal(h.document.head.children.length, 1);
  h.document.head.children[0].onerror();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.ids['skyyrose-mascot-recall'].replacement.href, 'http://localhost:8899/contact/');
  assert.equal(h.timers.size, 0);
});
test('Save-Data and reduced motion keep the canonical fallback without requesting 3D', () => {
  for (const options of [{ reduced: true }, { saveData: true }]) {
    const h = harness(options);
    h.window.SKYY_LOADER_CONFIG = { mascotUrl: '/mascot.js', skyy3dUrl: '/skyy-3d.js' };
    h.run('mascot-loader.js');
    h.document.dispatchEvent({ type: 'skyy:walking-in' });
    assert.equal(h.document.head.children.length, 0);
  }
});
test('unavailable WebGL returns a static character without fetching the model or throwing', async () => {
  const h = harness();
  h.ids['skyyrose-mascot'].dataset.state = 'walking-in';
  h.run('skyy-3d.js');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.window.skyyRoseMascot3D.isReady(), false);
  assert.equal(h.window.skyyRoseMascot3D.getFailureReason(), 'webgl-context');
  assert.equal(h.ids['skyyrose-mascot'].dataset.renderer, 'static');
  assert.equal(h.sprite.style.display, 'block');
});
test('canonical GLB contains the current embedded materials, skin and six populated action clips', () => {
  const bytes = fs.readFileSync(path.join(theme, 'assets/models/skyy-mascot.glb'));
  assert.equal(bytes.toString('ascii', 0, 4), 'glTF');
  const gltf = JSON.parse(bytes.subarray(20, 20 + bytes.readUInt32LE(12)).toString('utf8'));
  assert.ok(
    gltf.extensionsRequired.includes('KHR_draco_mesh_compression'),
    'The canonical asset requires the local Draco decoder'
  );
  assert.ok(gltf.skins.length > 0 && gltf.skins.every(skin => skin.joints.length > 0));
  assert.ok(gltf.images.length > 0 && gltf.images.every(image => Number.isInteger(image.bufferView)));
  for (const name of ['skyy_idle', 'skyy_walk', 'skyy_wave', 'skyy_talk', 'skyy_joy', 'skyy_exit']) {
    const clip = gltf.animations.find(clip => clip.name.toLowerCase() === name);
    assert.ok(clip && clip.channels.length && clip.samplers.length, name);
  }
});

test('ordinary greeting, product discovery and close invoke wave, joy and exit from actual controls', () => {
  const h = harness();
  h.window.SKYY_GUIDE_DATA = guide;
  h.run('mascot.js');
  h.window.skyyRoseConcierge.open();
  const seen = [];
  for (const state of ['wave', 'joy', 'exit', 'hidden'])
    h.document.addEventListener('skyy:' + state, () => seen.push(state));
  h.submit('hello Skyy');
  assert.equal(seen.at(-1), 'wave');
  h.submit('SG-005');
  assert.equal(seen.at(-1), 'joy');
  h.ids['skyyrose-mascot'].dataset.renderer = '3d';
  h.click(h.ids['skyy-ask-cancel']);
  assert.equal(seen.at(-1), 'exit');
  assert.equal(h.ids['skyy-ask-dialog'].open, true);
  [...h.timers.values()].at(-1)();
  assert.equal(seen.at(-1), 'hidden');
  assert.equal(h.ids['skyy-ask-dialog'].open, false);
});
test('a resumed renderer restores its visible Pause control after a lightweight fallback', () => {
  const h = harness();
  h.run('mascot.js');
  h.document.dispatchEvent({ type: 'skyy:3d-ready' });
  assert.equal(h.ids['skyy-motion-toggle'].hidden, false);
  h.document.dispatchEvent({ type: 'skyy:3d-fallback' });
  assert.equal(h.ids['skyy-motion-toggle'].hidden, true);
  h.document.dispatchEvent({ type: 'skyy:3d-visible' });
  assert.equal(h.ids['skyy-motion-toggle'].hidden, false);
  h.click(h.ids['skyy-motion-toggle']);
  assert.equal(h.ids['skyy-motion-toggle'].getAttribute('aria-pressed'), 'true');
});
test('delayed guide completion does not steal focus after the visitor moves elsewhere', async () => {
  const h = harness();
  h.window.SKYY_LOADER_CONFIG = { mascotUrl: '/mascot.js' };
  h.run('mascot-loader.js');
  h.click(h.ids['skyyrose-mascot-recall']);
  let opens = 0;
  h.window.skyyRoseConcierge = {
    open() {
      opens++;
    },
  };
  h.document.activeElement = new Element('input');
  h.document.head.children[0].onload();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(opens, 0);
  assert.equal(h.ids['skyyrose-mascot-recall'].getAttribute('aria-busy'), undefined);
});
test('a late 3D script cannot start after its loader timed out', async () => {
  const h = harness();
  h.window.SKYY_LOADER_CONFIG = { mascotUrl: '/mascot.js', skyy3dUrl: '/skyy-3d.js' };
  h.run('mascot-loader.js');
  h.document.dispatchEvent({ type: 'skyy:walking-in' });
  [...h.timers.values()][0]();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.window.SKYY_3D_CONFIG.loadFailed, true);
  h.ids['skyyrose-mascot'].dataset.state = 'walking-in';
  h.ids['skyy-3d-canvas'].getContext = () => {
    throw new Error('Late renderer must not initialize');
  };
  h.run('skyy-3d.js');
  assert.equal(h.window.skyyRoseMascot3D.isReady(), false);
});

test('Home mounts the same static stage immediately, then defers preparation while offscreen', () => {
  const h = harness({ home: true });
  let prepared = 0;
  h.document.addEventListener('skyy:prepare', () => prepared++);
  h.run('mascot.js');
  assert.equal(h.ids['skyyrose-mascot'].parentElement, h.ids['skyy-hero-stage']);
  assert.equal(prepared, 0);
  assert.equal(h.ids['skyy-ask-dialog'].open, undefined);
  h.intersect(false);
  h.window.skyyRoseConcierge.prepareHome();
  assert.equal(prepared, 0);
  h.intersect(true);
  assert.equal(prepared, 1);
  h.window.skyyRoseMascot3D = { isReady: () => true };
  h.document.dispatchEvent({ type: 'skyy:3d-ready' });
  assert.equal(h.ids['skyyrose-mascot'].dataset.state, 'walking-in');
  assert.equal(h.document.activeElement, h.ids['skyyrose-mascot-recall']);
});
test('Home stage is reparented into conversation and restored without duplicating the character', () => {
  const h = harness({ home: true });
  h.run('mascot.js');
  const stage = h.ids['skyyrose-mascot'];
  h.click(h.ids['skyy-hero-chat']);
  assert.equal(stage.parentElement, h.ids['skyy-dialog-stage']);
  assert.equal(h.ids['skyy-hero-stage'].children.length, 0);
  h.click(h.ids['skyy-ask-cancel']);
  assert.equal(stage.parentElement, h.ids['skyy-hero-stage']);
  assert.equal(h.ids['skyy-dialog-stage'].children.length, 0);
});
test('Home dismissal and offscreen state stop the renderer; header chat remains available', () => {
  const h = harness({ home: true });
  h.run('mascot.js');
  h.intersect(false);
  assert.equal(h.ids['skyyrose-mascot'].dataset.state, 'hidden');
  h.intersect(true);
  h.ids['skyy-hero-dismiss'].focus();
  h.click(h.ids['skyy-hero-dismiss']);
  assert.equal(h.ids['skyyrose-mascot'].hidden, true);
  assert.equal(h.document.activeElement, h.ids['skyyrose-mascot-recall']);
  h.click(h.ids['skyyrose-mascot-recall']);
  assert.equal(h.ids['skyy-ask-dialog'].open, true);
  assert.equal(h.ids['skyyrose-mascot'].hidden, false);
  h.click(h.ids['skyy-ask-cancel']);
  assert.equal(h.ids['skyyrose-mascot'].hidden, true);
});
test('reduced-motion and Save-Data Home retain static portrait without requesting automatic 3D', () => {
  for (const options of [{ reduced: true }, { saveData: true }]) {
    const h = harness({ home: true, ...options });
    let prepared = 0;
    h.document.addEventListener('skyy:prepare', () => prepared++);
    h.run('mascot.js');
    h.window.skyyRoseConcierge.prepareHome();
    assert.equal(prepared, 0);
    assert.equal(h.ids['skyyrose-mascot'].dataset.motionPaused, 'true');
    assert.equal(h.ids['skyyrose-mascot'].parentElement, h.ids['skyy-hero-stage']);
  }
});

test('hero conversation opener is restored after its stage leaves the closed dialog', () => {
  const h = harness({ home: true });
  h.run('mascot.js');
  const opener = h.ids['skyy-hero-chat'];
  opener.focus();
  h.click(opener);
  assert.equal(h.document.activeElement, h.ids['skyy-ask-input']);
  h.click(h.ids['skyy-ask-cancel']);
  assert.equal(opener.parentElement, h.ids['skyyrose-mascot']);
  assert.equal(h.document.activeElement, opener);
});

test('held canonical clips receive real quaternion motion on the same rig without altering source tracks', async () => {
  const vm = require('node:vm');
  const { pathToFileURL } = require('node:url');
  const THREE = await import(pathToFileURL(path.join(theme, 'assets/js/lib/three-r170/three.module.min.js')).href);
  const bytes = fs.readFileSync(path.join(theme, 'assets/models/skyy-mascot.glb'));
  const jsonLength = bytes.readUInt32LE(12),
    gltf = JSON.parse(bytes.subarray(20, 20 + jsonLength));
  const binary = 28 + jsonLength;
  function read(index) {
    const accessor = gltf.accessors[index],
      view = gltf.bufferViews[accessor.bufferView];
    const width = { SCALAR: 1, VEC3: 3, VEC4: 4 }[accessor.type];
    assert.equal(accessor.componentType, 5126);
    return Array.from({ length: accessor.count * width }, (_, i) =>
      bytes.readFloatLE(
        binary +
          (view.byteOffset || 0) +
          (accessor.byteOffset || 0) +
          Math.floor(i / width) * (view.byteStride || width * 4) +
          (i % width) * 4
      )
    );
  }
  const root = new THREE.Group();
  const joints = new Set(gltf.skins.flatMap(skin => skin.joints));
  const nodes = gltf.nodes.map((node, index) => {
    const object = joints.has(index) ? new THREE.Bone() : new THREE.Group();
    object.name = THREE.PropertyBinding.sanitizeNodeName(node.name || `node${index}`);
    if (node.rotation) object.quaternion.fromArray(node.rotation);
    if (node.translation) object.position.fromArray(node.translation);
    if (node.scale) object.scale.fromArray(node.scale);
    return object;
  });
  gltf.nodes.forEach((node, index) => (node.children || []).forEach(child => nodes[index].add(nodes[child])));
  nodes.filter(node => !node.parent).forEach(node => root.add(node));
  const clips = gltf.animations.map(
    animation =>
      new THREE.AnimationClip(
        animation.name,
        -1,
        animation.channels.map(channel => {
          const sampler = animation.samplers[channel.sampler],
            type = channel.target.path;
          const values = read(sampler.output),
            times = read(sampler.input),
            width = values.length / times.length;
          assert(
            values.every((value, i) => Math.abs(value - values[i % width]) < 0.00001),
            'Current source clip is a held pose; do not claim baked motion'
          );
          const property = { rotation: 'quaternion', translation: 'position', scale: 'scale' }[type];
          const Track = type === 'rotation' ? THREE.QuaternionKeyframeTrack : THREE.VectorKeyframeTrack;
          return new Track(nodes[channel.target.node].name + '.' + property, times, values);
        })
      )
  );
  const before = JSON.stringify(clips.map(clip => THREE.AnimationClip.toJSON(clip)));
  const source = fs.readFileSync(path.join(theme, 'assets/js/skyy-3d.js'), 'utf8');
  const helper = source.slice(source.indexOf('  function deriveRigMotion('), source.indexOf('  var required ='));
  const derive = vm.runInNewContext(helper + '\nderiveRigMotion', { Set });
  const motion = derive(THREE, clips, root);
  assert.equal(motion.clips.length, 6);
  assert(motion.source.every(clip => !clip.varying));
  for (const clip of motion.clips) {
    assert.equal(clip.skyyMotionSource, 'runtime-rig-upgrade-v1');
    const mixer = new THREE.AnimationMixer(root),
      action = mixer.clipAction(clip).play();
    mixer.setTime(0.1);
    const first = Object.values(motion.bones).flatMap(bone => bone.quaternion.toArray());
    mixer.setTime(0.32);
    const second = Object.values(motion.bones).flatMap(bone => bone.quaternion.toArray());
    assert(
      second.some((value, i) => Math.abs(value - first[i]) > 0.0001),
      `${clip.name} must visibly change rig pose`
    );
    if (clip.name === 'Skyy_Idle') {
      root.updateMatrixWorld(true);
      const shoulder = motion.bones.upperarml.getWorldPosition(new THREE.Vector3());
      const wrist = motion.bones.handl.getWorldPosition(new THREE.Vector3());
      assert(wrist.y < shoulder.y - 0.1, 'Relaxed hand must be below shoulder, not a forward-held source pose');
    }
    if (clip.name === 'Skyy_Walk') {
      const thighs = ['thighl', 'thighr'];
      for (const name of thighs) {
        mixer.setTime(0.1);
        const a = motion.bones[name].quaternion.clone();
        mixer.setTime(0.5);
        assert(a.angleTo(motion.bones[name].quaternion) > 0.3, `${name} changes across step phases`);
      }
    }
    action.stop();
    mixer.uncacheRoot(root);
  }
  assert.equal(JSON.stringify(clips.map(clip => THREE.AnimationClip.toJSON(clip))), before);
});

test('repeated transient requests renew the actual runtime interval without restarting its clip phase', () => {
  const vm = require('node:vm');
  const source = fs.readFileSync(path.join(theme, 'assets/js/skyy-3d.js'), 'utf8');
  const action = { phase: 2.3 };
  const context = {
    clearTimeout() {},
    timer: null,
    ready: true,
    actions: { skyy_talk: action },
    currentAction: action,
    lightMode: () => false,
    actionElapsed: 2.3,
    actionDuration: 2.4,
  };
  vm.createContext(context);
  vm.runInContext(source.slice(source.indexOf('  function play('), source.indexOf('  async function boot(')), context);
  context.play('skyy_talk', 2400);
  assert.equal(context.actionElapsed, 0);
  assert.equal(context.actionDuration, 2.4);
  assert.equal(action.phase, 2.3);
});

test('WebGL fallback teardown clears rig references and motion diagnostics return null', () => {
  const vm = require('node:vm');
  const source = fs.readFileSync(path.join(theme, 'assets/js/skyy-3d.js'), 'utf8');
  const context = {
    clearTimeout() {},
    timer: null,
    stop() {},
    controller: null,
    draco: null,
    mixer: null,
    model: {},
    rigMotion: { bones: {} },
    disposeModel() {},
    actions: {},
    currentAction: {},
    ready: true,
    renderer: null,
    scene: {},
  };
  vm.createContext(context);
  vm.runInContext(
    source.slice(source.indexOf('  function teardown('), source.indexOf('  function fallback(')),
    context
  );
  context.teardown();
  assert.equal(context.rigMotion, null);
  assert.equal(context.model, null);
  const start = source.indexOf('    getMotionEvidence: function () {');
  const end = source.indexOf('    getFailureReason:', start);
  vm.runInContext('var diagnostic = ({' + source.slice(start, end) + '}).getMotionEvidence;', context);
  assert.equal(context.diagnostic(), null);
});
