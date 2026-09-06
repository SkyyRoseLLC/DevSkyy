/** Approved heroes and native scroll-world choreography. No scroll interception. */
(() => {
  'use strict';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const connection = navigator.connection;
  const heroes = [];
  let suspended = false;
  const allowed = () => !reduced.matches && !connection?.saveData;
  const sync = (state) => {
    const active = allowed() && state.visible && !state.paused && !document.hidden && !suspended;
    state.el.dataset.recoveryMotion = active ? 'running' : 'paused';
    state.button.hidden = !allowed();
    state.button.textContent = state.paused ? 'Play motion' : 'Pause motion';
    state.button.setAttribute('aria-pressed', String(state.paused));
    if (!state.video || state.failed || !state.posterReady) return;
    if (!active) { state.video.pause(); return; }
    if (!state.loaded) {
      state.video.querySelectorAll('source[data-src]').forEach(source => { source.src = source.dataset.src; });
      state.video.muted = true;
      state.video.load();
      state.loaded = true;
    }
    if (state.pending || !state.video.paused) return;
    state.pending = true;
    Promise.resolve(state.video.play()).then(() => {
      state.pending = false;
      if (!allowed() || !state.visible || state.paused || document.hidden || suspended) state.video.pause();
    }).catch(error => {
      state.pending = false;
      if (error.name === 'AbortError') return;
      if (error.name === 'NotAllowedError') { state.paused = true; sync(state); return; }
      state.failed = true;
      state.el.classList.remove('is-hero-video-ready');
    });
  };
  const observer = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const state = heroes.find(item => item.el === entry.target);
      if (state) { state.visible = entry.isIntersecting; sync(state); }
    });
  }, { threshold: 0.12 }) : null;
  document.querySelectorAll('[data-recovery-hero]').forEach(el => {
    const button = el.querySelector('[data-recovery-motion-toggle]');
    if (!button) return;
    const video = el.querySelector('[data-recovery-hero-video]');
    const state = { el, button, video, paused: false, visible: !observer, loaded: false, pending: false, failed: false, posterReady: !video };
    heroes.push(state);
    button.addEventListener('click', () => { state.paused = !state.paused; sync(state); });
    if (video) {
      // Render the responsive approved image before competing for video bytes.
      // The matching video poster also stays visible until the first frame.
      const poster = el.querySelector('img');
      const prepare = () => {
        if (poster?.currentSrc) video.poster = poster.currentSrc;
        state.posterReady = true;
        sync(state);
      };
      if (poster?.decode) poster.decode().then(prepare, prepare);
      else prepare();
      video.addEventListener('playing', () => el.classList.add('is-hero-video-ready'));
      video.addEventListener('error', () => { state.failed = true; video.pause(); el.classList.remove('is-hero-video-ready'); });
    }
    observer?.observe(el);
    sync(state);
  });
  const syncAll = () => { document.documentElement.dataset.recoveryMotionAllowed = String(allowed()); heroes.forEach(sync); };
  reduced.addEventListener('change', syncAll);
  connection?.addEventListener?.('change', syncAll);
  document.addEventListener('visibilitychange', syncAll);
  window.addEventListener('pagehide', () => { suspended = true; observer?.disconnect(); syncAll(); });
  window.addEventListener('pageshow', () => { suspended = false; heroes.forEach(state => observer?.observe(state.el)); syncAll(); });
  syncAll();

  document.querySelectorAll('[data-recovery-rail]').forEach(rail => {
    const track = rail.querySelector('[data-recovery-track]');
    if (!track) return;
    const items = [...track.children];
    const previous = rail.querySelector('[data-recovery-prev]');
    const next = rail.querySelector('[data-recovery-next]');
    const count = rail.querySelector('[data-recovery-count]');
    let index = 0;
    let frame = 0;
    const update = () => {
      frame = 0;
      const bounds = track.getBoundingClientRect();
      let distance = Infinity;
      items.forEach((item, i) => {
        const rect = item.getBoundingClientRect();
        const offset = rect.left - bounds.left;
        if (Math.abs(offset) < distance) { distance = Math.abs(offset); index = i; }
      });
      previous.disabled = index === 0;
      next.disabled = index === items.length - 1;
      const label = `${String(index + 1).padStart(2, '0')} / ${String(items.length).padStart(2, '0')}`;
      if (count.textContent !== label) count.textContent = label;
    };
    const move = target => {
      const item = items[Math.max(0, Math.min(items.length - 1, target))];
      const offset = item.getBoundingClientRect().left - track.getBoundingClientRect().left;
      track.scrollBy({ left: offset, behavior: allowed() ? 'smooth' : 'instant' });
    };
    previous.addEventListener('click', () => move(index - 1));
    next.addEventListener('click', () => move(index + 1));
    track.addEventListener('keydown', event => {
      if (event.target !== track || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      move(event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : index + (event.key === 'ArrowRight' ? 1 : -1));
    });
    const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
    track.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule, { passive: true });
    reduced.addEventListener('change', schedule);
    connection?.addEventListener?.('change', schedule);
    window.addEventListener('pagehide', () => { cancelAnimationFrame(frame); frame = 0; });
    window.addEventListener('pageshow', schedule);
    rail.querySelector('.sr2-recovery-controls').hidden = false;
    update();
  });
})();
