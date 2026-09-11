/**
 * Viewport-scoped collection films. Heroes have their own unchanged controller.
 */
(() => {
  'use strict';
  const init = () => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    const small = window.matchMedia('(max-width: 767px)');
    const connection = navigator.connection;
    const states = [];
    let active = null;
    const automatic = () => !reduced.matches && !(connection && connection.saveData);
    const buttonText = (state) => {
      const playing = !state.video.paused || state.pending;
      state.button.textContent = playing ? 'Pause motion' : 'Play motion';
      state.button.setAttribute('aria-label', (playing ? 'Pause' : 'Play') + ' motion: ' + state.label);
    };
    const stop = (state) => {
      state.wanted = false;
      state.video.pause();
      buttonText(state);
    };
    const fail = (state) => {
      state.failed = true;
      state.pending = false;
      stop(state);
      state.frame.classList.remove('is-motion-ready');
      state.button.textContent = 'Still image shown';
      state.button.setAttribute('aria-label', 'Video unavailable. Still image shown: ' + state.label);
      state.button.disabled = true;
    };
    const start = (state) => {
      state.wanted = true;
      if (state.pending || !state.video.paused) return;
      if (!state.video.getAttribute('src')) {
        const slow = connection && /^(slow-2g|2g|3g)$/.test(connection.effectiveType || '');
        state.video.src = (small.matches || slow) ? state.video.dataset.mobileSrc : state.video.dataset.desktopSrc;
        state.video.muted = true;
        state.video.defaultMuted = true;
        state.video.load();
      }
      state.pending = true;
      buttonText(state);
      const attempt = state.video.play();
      Promise.resolve(attempt).then(() => {
        state.pending = false;
        if (!state.wanted || document.hidden || active !== state) state.video.pause();
        buttonText(state);
      }).catch((error) => {
        state.pending = false;
        if (error && error.name === 'AbortError') {
          buttonText(state);
          if (state.wanted && active === state && !document.hidden) start(state);
          return;
        }
        if (error && error.name === 'NotAllowedError') {
          state.manualRequired = true;
          state.wanted = false;
          buttonText(state);
          return;
        }
        fail(state);
      });
    };
    const sync = (preferred) => {
      const candidates = document.hidden ? [] : states.filter((state) =>
        state.ratio > 0 && !state.failed && !state.paused &&
        !state.manualRequired && (automatic() || state.optIn)
      );
      candidates.sort((a, b) => b.ratio - a.ratio);
      const next = candidates.includes(preferred) ? preferred : (candidates[0] || null);
      states.forEach((state) => { if (state !== next && (state.wanted || !state.video.paused)) stop(state); });
      active = next;
      if (next) start(next);
    };
    document.querySelectorAll('[data-collection-scene-motion]').forEach((video) => {
      if (video.dataset.motionInitialized) return;
      const frame = video.closest('.sr2-hero-commerce__frame');
      const button = frame && frame.querySelector('[data-scene-motion-toggle]');
      if (!button) return;
      video.dataset.motionInitialized = 'true';
      const state = {
        video, frame, button, label: video.dataset.sceneLabel || 'Collection scene',
        ratio: 0, wanted: false, pending: false, paused: false,
        optIn: false, manualRequired: false, failed: false
      };
      states.push(state);
      button.hidden = false;
      button.addEventListener('click', () => {
        if (state.failed) return;
        if (!video.paused || state.pending) {
          state.paused = true;
          stop(state);
          sync();
        } else {
          state.paused = false;
          state.optIn = true;
          state.manualRequired = false;
          if (!('IntersectionObserver' in window)) state.ratio = 1;
          sync(state);
        }
      });
      video.addEventListener('playing', () => {
        if (!state.wanted || active !== state || document.hidden) {
          video.pause();
          return;
        }
        frame.classList.add('is-motion-ready');
        buttonText(state);
      });
      video.addEventListener('pause', () => buttonText(state));
      video.addEventListener('error', () => { fail(state); sync(); });
    });
    if (!states.length) return;
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          const state = states.find((item) => item.frame === entry.target);
          if (state) state.ratio = entry.isIntersecting ? entry.intersectionRatio : 0;
        });
        sync();
      }, { threshold: [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1] });
      states.forEach((state) => observer.observe(state.frame));
    }
    const preferencesChanged = () => {
      states.forEach((state) => { state.optIn = false; });
      sync();
    };
    if (reduced.addEventListener) reduced.addEventListener('change', preferencesChanged);
    else if (reduced.addListener) reduced.addListener(preferencesChanged);
    if (connection && connection.addEventListener) connection.addEventListener('change', preferencesChanged);
    document.addEventListener('visibilitychange', () => sync());
    window.addEventListener('pagehide', () => { active = null; states.forEach(stop); });
    window.addEventListener('pageshow', () => sync());
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
