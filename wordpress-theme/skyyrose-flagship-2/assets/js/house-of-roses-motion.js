/**
 * House of Roses motion controller.
 *
 * Progressive enhancement only: the server-rendered header reel, film poster,
 * transcript, chapter links, and commerce links remain complete without this
 * file. CSS owns composition and transitions; this controller owns state,
 * cancellation, media eligibility, and accessible controls.
 *
 * @package SkyyRose_Flagship_2
 */

(() => {
  'use strict';

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const expandedHeader = window.matchMedia('(min-width: 64em)');
  const filmViewport = window.matchMedia('(min-width: 48em)');
  const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  const controllers = [];
  const focusableSelector = [
    'a[href]',
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[tabindex]',
    '[contenteditable="true"]'
  ].join(',');

  const dataSaving = () => Boolean(connection && connection.saveData);
  const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));

  const listenToMedia = (query, callback, signal) => {
    if (typeof query.addEventListener === 'function') {
      query.addEventListener('change', callback, { signal });
    } else if (typeof query.addListener === 'function') {
      query.addListener(callback);
      signal.addEventListener('abort', () => query.removeListener(callback), { once: true });
    }
  };

  const restoreFocusable = (container) => {
    container.querySelectorAll('[data-house-original-tabindex]').forEach((element) => {
      const original = element.dataset.houseOriginalTabindex;
      if (original === '') {
        element.removeAttribute('tabindex');
      } else {
        element.setAttribute('tabindex', original);
      }
      delete element.dataset.houseOriginalTabindex;
    });
  };

  const makeSlideAvailable = (slide, available) => {
    if (available) {
      slide.removeAttribute('aria-hidden');
      slide.removeAttribute('inert');
      restoreFocusable(slide);
      return;
    }

    slide.setAttribute('aria-hidden', 'true');
    slide.setAttribute('inert', '');
    slide.querySelectorAll(focusableSelector).forEach((element) => {
      if (!Object.prototype.hasOwnProperty.call(element.dataset, 'houseOriginalTabindex')) {
        element.dataset.houseOriginalTabindex = element.getAttribute('tabindex') || '';
      }
      element.setAttribute('tabindex', '-1');
    });
  };

  const initHeaderReel = (root) => {
    if (root.dataset.houseController === 'ready') return;

    const track = root.querySelector('[data-house-header-track]');
    const slides = Array.from(root.querySelectorAll('[data-house-header-slide]'));
    if (!track || slides.length < 2) return;

    const previous = root.querySelector('[data-house-header-prev]');
    const next = root.querySelector('[data-house-header-next]');
    const toggle = root.querySelector('[data-house-header-toggle]');
    const count = root.querySelector('[data-house-header-count]');
    const status = root.querySelector('[data-house-header-status]');
    const abortController = new AbortController();
    const { signal } = abortController;
    const interval = clamp(Number.parseInt(root.dataset.houseHeaderInterval || '6500', 10) || 6500, 5000, 15000);
    let activeIndex = clamp(Number.parseInt(root.dataset.houseHeaderStart || '0', 10) || 0, 0, slides.length - 1);
    let timeout = 0;
    let scrollFrame = 0;
    let userPaused = false;
    let pointerInside = false;
    let focusInside = false;
    let outsideViewport = false;
    let observer = null;

    root.dataset.houseController = 'ready';

    const carouselEligible = () => expandedHeader.matches && !reducedMotion.matches && !dataSaving();

    const describeSlide = (slide, index) => {
      const label = slide.dataset.houseHeaderLabel || slide.getAttribute('aria-label') || `Collection scene ${index + 1}`;
      return `${label}. ${index + 1} of ${slides.length}.`;
    };

    const announce = (source) => {
      if (!status || source === 'auto' || source === 'scroll') return;
      status.textContent = describeSlide(slides[activeIndex], activeIndex);
    };

    const applySlideState = (source = 'sync') => {
      const enhanced = carouselEligible();
      root.dataset.houseHeaderMode = enhanced ? 'carousel' : 'native';
      root.dataset.houseHeaderIndex = String(activeIndex);

      slides.forEach((slide, index) => {
        const active = index === activeIndex;
        slide.classList.toggle('is-active', active);
        slide.dataset.houseHeaderActive = active ? 'true' : 'false';
        makeSlideAvailable(slide, !enhanced || active);
      });

      if (count) {
        count.textContent = `${String(activeIndex + 1).padStart(2, '0')} / ${String(slides.length).padStart(2, '0')}`;
      }

      root.dispatchEvent(new CustomEvent('house:headerchange', {
        bubbles: true,
        detail: {
          index: activeIndex,
          label: slides[activeIndex].dataset.houseHeaderLabel || '',
          source
        }
      }));
      announce(source);
    };

    const state = () => {
      if (!carouselEligible()) return 'static';
      if (document.hidden) return 'paused-hidden';
      if (outsideViewport) return 'paused-offscreen';
      if (userPaused) return 'paused-user';
      if (focusInside) return 'paused-focus';
      if (pointerInside) return 'paused-hover';
      return 'running';
    };

    const clearAdvance = () => {
      if (timeout) window.clearTimeout(timeout);
      timeout = 0;
    };

    const scheduleAdvance = () => {
      clearAdvance();
      if (state() !== 'running') return;
      timeout = window.setTimeout(() => {
        activeIndex = (activeIndex + 1) % slides.length;
        applySlideState('auto');
        syncState();
      }, interval);
    };

    const syncState = () => {
      const current = state();
      root.dataset.houseHeaderState = current;
      if (toggle) {
        toggle.hidden = !carouselEligible();
        toggle.setAttribute('aria-pressed', String(userPaused));
        toggle.textContent = userPaused ? 'Resume collection scenes' : 'Pause collection scenes';
      }
      if (current === 'running') {
        scheduleAdvance();
      } else {
        clearAdvance();
      }
    };

    const scrollNativeTrack = () => {
      if (carouselEligible()) return;
      const slide = slides[activeIndex];
      const left = slide.offsetLeft - track.offsetLeft - Math.max(0, (track.clientWidth - slide.clientWidth) / 2);
      track.scrollTo({ left: Math.max(0, left), behavior: reducedMotion.matches ? 'auto' : 'smooth' });
    };

    const setIndex = (nextIndex, source) => {
      activeIndex = (nextIndex + slides.length) % slides.length;
      applySlideState(source);
      scrollNativeTrack();
      syncState();
    };

    previous?.addEventListener('click', () => setIndex(activeIndex - 1, 'previous'), { signal });
    next?.addEventListener('click', () => setIndex(activeIndex + 1, 'next'), { signal });
    toggle?.addEventListener('click', () => {
      userPaused = !userPaused;
      syncState();
      announce('toggle');
    }, { signal });

    root.addEventListener('pointerenter', () => {
      pointerInside = true;
      syncState();
    }, { signal });
    root.addEventListener('pointerleave', () => {
      pointerInside = false;
      syncState();
    }, { signal });
    root.addEventListener('focusin', () => {
      focusInside = true;
      syncState();
    }, { signal });
    root.addEventListener('focusout', (event) => {
      focusInside = root.contains(event.relatedTarget);
      syncState();
    }, { signal });
    root.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || userPaused || !carouselEligible()) return;
      userPaused = true;
      syncState();
      toggle?.focus();
    }, { signal });

    track.addEventListener('scroll', () => {
      if (carouselEligible() || scrollFrame) return;
      scrollFrame = window.requestAnimationFrame(() => {
        scrollFrame = 0;
        const center = track.scrollLeft + (track.clientWidth / 2);
        let closestIndex = activeIndex;
        let closestDistance = Number.POSITIVE_INFINITY;
        slides.forEach((slide, index) => {
          const slideCenter = slide.offsetLeft + (slide.clientWidth / 2);
          const distance = Math.abs(center - slideCenter);
          if (distance < closestDistance) {
            closestDistance = distance;
            closestIndex = index;
          }
        });
        if (closestIndex !== activeIndex) {
          activeIndex = closestIndex;
          applySlideState('scroll');
        }
      });
    }, { passive: true, signal });

    const preferenceChanged = () => {
      applySlideState('preference');
      syncState();
    };
    listenToMedia(reducedMotion, preferenceChanged, signal);
    listenToMedia(expandedHeader, preferenceChanged, signal);
    connection?.addEventListener?.('change', preferenceChanged, { signal });
    document.addEventListener('visibilitychange', syncState, { signal });

    if ('IntersectionObserver' in window) {
      observer = new IntersectionObserver((entries) => {
        outsideViewport = !entries[0]?.isIntersecting;
        syncState();
      }, { threshold: 0.05 });
      observer.observe(root);
    }

    const cleanup = () => {
      clearAdvance();
      if (scrollFrame) window.cancelAnimationFrame(scrollFrame);
      observer?.disconnect();
      abortController.abort();
      slides.forEach((slide) => makeSlideAvailable(slide, true));
      delete root.dataset.houseController;
      delete root.dataset.houseHeaderMode;
      delete root.dataset.houseHeaderState;
    };

    applySlideState('initial');
    syncState();
    controllers.push(cleanup);
  };

  const initFilm = (root) => {
    if (root.dataset.houseController === 'ready') return;

    const video = root.querySelector('[data-house-film-video]');
    if (!video) return;

    const sources = Array.from(video.querySelectorAll('source[data-src]'));
    const directSource = video.dataset.src || '';
    const toggle = root.querySelector('[data-house-film-toggle]');
    const sound = root.querySelector('[data-house-film-sound]');
    const status = root.querySelector('[data-house-film-status]');
    const chapters = Array.from(root.querySelectorAll('[data-house-film-chapter][data-start]'));
    const abortController = new AbortController();
    const { signal } = abortController;
    let observer = null;
    let loaded = false;
    let inView = false;
    let userPaused = false;
    let completed = false;
    let autoplayAttempted = false;
    let resumeMutedPlayback = false;

    root.dataset.houseController = 'ready';
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.loop = false;
    video.preload = 'none';
    video.removeAttribute('autoplay');

    const hasSource = () => Boolean(sources.some((source) => source.dataset.src) || directSource);
    const mediaEligible = () => filmViewport.matches && !reducedMotion.matches && !dataSaving() && hasSource();

    const setStatus = (value, message = '', announce = false) => {
      root.dataset.houseFilmState = value;
      if (status && announce && message) status.textContent = message;
      root.dispatchEvent(new CustomEvent('house:filmstatechange', {
        bubbles: true,
        detail: { state: value }
      }));
    };

    const syncControls = () => {
      const eligible = mediaEligible();
      if (toggle) {
        toggle.hidden = !eligible;
        toggle.setAttribute('aria-pressed', String(!video.paused && !video.ended));
        toggle.textContent = completed ? 'Replay film' : (video.paused ? 'Play film' : 'Pause film');
      }
      if (sound) {
        sound.hidden = !eligible || !loaded;
        sound.setAttribute('aria-pressed', String(!video.muted));
        sound.textContent = video.muted ? 'Turn sound on' : 'Mute film';
      }
    };

    const unload = () => {
      video.pause();
      sources.forEach((source) => source.removeAttribute('src'));
      video.removeAttribute('src');
      if (loaded) video.load();
      loaded = false;
      resumeMutedPlayback = false;
      root.dataset.houseFilmMode = 'poster';
    };

    const load = () => {
      if (loaded || !mediaEligible()) return false;
      if (sources.length) {
        sources.forEach((source) => {
          if (source.dataset.src) source.src = source.dataset.src;
        });
      } else if (directSource) {
        video.src = directSource;
      }
      loaded = true;
      root.dataset.houseFilmMode = 'video';
      setStatus('loading');
      video.load();
      syncControls();
      return true;
    };

    const play = (sourceType) => {
      if (!mediaEligible()) return;
      load();
      if (completed || video.ended) {
        video.currentTime = 0;
        completed = false;
      }
      if (sourceType !== 'sound') video.muted = sourceType !== 'user-sound';
      video.play().then(() => {
        userPaused = false;
        setStatus('playing', sourceType.startsWith('user') ? 'Skyy Rose Tour previsualization playing.' : '', sourceType.startsWith('user'));
        syncControls();
      }).catch(() => {
        setStatus('poster', 'Film playback is unavailable. The poster, chapters, and collection links remain available.', true);
        syncControls();
      });
    };

    const pause = (reason, announce = false) => {
      if (!video.paused) video.pause();
      setStatus(reason, announce ? 'Skyy Rose Tour previsualization paused.' : '', announce);
      syncControls();
    };

    const updateChapter = () => {
      if (!chapters.length) return;
      const currentTime = video.currentTime || 0;
      let active = 0;
      chapters.forEach((chapter, index) => {
        if (currentTime >= Number.parseFloat(chapter.dataset.start || '0')) active = index;
      });
      chapters.forEach((chapter, index) => {
        const selected = index === active;
        chapter.dataset.houseFilmActive = selected ? 'true' : 'false';
        if (selected) {
          chapter.setAttribute('aria-current', 'true');
        } else {
          chapter.removeAttribute('aria-current');
        }
      });
      root.style.setProperty('--house-film-progress', video.duration ? String(currentTime / video.duration) : '0');
    };

    const maybeAutoplay = () => {
      const requested = root.dataset.houseFilmAutoplay === 'true' || root.dataset.houseFilmAutoplay === 'once';
      if (!requested || autoplayAttempted || !inView || document.hidden || !mediaEligible()) return;
      autoplayAttempted = true;
      play('auto');
    };

    const eligibilityChanged = () => {
      if (!mediaEligible()) {
        unload();
        setStatus('poster');
      } else if (inView) {
        maybeAutoplay();
      }
      syncControls();
    };

    toggle?.addEventListener('click', () => {
      if (!video.paused && !video.ended) {
        userPaused = true;
        pause('paused-user', true);
      } else {
        play('user');
      }
    }, { signal });

    sound?.addEventListener('click', () => {
      if (!loaded || video.paused) return;
      video.muted = !video.muted;
      setStatus(video.muted ? 'playing-muted' : 'playing-sound', video.muted ? 'Film muted.' : 'Film sound on.', true);
      syncControls();
    }, { signal });

    video.addEventListener('loadedmetadata', () => {
      setStatus('ready');
      syncControls();
    }, { signal });
    video.addEventListener('play', syncControls, { signal });
    video.addEventListener('pause', syncControls, { signal });
    video.addEventListener('timeupdate', updateChapter, { passive: true, signal });
    video.addEventListener('ended', () => {
      completed = true;
      userPaused = true;
      setStatus('complete', 'Skyy Rose Tour previsualization complete.', true);
      syncControls();
    }, { signal });
    video.addEventListener('error', () => {
      loaded = false;
      root.dataset.houseFilmMode = 'poster';
      setStatus('error', 'Film playback is unavailable. The poster, chapters, and collection links remain available.', true);
      syncControls();
    }, { signal });

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        resumeMutedPlayback = !video.paused && video.muted && !userPaused;
        pause('paused-hidden');
      } else if (resumeMutedPlayback && inView && mediaEligible()) {
        resumeMutedPlayback = false;
        play('auto-resume');
      }
    }, { signal });

    listenToMedia(reducedMotion, eligibilityChanged, signal);
    listenToMedia(filmViewport, eligibilityChanged, signal);
    connection?.addEventListener?.('change', eligibilityChanged, { signal });

    if ('IntersectionObserver' in window) {
      observer = new IntersectionObserver((entries) => {
        inView = Boolean(entries[0]?.isIntersecting);
        if (inView) {
          if (resumeMutedPlayback && mediaEligible()) {
            resumeMutedPlayback = false;
            play('auto-resume');
          } else {
            maybeAutoplay();
          }
        } else if (!video.paused) {
          resumeMutedPlayback = video.muted && !userPaused;
          pause('paused-offscreen');
        }
      }, { threshold: 0.35 });
      observer.observe(root);
    } else {
      inView = true;
    }

    const cleanup = () => {
      video.pause();
      observer?.disconnect();
      abortController.abort();
      root.style.removeProperty('--house-film-progress');
      delete root.dataset.houseController;
    };

    eligibilityChanged();
    maybeAutoplay();
    controllers.push(cleanup);
  };

  const init = () => {
    document.querySelectorAll('[data-house-header-reel]').forEach(initHeaderReel);
    document.querySelectorAll('[data-house-film]').forEach(initFilm);
  };

  const cleanupAll = () => {
    while (controllers.length) controllers.pop()();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }

  window.addEventListener('pagehide', cleanupAll, { once: true });
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) init();
  });
})();
