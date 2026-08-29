(() => {
  'use strict';

  const root = document.documentElement;
  const body = document.body;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  const header = document.querySelector('[data-site-header]');
  const menuButton = document.querySelector('[data-sr2-menu]');
  const menu = document.querySelector('[data-sr2-nav]');

  root.classList.add('sr2-motion-ready');

  const saveData = Boolean(navigator.connection?.saveData);
  if (reducedMotion || saveData) {
    root.classList.add('sr2-motion-reduced');
  }
  document.querySelectorAll('[data-brand-animation]').forEach((image) => {
    if (reducedMotion || saveData) {
      return;
    }

    const loadAnimation = () => {
      if (image.dataset.brandAnimationLoaded === 'true') return;
      image.dataset.brandAnimationLoaded = 'true';
      image.src = image.dataset.brandAnimation;
    };

    // Below-fold brand motion stays on its tiny still until it approaches the
    // viewport. The header animation remains immediate and layout-stable.
    if (image.dataset.brandAnimationMode === 'viewport' && 'IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        if (!entries[0]?.isIntersecting) return;
        loadAnimation();
        observer.disconnect();
      }, { rootMargin: '320px 0px' });
      observer.observe(image);
      return;
    }
    loadAnimation();
  });

  const setMenu = (open) => {
    if (!menuButton || !menu) return;
    menu.classList.toggle('is-open', open);
    menuButton.setAttribute('aria-expanded', String(open));
    menuButton.setAttribute('aria-label', open ? 'Close site menu' : 'Open site menu');
    body.classList.toggle('sr2-nav-open', open);
  };

  if (menuButton && menu) {
    menuButton.addEventListener('click', () => {
      setMenu(menuButton.getAttribute('aria-expanded') !== 'true');
    });

    menu.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => setMenu(false));
    });

    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || menuButton.getAttribute('aria-expanded') !== 'true') return;
      setMenu(false);
      menuButton.focus();
    });
  }

  if (header) {
    let previousY = window.scrollY;
    let ticking = false;

    const updateHeader = () => {
      const currentY = window.scrollY;
      header.classList.toggle('is-scrolled', currentY > 48);
      header.classList.toggle(
        'is-hidden',
        currentY > previousY && currentY > 500 && !body.classList.contains('sr2-nav-open')
      );
      previousY = currentY;
      ticking = false;
    };

    window.addEventListener('scroll', () => {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(updateHeader);
    }, { passive: true });
  }

  /* Narrative photography can reveal as it enters the viewport. Structural
     headings, commerce cards, and service content must never begin hidden: a
     delayed observer or full-page capture must still render a complete page. */
  const revealTargets = document.querySelectorAll('.sr2-image-reveal');

  if ('IntersectionObserver' in window && !reducedMotion) {
    const revealObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-seen');
        observer.unobserve(entry.target);
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.08 });

    revealTargets.forEach((target) => {
      if (!target.classList.contains('sr2-image-reveal')) target.classList.add('sr2-reveal');
      revealObserver.observe(target);
    });
  } else {
    revealTargets.forEach((target) => target.classList.add('is-seen'));
  }

  /* Collection monuments are one distinct scene per world. Keep the motion
     declarative in CSS, and use this controller exclusively to prevent any
     animation work once the scene is off-screen, the tab is hidden, or the
     visitor explicitly asks for less data or motion. */
  document.querySelectorAll('[data-scene-motion]').forEach((scene) => {
    const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
    let inViewport = false;

    const syncScene = () => {
      const canAnimate = !motionPreference.matches && !saveData && !document.hidden;
      const state = canAnimate && inViewport ? 'running' : (canAnimate ? 'paused' : 'static');
      scene.dataset.sceneState = state;
      scene.dataset.sceneMode = motionPreference.matches ? 'reduced' : (saveData ? 'data-save' : 'cinematic');
    };

    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        inViewport = Boolean(entries[0]?.isIntersecting);
        syncScene();
      }, { rootMargin: '18% 0px', threshold: 0.01 });
      observer.observe(scene);
      window.addEventListener('pagehide', () => observer.disconnect(), { once: true });
    } else {
      // A static scene is the correct progressive fallback; do not start an
      // infinite effect if visibility cannot be measured.
      inViewport = false;
    }

    document.addEventListener('visibilitychange', syncScene);
    if (motionPreference.addEventListener) {
      motionPreference.addEventListener('change', syncScene);
    } else if (motionPreference.addListener) {
      motionPreference.addListener(syncScene);
    }
    syncScene();
  });

  const setupPinnedWorld = (world, rail, chapters, previous, next, count, progress) => {
    const stage = world.querySelector('[data-scroll-world-stage]');
    if (!stage) return false;

    // Scroll World is an expanded-desktop enhancement only. The rail remains a
    // native horizontal scroller at compact and medium widths, where a pinned
    // scene would compromise touch and keyboard reading order.
    const expandedPointer = window.matchMedia('(min-width: 1200px) and (hover: hover) and (pointer: fine)');
    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    let start = 0;
    let distance = 1;
    let frame = 0;
    let layoutFrame = 0;
    let active = false;
    let resizeObserver = null;

    const setPosition = (ratio) => {
      const safeRatio = Math.min(1, Math.max(0, ratio));
      rail.style.transform = `translate3d(${-safeRatio * distance}px, 0, 0)`;
      if (progress) progress.style.transform = `scaleX(${1 + safeRatio * 3})`;
      if (count && chapters.length) {
        const current = Math.min(chapters.length - 1, Math.round(safeRatio * (chapters.length - 1)));
        count.textContent = `${String(current + 1).padStart(2, '0')} / ${String(chapters.length).padStart(2, '0')}`;
      }
    };

    const update = () => {
      frame = 0;
      if (!active || document.hidden) return;
      setPosition((window.scrollY - start) / distance);
    };

    const requestUpdate = () => {
      if (!active || frame) return;
      frame = window.requestAnimationFrame(update);
    };

    const layout = () => {
      layoutFrame = 0;
      if (!active) return;
      const headerHeight = Number.parseFloat(getComputedStyle(root).getPropertyValue('--sr2-header')) || 0;
      const worldTop = world.getBoundingClientRect().top + window.scrollY;
      distance = Math.max(1, rail.scrollWidth - stage.clientWidth);
      start = worldTop + stage.offsetTop - headerHeight;
      world.style.height = `${stage.offsetTop + stage.clientHeight + distance}px`;
      requestUpdate();
    };

    const requestLayout = () => {
      if (!active || layoutFrame) return;
      layoutFrame = window.requestAnimationFrame(layout);
    };

    const goToChapter = (offset) => {
      if (!active) {
        const first = chapters[0];
        const amount = first ? first.getBoundingClientRect().width + 24 : rail.clientWidth * 0.8;
        rail.scrollBy({ left: offset * amount, behavior: 'auto' });
        return;
      }
      const currentRatio = Math.min(1, Math.max(0, (window.scrollY - start) / distance));
      const current = Math.round(currentRatio * (chapters.length - 1));
      const target = Math.min(chapters.length - 1, Math.max(0, current + offset));
      const top = start + (target / Math.max(1, chapters.length - 1)) * distance;
      window.scrollTo({ top, behavior: 'auto' });
    };

    const disable = () => {
      if (!active) return;
      active = false;
      if (frame) window.cancelAnimationFrame(frame);
      if (layoutFrame) window.cancelAnimationFrame(layoutFrame);
      frame = 0;
      layoutFrame = 0;
      setPosition(0);
      rail.style.transform = '';
      world.style.height = '';
      world.classList.remove('is-scroll-world');
    };

    const enable = () => {
      if (active || reducedMotionQuery.matches || !expandedPointer.matches) return;
      active = true;
      world.classList.add('is-scroll-world');
      rail.scrollLeft = 0;
      requestLayout();
    };

    const syncCapability = () => {
      if (reducedMotionQuery.matches || !expandedPointer.matches) {
        disable();
        return;
      }
      enable();
    };

    const onVisibilityChange = () => {
      if (document.hidden) {
        if (frame) window.cancelAnimationFrame(frame);
        frame = 0;
        return;
      }
      requestLayout();
      requestUpdate();
    };

    const onPageHide = () => {
      disable();
      resizeObserver?.disconnect();
    };

    const onPageShow = () => {
      resizeObserver?.observe(stage);
      resizeObserver?.observe(rail);
      syncCapability();
      requestLayout();
    };

    previous?.addEventListener('click', () => goToChapter(-1));
    next?.addEventListener('click', () => goToChapter(1));
    window.addEventListener('scroll', requestUpdate, { passive: true });
    window.addEventListener('resize', requestLayout, { passive: true });
    document.addEventListener('visibilitychange', onVisibilityChange, { passive: true });
    window.addEventListener('pagehide', onPageHide, { passive: true });
    window.addEventListener('pageshow', onPageShow, { passive: true });
    expandedPointer.addEventListener('change', syncCapability);
    reducedMotionQuery.addEventListener('change', syncCapability);

    if ('ResizeObserver' in window) {
      resizeObserver = new ResizeObserver(requestLayout);
      resizeObserver.observe(stage);
      resizeObserver.observe(rail);
    }

    syncCapability();
    return true;
  };

  const setupRail = (rail) => {
    const world = rail.closest('[data-horizontal-world]');
    const previous = world ? world.querySelector('[data-rail-prev]') : null;
    const next = world ? world.querySelector('[data-rail-next]') : null;
    const count = world ? world.querySelector('[data-rail-count]') : null;
    const progress = world ? world.querySelector('[data-rail-progress]') : null;
    const chapters = Array.from(rail.children);
    const storyProgress = rail.parentElement ? rail.parentElement.querySelector('.sr2-world-story__progress span') : null;

    if (world?.hasAttribute('data-scroll-world-pinned') && finePointer && !reducedMotion && window.matchMedia('(min-width: 1200px)').matches) {
      if (setupPinnedWorld(world, rail, chapters, previous, next, count, progress)) return;
    }

    const amount = () => {
      const first = chapters[0];
      return first ? first.getBoundingClientRect().width + 24 : rail.clientWidth * 0.8;
    };

    const updateRail = () => {
      const max = Math.max(1, rail.scrollWidth - rail.clientWidth);
      const ratio = Math.min(1, Math.max(0, rail.scrollLeft / max));
      if (progress) progress.style.transform = `scaleX(${1 + ratio * 3})`;
      if (storyProgress) storyProgress.style.transform = `scaleX(${ratio})`;

      if (count && chapters.length) {
        const center = rail.scrollLeft + rail.clientWidth / 2;
        let current = 0;
        chapters.forEach((chapter, index) => {
          if (chapter.offsetLeft <= center) current = index;
        });
        count.textContent = `${String(Math.min(current + 1, chapters.length)).padStart(2, '0')} / ${String(chapters.length).padStart(2, '0')}`;
      }
    };

    previous?.addEventListener('click', () => rail.scrollBy({ left: -amount(), behavior: reducedMotion ? 'auto' : 'smooth' }));
    next?.addEventListener('click', () => rail.scrollBy({ left: amount(), behavior: reducedMotion ? 'auto' : 'smooth' }));
    rail.addEventListener('scroll', () => window.requestAnimationFrame(updateRail), { passive: true });

    updateRail();
  };

  document.querySelectorAll('[data-horizontal-rail]').forEach(setupRail);

  document.querySelectorAll('[data-interactive-scene]').forEach((scene) => {
    const hotspots = Array.from(scene.querySelectorAll('[data-scene-hotspot]'));
    const cards = Array.from(scene.querySelectorAll('[data-scene-card]'));
    const activate = (index) => {
      hotspots.forEach((item, itemIndex) => item.classList.toggle('is-active', itemIndex === index));
      cards.forEach((item, itemIndex) => item.classList.toggle('is-active', itemIndex === index));
    };
    hotspots.forEach((hotspot, index) => {
      hotspot.addEventListener('mouseenter', () => activate(index));
      hotspot.addEventListener('focus', () => activate(index));
    });
    cards.forEach((card, index) => {
      card.addEventListener('mouseenter', () => activate(index));
      card.addEventListener('focus', () => activate(index));
    });
  });

  const setupProductReel = (card) => {
    const frames = card.querySelectorAll('.sr2-c-product-card__reel-frame, .sr2-c-product-portal__reel-frame');
    if (frames.length < 2 || card.querySelector('[data-portal-hover-insight]') || reducedMotion || !finePointer) return;

    let timer = 0;
    let activeIndex = 0;
    const setFrame = (index) => {
      activeIndex = index % frames.length;
      card.style.setProperty('--sr2-reel-index', String(activeIndex));
    };
    const stop = () => {
      if (timer) window.clearInterval(timer);
      timer = 0;
      card.dataset.reelState = 'idle';
      setFrame(0);
    };
    const play = () => {
      if (timer) return;
      card.dataset.reelState = 'playing';
      setFrame(0);
      timer = window.setInterval(() => setFrame(activeIndex + 1), 1500);
    };

    card.addEventListener('pointerenter', play);
    card.addEventListener('pointerleave', stop);
    card.addEventListener('focusin', play);
    card.addEventListener('focusout', (event) => {
      if (!card.contains(event.relatedTarget)) stop();
    });
  };

  document.querySelectorAll('[data-product-reel]').forEach(setupProductReel);

  /* Product portals never resolve variants in the card. A loop action exists
   * only when PHP has already established that the product is a simple,
   * purchasable, in-stock item. WooCommerce remains the cart authority; this
   * layer merely scopes its lifecycle feedback to the initiating card. */
  const portalQuickAdd = (candidate) => {
    const button = candidate && candidate.jquery ? candidate.get(0) : candidate;
    return button instanceof Element && button.matches('[data-portal-quick-add]') ? button : null;
  };
  const portalQuickAddMessages = window.SKYYROSE2_PORTAL_QUICK_ADD || {};
  const portalQuickAddMessage = (key, fallback) => typeof portalQuickAddMessages[key] === 'string' ? portalQuickAddMessages[key] : fallback;
  const portalQuickAddTimers = new WeakMap();
  const clearPortalQuickAddRecovery = (button) => {
    const timer = portalQuickAddTimers.get(button);
    if (timer) window.clearTimeout(timer);
    portalQuickAddTimers.delete(button);
  };
  const setPortalQuickAddState = (candidate, state, message = '') => {
    const button = portalQuickAdd(candidate);
    if (!button) return;
    const card = button.closest('.sr2-c-product-portal');
    const status = card?.querySelector('[data-portal-action-status]');
    if (!button.dataset.sr2OriginalLabel) button.dataset.sr2OriginalLabel = button.textContent.trim();
    if (!button.dataset.sr2OriginalHref) button.dataset.sr2OriginalHref = button.getAttribute('href') || '';

    if (state === 'adding') {
      button.setAttribute('aria-busy', 'true');
      button.setAttribute('aria-disabled', 'true');
      button.removeAttribute('href');
      button.textContent = portalQuickAddMessage('addingLabel', 'Adding…');
    } else if (state === 'pending') {
      button.setAttribute('aria-busy', 'true');
      button.setAttribute('aria-disabled', 'true');
      button.removeAttribute('href');
      button.textContent = portalQuickAddMessage('pendingLabel', 'Still processing');
    } else {
      clearPortalQuickAddRecovery(button);
      button.removeAttribute('aria-busy');
      button.removeAttribute('aria-disabled');
      if (button.dataset.sr2OriginalHref) button.setAttribute('href', button.dataset.sr2OriginalHref);
      button.textContent = state === 'success' ? portalQuickAddMessage('successLabel', 'Added to bag') : button.dataset.sr2OriginalLabel;
    }

    if (status && message) {
      status.dataset.state = state;
      status.textContent = message;
    }

    if (state === 'adding') {
      clearPortalQuickAddRecovery(button);
      portalQuickAddTimers.set(button, window.setTimeout(() => {
        portalQuickAddTimers.delete(button);
        if (button.getAttribute('aria-busy') === 'true') {
          setPortalQuickAddState(button, 'pending', portalQuickAddMessage('pendingStatus', 'This piece is still processing. Check your bag before trying again.'));
        }
      }, 8000));
    }
  };

  if (window.jQuery) {
    const $ = window.jQuery;
    $(document.body).on('adding_to_cart', (_event, $button) => {
      setPortalQuickAddState($button, 'adding', portalQuickAddMessage('addingStatus', 'Adding this piece to your bag.'));
    });
    $(document.body).on('added_to_cart', (_event, _fragments, _cartHash, $button) => {
      const button = portalQuickAdd($button);
      setPortalQuickAddState(button, 'success', portalQuickAddMessage('successStatus', 'Added to bag. Your bag count is updated.'));
      if (button) {
        window.setTimeout(() => setPortalQuickAddState(button, 'ready', portalQuickAddMessage('readyStatus', 'Ready to add another piece.')), reducedMotion ? 0 : 1800);
      }
    });
  }

  /* Product-card quick view is a progressive layer over the direct PDP link.
   * All facts are copied from the live card payload; the full product page
   * remains the canonical place for options, variation resolution, and cart. */
  const quickView = document.querySelector('[data-quick-view-dialog], #sr2-quick-view-dialog');
  if (quickView && typeof quickView.showModal === 'function') {
    const fields = {
      name: quickView.querySelector('[data-quick-view-name]'),
      collection: quickView.querySelector('[data-quick-view-collection]'),
      price: quickView.querySelector('[data-quick-view-price]'),
      availability: quickView.querySelector('[data-quick-view-availability]'),
      excerpt: quickView.querySelector('[data-quick-view-excerpt]'),
      image: quickView.querySelector('[data-quick-view-image]'),
      media: quickView.querySelector('[data-quick-view-media]'),
      url: quickView.querySelector('[data-quick-view-url]')
    };
    let opener = null;
    const closeQuickView = () => {
      if (quickView.open) quickView.close();
      opener?.focus();
    };
    document.querySelectorAll('[data-quick-view]').forEach((button) => {
      button.addEventListener('click', () => {
        opener = button;
        Object.entries(fields).forEach(([key, field]) => {
          if (!field || key === 'media') return;
          const value = button.dataset[`quickView${key[0].toUpperCase()}${key.slice(1)}`] || '';
          if (key === 'image') {
            field.src = value;
            field.alt = button.dataset.quickViewName || '';
            if (fields.media) fields.media.hidden = !value;
          } else if (key === 'url') {
            field.href = value || '#';
          } else {
            field.textContent = value;
          }
        });
        quickView.showModal();
      });
    });
    quickView.querySelectorAll('[data-quick-view-dismiss]').forEach((button) => button.addEventListener('click', closeQuickView));
    quickView.addEventListener('click', (event) => { if (event.target === quickView) closeQuickView(); });
    quickView.addEventListener('close', () => opener?.focus());
  }

  const sizeGuide = document.querySelector('#sr2-size-guide-dialog');
  if (sizeGuide && typeof sizeGuide.showModal === 'function') {
    let sizeGuideOpener = null;
    const closeSizeGuide = () => {
      if (sizeGuide.open) sizeGuide.close();
      sizeGuideOpener?.focus();
    };
    document.querySelectorAll('[data-size-guide-open]').forEach((button) => {
      button.addEventListener('click', () => {
        sizeGuideOpener = button;
        sizeGuide.showModal();
      });
    });
    sizeGuide.addEventListener('click', (event) => { if (event.target === sizeGuide) closeSizeGuide(); });
    sizeGuide.addEventListener('close', () => sizeGuideOpener?.focus());
  }

  /* Global search remains a native GET form so WordPress owns the results,
   * filters, and indexing. The dialog only removes the blank-query detour. */
  const searchDialog = document.querySelector('#sr2-search-dialog');
  if (searchDialog && typeof searchDialog.showModal === 'function') {
    let searchOpener = null;
    const closeSearch = () => {
      if (searchDialog.open) searchDialog.close();
      searchOpener?.focus();
    };
    document.querySelectorAll('[data-search-open]').forEach((button) => {
      button.addEventListener('click', () => {
        searchOpener = button;
        if (menuButton?.getAttribute('aria-expanded') === 'true') setMenu(false);
        searchDialog.showModal();
        window.requestAnimationFrame(() => searchDialog.querySelector('[data-search-input]')?.focus());
      });
    });
    searchDialog.addEventListener('click', (event) => { if (event.target === searchDialog) closeSearch(); });
    searchDialog.addEventListener('close', () => searchOpener?.focus());
  }

  if (finePointer && !reducedMotion) {
    document.querySelectorAll('[data-depth-card]').forEach((card) => {
      card.addEventListener('pointermove', (event) => {
        const bounds = card.getBoundingClientRect();
        const x = (event.clientX - bounds.left) / bounds.width - 0.5;
        const y = (event.clientY - bounds.top) / bounds.height - 0.5;
        card.style.transform = `perspective(900px) rotateX(${-y * 2.5}deg) rotateY(${x * 2.5}deg) translateY(-2px)`;
      });
      card.addEventListener('pointerleave', () => {
        card.style.transform = '';
      });
    });

    document.querySelectorAll('[data-hero-depth]').forEach((hero) => {
      const media = hero.querySelector('img');
      if (!media) return;
      hero.addEventListener('pointermove', (event) => {
        const x = event.clientX / window.innerWidth - 0.5;
        const y = event.clientY / window.innerHeight - 0.5;
        media.style.transform = `scale(1.025) translate(${x * -8}px, ${y * -6}px)`;
      });
      hero.addEventListener('pointerleave', () => {
        media.style.transform = '';
      });
    });
  }

  /* Cinematic hero video: opt-in enhancement over the poster/static hero. */
  const heroVideo = document.querySelector('[data-hero-video]');
  const heroNav = document.querySelector('.sr-home__hero-nav');
  if (heroVideo) {
    const heroSource = heroVideo.querySelector('source[data-src]');
    // The responsive poster is the mobile source of truth. Do not fetch or
    // initialize the cinematic video below the matching CSS breakpoint.
    const canPlayHero = !reducedMotion && !saveData && window.matchMedia('(min-width: 781px)').matches;
    const FADE_MS = 500;
    let fadeFrame = 0;
    let fadeStart = 0;
    let fadeFrom = 0;
    let fadeTarget = 0;

    const fadeTo = (target) => {
      if (reducedMotion || saveData) {
        heroVideo.style.opacity = String(target);
        return;
      }
      fadeStart = performance.now();
      fadeFrom = Number.parseFloat(heroVideo.style.opacity || getComputedStyle(heroVideo).opacity) || 0;
      fadeTarget = target;
      if (fadeFrame) return;
      const tick = (now) => {
        const progress = Math.min(1, (now - fadeStart) / FADE_MS);
        const eased = 1 - ((1 - progress) ** 3);
        const lerped = fadeFrom + ((fadeTarget - fadeFrom) * eased);
        heroVideo.style.opacity = String(lerped);
        if (progress < 1) {
          fadeFrame = window.requestAnimationFrame(tick);
        } else {
          fadeFrame = 0;
        }
      };
      fadeFrame = window.requestAnimationFrame(tick);
    };

    const stopHero = () => {
      if (fadeFrame) window.cancelAnimationFrame(fadeFrame);
      fadeFrame = 0;
      heroVideo.pause();
      heroVideo.style.opacity = '1';
    };

    if (!canPlayHero || !heroSource) {
      stopHero();
    } else {
      heroSource.src = heroSource.dataset.src;
      heroVideo.addEventListener('loadeddata', () => {
        heroVideo.play().then(() => fadeTo(1)).catch(() => {});
      }, { once: true });
      heroVideo.addEventListener('error', stopHero, { once: true });
      heroVideo.addEventListener('timeupdate', () => {
        if (Number.isFinite(heroVideo.duration) && heroVideo.duration - heroVideo.currentTime <= 0.55) fadeTo(0);
      });
      heroVideo.addEventListener('ended', () => {
        heroVideo.style.opacity = '0';
        heroVideo.currentTime = 0;
        window.setTimeout(() => heroVideo.play().then(() => fadeTo(1)).catch(() => {}), 100);
      });
      heroVideo.load();
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) stopHero();
      });
      window.addEventListener('pagehide', stopHero, { once: true });
    }
  }

  /* V2 collection model loop. CSS owns the seamless track; JavaScript adds
     explicit user, visibility, and viewport pause states without taking over
     scrolling or collection navigation. */
  const heroModelLoop = document.querySelector('[data-home-model-loop]');
  if (heroModelLoop) {
    const loopToggle = heroModelLoop.querySelector('[data-home-model-toggle]');
    const desktopMotion = window.matchMedia('(min-width: 781px) and (prefers-reduced-motion: no-preference)');
    let userPaused = false;
    let outsideViewport = false;

    const motionAllowed = () => desktopMotion.matches && !saveData;
    const syncModelLoop = () => {
      const canMove = motionAllowed();
      const paused = !canMove || userPaused || outsideViewport || document.hidden;
      heroModelLoop.dataset.motion = canMove ? 'continuous' : 'static';
      heroModelLoop.dataset.loopState = paused ? 'paused' : 'running';

      if (canMove) {
        heroModelLoop.dataset.enhanced = 'true';
      } else {
        delete heroModelLoop.dataset.enhanced;
      }

      if (loopToggle) {
        loopToggle.setAttribute('aria-pressed', userPaused ? 'true' : 'false');
        loopToggle.textContent = userPaused ? 'Resume rotation' : 'Pause rotation';
      }
    };

    loopToggle?.addEventListener('click', () => {
      userPaused = !userPaused;
      syncModelLoop();
    });
    heroModelLoop.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && !userPaused) {
        userPaused = true;
        syncModelLoop();
        loopToggle?.focus();
      }
    });
    document.addEventListener('visibilitychange', syncModelLoop);
    desktopMotion.addEventListener?.('change', syncModelLoop);

    if ('IntersectionObserver' in window) {
      const modelLoopObserver = new IntersectionObserver((entries) => {
        outsideViewport = !entries[0]?.isIntersecting;
        syncModelLoop();
      }, { threshold: 0.05 });
      modelLoopObserver.observe(heroModelLoop);
      window.addEventListener('pagehide', () => modelLoopObserver.disconnect(), { once: true });
    }

    syncModelLoop();
  }

  if (heroNav) {
    const updateHeroNav = () => heroNav.classList.toggle('is-scrolled', window.scrollY > 100);
    updateHeroNav();
    window.addEventListener('scroll', updateHeroNav, { passive: true });
  }

  /* WooCommerce owns variation resolution and cart writes. This adapter only
     reflects confirmed form events as V2 state/status; it never calculates
     price, stock, or a variation client-side. */
  const pdpStatus = document.querySelector('[data-sr2-pdp-status]');
  const setPdpStatus = (form, state, message = '') => {
    if (form) form.dataset.sr2VariationState = state;
    if (pdpStatus) {
      pdpStatus.dataset.state = state;
      pdpStatus.textContent = message;
    }
  };

  if (window.jQuery) {
    const $ = window.jQuery;
    $('.variations_form').each(function attachVariationState() {
      const form = this;
      setPdpStatus(form, 'incomplete', 'Select options to see the current piece availability.');
      $(form).on('found_variation', (_event, variation) => {
        const available = variation?.is_in_stock !== false && variation?.is_purchasable !== false;
        setPdpStatus(
          form,
          available ? 'valid' : 'unavailable',
          available ? 'Selection confirmed. Current price and availability are shown above.' : 'This selection is unavailable. Choose another option.'
        );
      });
      $(form).on('hide_variation reset_data', () => {
        setPdpStatus(form, 'incomplete', 'Select options to see the current piece availability.');
      });
      $(form).on('woocommerce_variation_has_changed', () => {
        if (form.dataset.sr2VariationState !== 'valid') setPdpStatus(form, 'resolving', 'Checking this selection.');
      });
    });
  }

  document.querySelectorAll('.single_add_to_cart_button, form.cart button[type="submit"]').forEach((button) => {
    const form = button.closest('form.cart');
    if (!form) return;
    const restoreCartButton = () => {
      button.removeAttribute('aria-busy');
      if (button.dataset.sr2OriginalLabel) button.textContent = button.dataset.sr2OriginalLabel;
    };
    form.addEventListener('submit', () => {
      if (button.disabled || button.getAttribute('aria-busy') === 'true') return;
      button.setAttribute('aria-busy', 'true');
      button.dataset.sr2OriginalLabel = button.textContent;
      button.textContent = 'Adding…';
    });
    // WooCommerce emits its cart lifecycle through jQuery when that runtime is
    // present. Keep a native listener as a progressive fallback for a custom
    // cart integration, but never assume one event transport for both cases.
    if (window.jQuery) {
      window.jQuery(document.body).on('added_to_cart wc_fragments_refreshed', restoreCartButton);
    } else {
      document.body.addEventListener('added_to_cart', restoreCartButton);
      document.body.addEventListener('wc_fragments_refreshed', restoreCartButton);
    }
  });

  const heroHeadline = document.querySelector('[data-hero-headline]');
  if (heroHeadline && !reducedMotion) {
    const words = heroHeadline.textContent.trim().split(/\s+/).filter(Boolean);
    const fragment = document.createDocumentFragment();
    words.forEach((word, index) => {
      const wordElement = document.createElement('span');
      wordElement.className = 'sr-home__hero-word';
      wordElement.style.setProperty('--word-delay', `${index * 100}ms`);
      wordElement.textContent = word;
      fragment.append(wordElement);
      if (index < words.length - 1) fragment.append(document.createTextNode(' '));
    });
    heroHeadline.replaceChildren(fragment);
  }

  const bayMap = document.querySelector('[data-bay-map]');
  if (bayMap) {
    const stops = Array.from(bayMap.querySelectorAll('[data-bay-stop]'));
    const status = bayMap.querySelector('[data-bay-status]');
    const labels = {
      oakland: 'Oakland · The root',
      'san-francisco': 'San Francisco · The fog',
      'san-jose': 'San Jose · The night',
    };
    const lightStop = (stop, index) => {
      window.setTimeout(() => {
        stop.classList.add('is-lit');
        if (status) status.textContent = labels[stop.dataset.bayStop] || `Chapter ${index + 1}`;
      }, reducedMotion ? 0 : index * 650);
    };
    const lightAll = () => {
      bayMap.classList.add('is-active');
      stops.forEach(lightStop);
    };
    if (reducedMotion || !('IntersectionObserver' in window)) {
      lightAll();
    } else {
      const mapObserver = new IntersectionObserver((entries, observer) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        lightAll();
        observer.disconnect();
      }, { threshold: 0.35 });
      mapObserver.observe(bayMap);
    }
    stops.forEach((stop, index) => stop.addEventListener('focus', () => {
      stop.classList.add('is-lit');
      if (status) status.textContent = labels[stop.dataset.bayStop] || `Chapter ${index + 1}`;
    }));
  }
})();
