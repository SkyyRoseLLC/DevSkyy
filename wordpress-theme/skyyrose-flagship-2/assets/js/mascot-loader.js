/** The invitation is server-rendered. Guide and 3D load only when invited. */
(function () {
  'use strict';
  var config = window.SKYY_LOADER_CONFIG || {};
  var invite = document.getElementById('skyyrose-mascot-recall');
  if (!invite || !config.mascotUrl) return;
  var pending;
  var threePending;
  var home = document.getElementById('skyy-hero-stage');
  function loadGuide() {
    if (!pending) pending = script(config.mascotUrl);
    return pending;
  }
  function localUrl(value) {
    try {
      var url = new URL(value, location.href);
      return url.origin === location.origin && /^https?:$/.test(url.protocol) ? url.href : '';
    } catch (_) {
      return '';
    }
  }
  function script(value) {
    return new Promise(function (resolve, reject) {
      var src = localUrl(value);
      if (!src) {
        reject(new Error('Skyy script must be local'));
        return;
      }
      var el = document.createElement('script');
      var timer = setTimeout(function () {
        finish(new Error('Skyy script timed out'));
      }, 15000);
      function finish(error) {
        clearTimeout(timer);
        el.onload = el.onerror = null;
        if (error) {
          el.remove();
          reject(error);
        } else resolve();
      }
      el.async = true;
      el.src = src;
      el.onload = function () {
        finish();
      };
      el.onerror = function () {
        finish(new Error('Skyy script unavailable'));
      };
      document.head.appendChild(el);
    });
  }
  function loadThree() {
    if (
      threePending ||
      !config.skyy3dUrl ||
      (navigator.connection && navigator.connection.saveData) ||
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    )
      return;
    window.SKYY_3D_CONFIG = window.SKYY_3D_CONFIG || {};
    window.SKYY_3D_CONFIG.startVisible = true;
    threePending = script(config.skyy3dUrl).catch(function () {
      window.SKYY_3D_CONFIG.loadFailed = true;
      document.dispatchEvent(new CustomEvent('skyy:3d-fallback'));
    });
  }
  document.addEventListener('skyy:walking-in', loadThree);
  document.addEventListener('skyy:prepare', loadThree);
  if (home) {
    // Mount the lightweight canonical portrait promptly. The heavy renderer
    // waits for the actual hero image, page load and a genuine idle slot.
    loadGuide().catch(function () {});
    var heroImage = home.closest('[data-recovery-hero]')?.querySelector('picture img, img');
    var poster = heroImage?.decode ? heroImage.decode().catch(function () {}) : Promise.resolve();
    var loaded =
      document.readyState === 'complete'
        ? Promise.resolve()
        : new Promise(function (resolve) {
            window.addEventListener('load', resolve, { once: true });
          });
    Promise.all([loadGuide(), poster, loaded])
      .then(function () {
        var prepare = function () {
          window.skyyRoseConcierge?.prepareHome();
        };
        if (window.requestIdleCallback) window.requestIdleCallback(prepare, { timeout: 2000 });
        else setTimeout(prepare, 0);
      })
      .catch(function () {});
  }
  invite.addEventListener('click', function (event) {
    var dialog = document.getElementById('skyy-ask-dialog');
    if (!dialog || typeof dialog.showModal !== 'function') return;
    if (
      event.defaultPrevented ||
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.altKey ||
      event.shiftKey
    )
      return;
    if (window.skyyRoseConcierge) return;
    event.preventDefault();
    var invitedFrom = document.activeElement;
    invite.setAttribute('aria-busy', 'true');
    loadGuide()
      .then(function () {
        invite.removeAttribute('aria-busy');
        if (window.skyyRoseConcierge) {
          if (!document.hidden && (document.activeElement === invitedFrom || document.activeElement === invite))
            window.skyyRoseConcierge.open();
        } else throw new Error('Skyy guide unavailable');
      })
      .catch(function () {
        invite.removeAttribute('aria-busy');
        // A real contact link remains the fallback; a second activation follows it.
        config.mascotUrl = '';
        invite.title = 'The guide could not load. Open Contact instead.';
        invite.replaceWith(invite.cloneNode(true));
      });
  });
})();
