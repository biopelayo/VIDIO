/* ============================================================
   VIDIO Frontend — Hash Router
   Simple hash-based SPA router with auth guard.
   ============================================================ */

const Router = (() => {
  const routes = {};
  let currentPage = null;

  function register(path, handler) {
    routes[path] = handler;
  }

  function navigate(path) {
    window.location.hash = '#' + path;
  }

  function resolve() {
    const hash = window.location.hash.slice(1) || '/dashboard';
    const parts = hash.split('/').filter(Boolean);

    // Auth guard
    if (hash !== '/login' && !API.isLoggedIn()) {
      navigate('/login');
      return;
    }
    if (hash === '/login' && API.isLoggedIn()) {
      navigate('/dashboard');
      return;
    }

    // Match route pattern: try exact, then /entity/:id patterns
    let handler = routes['/' + parts.join('/')];
    let params = {};

    if (!handler) {
      // Try pattern matching like /patients/:id
      for (const [pattern, h] of Object.entries(routes)) {
        const patternParts = pattern.split('/').filter(Boolean);
        if (patternParts.length !== parts.length) continue;

        let match = true;
        const extracted = {};
        for (let i = 0; i < patternParts.length; i++) {
          if (patternParts[i].startsWith(':')) {
            extracted[patternParts[i].slice(1)] = parts[i];
          } else if (patternParts[i] !== parts[i]) {
            match = false;
            break;
          }
        }
        if (match) {
          handler = h;
          params = extracted;
          break;
        }
      }
    }

    if (handler) {
      currentPage = handler;
      handler(params);
    } else {
      navigate('/dashboard');
    }
  }

  function init() {
    window.addEventListener('hashchange', resolve);
    resolve();
  }

  return { register, navigate, init, resolve };
})();
