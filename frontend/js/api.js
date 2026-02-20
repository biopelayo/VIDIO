/* ============================================================
   VIDIO Frontend — API Client
   Handles all HTTP communication with the Falcon backend.
   ============================================================ */

const API = (() => {
  const BASE = '';  // same origin

  function getToken() {
    return localStorage.getItem('vidio_token');
  }

  function setToken(token) {
    localStorage.setItem('vidio_token', token);
  }

  function clearToken() {
    localStorage.removeItem('vidio_token');
    localStorage.removeItem('vidio_user');
  }

  function getUser() {
    try { return JSON.parse(localStorage.getItem('vidio_user')); }
    catch { return null; }
  }

  function setUser(user) {
    localStorage.setItem('vidio_user', JSON.stringify(user));
  }

  function isLoggedIn() {
    return !!getToken();
  }

  async function request(method, path, body, isMultipart) {
    const headers = {};
    const token = getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const opts = { method, headers };

    if (body && !isMultipart) {
      headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    } else if (body && isMultipart) {
      opts.body = body; // FormData
    }

    const res = await fetch(BASE + path, opts);

    if (res.status === 401) {
      clearToken();
      window.location.hash = '#/login';
      throw new Error('Session expired');
    }

    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { data = text; }

    if (!res.ok) {
      const msg = data?.error || data?.message || `HTTP ${res.status}`;
      throw new Error(msg);
    }

    return data;
  }

  // --- Auth ---
  async function login(username, password) {
    const data = await request('POST', '/auth', { username, password });
    setToken(data.token);
    setUser(data.user);
    return data;
  }

  function logout() {
    clearToken();
    window.location.hash = '#/login';
  }

  // --- CRUD helpers ---
  const get    = (path) => request('GET', path);
  const post   = (path, body) => request('POST', path, body);
  const del    = (path) => request('DELETE', path);
  const upload = (path, formData) => request('POST', path, formData, true);

  return {
    getToken, setToken, clearToken, getUser, setUser, isLoggedIn,
    login, logout,
    get, post, del, upload,
    // Convenience endpoints
    patients:  { list: () => get('/patients'),   byId: id => get(`/patients/${id}`),  create: d => post('/patients', d), remove: id => del(`/patients/${id}`) },
    studies:   { list: () => get('/studies'),     byId: id => get(`/studies/${id}`),   create: d => post('/studies', d),  remove: id => del(`/studies/${id}`),
                 series: id => get(`/studies/${id}/series`), findings: id => get(`/studies/${id}/findings`) },
    series:    { list: () => get('/series'),      byId: id => get(`/series/${id}`),    create: d => post('/series', d),   remove: id => del(`/series/${id}`),
                 images: id => get(`/series/${id}/images`) },
    images:    { list: () => get('/images'),      byId: id => get(`/images/${id}`),    create: d => post('/images', d),   remove: id => del(`/images/${id}`) },
    findings:  { list: () => get('/findings'),    byId: id => get(`/findings/${id}`),  review: (id, d) => post(`/findings/${id}`, d) },
    processes: { list: () => get('/processes'),   byId: id => get(`/processes/${id}`) },
    users:     { list: () => get('/users'),       byId: id => get(`/users/${id}`),     create: d => post('/users', d),    remove: id => del(`/users/${id}`) },
    uploads:   { send: formData => upload('/uploads', formData) },
    analysis:  {
      retinal:    d => post('/analysis/retinal', d),
      histology:  d => post('/analysis/histology', d),
      radiology:  d => post('/analysis/radiology', d),
      spatial:    d => post('/analysis/spatial', d),
    },
  };
})();
