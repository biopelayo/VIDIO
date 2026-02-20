/* ============================================================
   VIDIO Frontend — Reusable UI Components
   Toast notifications, modals, sidebar, tables, etc.
   ============================================================ */

// --- Toast System ---
const Toast = (() => {
  function show(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toasts');
    const icons = { success: '\u2713', error: '\u2717', info: '\u2139', warning: '\u26A0' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type] || ''}</span><span class="toast-msg">${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, duration);
  }

  return {
    success: msg => show(msg, 'success'),
    error:   msg => show(msg, 'error'),
    info:    msg => show(msg, 'info'),
    warning: msg => show(msg, 'warning'),
  };
})();


// --- Modal System ---
const Modal = (() => {
  let overlay = null;

  function open(title, bodyHtml, footerHtml) {
    close();
    overlay = document.createElement('div');
    overlay.className = 'modal-overlay active';
    overlay.innerHTML = `
      <div class="modal">
        <div class="modal-header">
          <h2>${title}</h2>
          <button class="modal-close" onclick="Modal.close()">&times;</button>
        </div>
        <div class="modal-body">${bodyHtml}</div>
        ${footerHtml ? `<div class="modal-footer">${footerHtml}</div>` : ''}
      </div>`;
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
    document.body.appendChild(overlay);
    return overlay;
  }

  function close() {
    if (overlay) { overlay.remove(); overlay = null; }
  }

  return { open, close };
})();


// --- Sidebar Component ---
function renderSidebar() {
  const user = API.getUser() || {};
  const initials = (user.name || user.username || 'U').slice(0, 2).toUpperCase();

  return `
    <aside class="sidebar">
      <div class="sidebar-header">
        <div class="brand">VIDIO</div>
        <div class="version">v1.0 &middot; Biomedical Imaging</div>
      </div>
      <nav class="sidebar-nav">
        <div class="nav-section">
          <div class="nav-section-title">Overview</div>
          <div class="nav-item" data-route="/dashboard">
            <span class="icon">\u2302</span> Dashboard
          </div>
        </div>
        <div class="nav-section">
          <div class="nav-section-title">Data</div>
          <div class="nav-item" data-route="/patients">
            <span class="icon">\u263A</span> Patients
          </div>
          <div class="nav-item" data-route="/studies">
            <span class="icon">\u2630</span> Studies
          </div>
          <div class="nav-item" data-route="/images">
            <span class="icon">\u25A3</span> Images
          </div>
          <div class="nav-item" data-route="/upload">
            <span class="icon">\u21E7</span> Upload
          </div>
        </div>
        <div class="nav-section">
          <div class="nav-section-title">Analysis</div>
          <div class="nav-item" data-route="/analysis">
            <span class="icon">\u2699</span> Run Analysis
          </div>
          <div class="nav-item" data-route="/findings">
            <span class="icon">\u2691</span> Findings
          </div>
          <div class="nav-item" data-route="/processes">
            <span class="icon">\u21BB</span> Processes
          </div>
        </div>
        <div class="nav-section">
          <div class="nav-section-title">Admin</div>
          <div class="nav-item" data-route="/users">
            <span class="icon">\u2605</span> Users
          </div>
        </div>
      </nav>
      <div class="sidebar-footer">
        <div class="user-avatar">${initials}</div>
        <div class="user-info">
          <div class="user-name">${user.name || user.username || 'User'}</div>
          <div class="user-role">${user.role || 'user'}</div>
        </div>
        <button class="logout-btn" onclick="API.logout()" title="Logout">\u23FB</button>
      </div>
    </aside>`;
}


// --- Page Shell ---
function renderShell(pageTitle, breadcrumb, headerRight, bodyHtml) {
  const hash = window.location.hash.slice(1) || '/dashboard';

  const app = document.getElementById('app');
  app.innerHTML = renderSidebar() + `
    <main class="main-content">
      <div class="page-header">
        <div>
          <h1>${pageTitle}</h1>
          ${breadcrumb ? `<div class="breadcrumb">${breadcrumb}</div>` : ''}
        </div>
        <div class="flex gap-sm">${headerRight || ''}</div>
      </div>
      <div class="page-body">${bodyHtml}</div>
    </main>`;

  // Activate current nav item
  app.querySelectorAll('.nav-item').forEach(item => {
    const route = item.dataset.route;
    item.classList.toggle('active', hash.startsWith(route));
    item.addEventListener('click', () => Router.navigate(route));
  });
}


// --- Data Table ---
function renderTable(columns, rows, actions) {
  if (!rows || rows.length === 0) {
    return `<div class="empty-state">
      <div class="empty-icon">\u2205</div>
      <div class="empty-title">No data found</div>
      <div class="empty-desc">There are no records to display.</div>
    </div>`;
  }

  const headerCells = columns.map(c => `<th>${c.label}</th>`).join('') +
    (actions ? '<th style="width:100px">Actions</th>' : '');

  const bodyRows = rows.map(row => {
    const cells = columns.map(c => {
      const val = c.render ? c.render(row) : (row[c.key] ?? '—');
      return `<td>${val}</td>`;
    }).join('');
    const actCells = actions ? `<td>${actions(row)}</td>` : '';
    return `<tr>${cells}${actCells}</tr>`;
  }).join('');

  return `<div class="table-container"><table><thead><tr>${headerCells}</tr></thead><tbody>${bodyRows}</tbody></table></div>`;
}


// --- Severity Badge ---
function severityBadge(severity) {
  const s = (severity || '').toLowerCase();
  return `<span class="badge sev-${s}">${severity || '—'}</span>`;
}

// --- Status Badge ---
function statusBadge(status) {
  const s = (status || '').toLowerCase();
  return `<span class="badge status-${s}">${status || '—'}</span>`;
}

// --- Short UUID ---
function shortId(uuid) {
  return uuid ? uuid.slice(0, 8) : '—';
}

// --- Date Formatter ---
function fmtDate(dateStr) {
  if (!dateStr) return '—';
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
  } catch { return dateStr; }
}

function fmtDateTime(dateStr) {
  if (!dateStr) return '—';
  try {
    const d = new Date(dateStr);
    return d.toLocaleString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch { return dateStr; }
}
