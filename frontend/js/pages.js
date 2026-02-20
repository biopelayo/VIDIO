/* ============================================================
   VIDIO Frontend — Page Renderers
   Each function renders a complete page into the #app shell.
   ============================================================ */

// ===================== LOGIN =====================
function pageLogin() {
  const app = document.getElementById('app');
  app.innerHTML = `
    <div class="login-page">
      <div class="login-card">
        <div class="logo">VIDIO</div>
        <div class="subtitle">Vision-Integrated Diagnostic Imaging Orchestrator</div>
        <div id="login-error" class="login-error"></div>
        <div class="form-group">
          <label>Username</label>
          <input id="login-user" type="text" placeholder="Enter username" autofocus>
        </div>
        <div class="form-group">
          <label>Password</label>
          <input id="login-pass" type="password" placeholder="Enter password">
        </div>
        <button class="btn btn-primary btn-block btn-lg" id="login-btn" style="margin-top:20px">
          Sign In
        </button>
      </div>
    </div>`;

  const doLogin = async () => {
    const user = document.getElementById('login-user').value.trim();
    const pass = document.getElementById('login-pass').value;
    const errEl = document.getElementById('login-error');
    const btn = document.getElementById('login-btn');

    if (!user || !pass) { errEl.textContent = 'Please enter username and password'; return; }

    btn.disabled = true;
    btn.textContent = 'Signing in...';
    try {
      await API.login(user, pass);
      Router.navigate('/dashboard');
    } catch (e) {
      errEl.textContent = e.message || 'Invalid credentials';
      btn.disabled = false;
      btn.textContent = 'Sign In';
    }
  };

  document.getElementById('login-btn').addEventListener('click', doLogin);
  document.getElementById('login-pass').addEventListener('keydown', e => { if (e.key === 'Enter') doLogin(); });
  document.getElementById('login-user').addEventListener('keydown', e => { if (e.key === 'Enter') document.getElementById('login-pass').focus(); });
}


// ===================== DASHBOARD =====================
async function pageDashboard() {
  renderShell('Dashboard', 'Overview of your imaging data', '', '<div class="loading">Loading...</div>');

  try {
    const [patients, studies, findings, processes] = await Promise.all([
      API.patients.list().catch(() => []),
      API.studies.list().catch(() => []),
      API.findings.list().catch(() => []),
      API.processes.list().catch(() => []),
    ]);

    const pList = Array.isArray(patients) ? patients : [];
    const sList = Array.isArray(studies) ? studies : [];
    const fList = Array.isArray(findings) ? findings : [];
    const prList = Array.isArray(processes) ? processes : [];

    const runningCount = prList.filter(p => p.status === 'RUNNING').length;
    const criticalCount = fList.filter(f => (f.severity || '').toUpperCase() === 'CRITICAL').length;

    const recentFindings = fList.slice(0, 8);
    const recentProcesses = prList.slice(0, 5);

    const body = `
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-icon blue">\u263A</div>
          <div>
            <div class="stat-value">${pList.length}</div>
            <div class="stat-label">Patients</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon green">\u2630</div>
          <div>
            <div class="stat-value">${sList.length}</div>
            <div class="stat-label">Studies</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon orange">\u2691</div>
          <div>
            <div class="stat-value">${fList.length}</div>
            <div class="stat-label">Findings ${criticalCount > 0 ? `<span class="badge badge-red">${criticalCount} critical</span>` : ''}</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon ${runningCount > 0 ? 'blue' : 'green'}">\u21BB</div>
          <div>
            <div class="stat-value">${prList.length}</div>
            <div class="stat-label">Processes ${runningCount > 0 ? `<span class="badge badge-blue">${runningCount} running</span>` : ''}</div>
          </div>
        </div>
      </div>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px">
        <div class="card">
          <div class="card-header"><h3>Recent Findings</h3></div>
          <div class="card-body" style="padding:8px">
            ${recentFindings.length === 0 ? '<div class="empty-state" style="padding:20px"><div class="empty-title">No findings yet</div></div>' :
              recentFindings.map(f => `
                <div class="finding-item" style="cursor:pointer" onclick="Router.navigate('/findings/${f.id}')">
                  <div class="finding-severity ${(f.severity||'low').toLowerCase()}"></div>
                  <div class="finding-content">
                    <div class="finding-title">${f.label || f.type || 'Finding'}</div>
                    <div class="finding-meta">${severityBadge(f.severity)} &middot; ${fmtDate(f.created_at || f.timestamp)}</div>
                  </div>
                </div>`).join('')}
          </div>
        </div>

        <div class="card">
          <div class="card-header"><h3>Active Processes</h3></div>
          <div class="card-body" style="padding:0">
            ${recentProcesses.length === 0 ? '<div class="empty-state" style="padding:20px"><div class="empty-title">No processes</div></div>' :
              recentProcesses.map(p => `
                <div class="process-item">
                  ${p.status === 'RUNNING' ? '<div class="process-spinner"></div>' : `<span>${p.status === 'COMPLETED' ? '\u2713' : '\u2717'}</span>`}
                  <div class="process-info">
                    <div class="process-name">${p.type || 'Analysis'}</div>
                    <div class="process-detail">${statusBadge(p.status)} &middot; ${fmtDateTime(p.created_at || p.started_at)}</div>
                  </div>
                </div>`).join('')}
          </div>
        </div>
      </div>`;

    document.querySelector('.page-body').innerHTML = body;
  } catch (e) {
    document.querySelector('.page-body').innerHTML = `<div class="empty-state"><div class="empty-title">Error loading dashboard</div><div class="empty-desc">${e.message}</div></div>`;
  }
}


// ===================== PATIENTS =====================
async function pagePatients() {
  renderShell('Patients', 'Manage patient records',
    '<button class="btn btn-primary" onclick="openAddPatientModal()">+ Add Patient</button>', '');

  try {
    const patients = await API.patients.list();
    const list = Array.isArray(patients) ? patients : [];

    document.querySelector('.page-body').innerHTML = renderTable(
      [
        { label: 'ID', render: r => `<code>${shortId(r.id)}</code>` },
        { label: 'Name', key: 'name' },
        { label: 'DOB', render: r => fmtDate(r.date_of_birth || r.dob) },
        { label: 'Gender', key: 'gender' },
        { label: 'Created', render: r => fmtDate(r.created_at) },
      ],
      list,
      r => `<button class="btn btn-sm btn-secondary" onclick="Router.navigate('/patients/${r.id}')">View</button>
            <button class="btn btn-sm btn-danger" onclick="deletePatient('${r.id}')">Del</button>`
    );
  } catch (e) {
    Toast.error('Failed to load patients: ' + e.message);
  }
}

window.openAddPatientModal = function() {
  Modal.open('Add Patient', `
    <div class="form-group">
      <label>Patient Name</label>
      <input class="form-control" id="m-pat-name" placeholder="Full name">
    </div>
    <div class="form-row">
      <div class="form-group">
        <label>Date of Birth</label>
        <input class="form-control" id="m-pat-dob" type="date">
      </div>
      <div class="form-group">
        <label>Gender</label>
        <select class="form-control" id="m-pat-gender">
          <option value="">Select...</option>
          <option value="M">Male</option>
          <option value="F">Female</option>
          <option value="O">Other</option>
        </select>
      </div>
    </div>
    <div class="form-group">
      <label>Notes</label>
      <textarea class="form-control" id="m-pat-notes" rows="2"></textarea>
    </div>`,
    `<button class="btn btn-secondary" onclick="Modal.close()">Cancel</button>
     <button class="btn btn-primary" onclick="submitPatient()">Create</button>`
  );
};

window.submitPatient = async function() {
  const data = {
    name: document.getElementById('m-pat-name').value.trim(),
    date_of_birth: document.getElementById('m-pat-dob').value,
    gender: document.getElementById('m-pat-gender').value,
    notes: document.getElementById('m-pat-notes').value.trim(),
  };
  if (!data.name) { Toast.warning('Name is required'); return; }
  try {
    await API.patients.create(data);
    Modal.close();
    Toast.success('Patient created');
    pagePatients();
  } catch (e) { Toast.error(e.message); }
};

window.deletePatient = async function(id) {
  if (!confirm('Delete this patient?')) return;
  try {
    await API.patients.remove(id);
    Toast.success('Patient deleted');
    pagePatients();
  } catch (e) { Toast.error(e.message); }
};


// ===================== PATIENT DETAIL =====================
async function pagePatientDetail(params) {
  renderShell('Patient Detail', 'Patients', '', 'Loading...');
  try {
    const patient = await API.patients.byId(params.id);
    document.querySelector('.page-body').innerHTML = `
      <div class="card mb-lg">
        <div class="card-header"><h3>${patient.name || 'Unknown'}</h3></div>
        <div class="card-body">
          <div class="form-row">
            <div><strong>ID:</strong> <code>${patient.id}</code></div>
            <div><strong>Gender:</strong> ${patient.gender || '—'}</div>
          </div>
          <div class="form-row mt-md">
            <div><strong>DOB:</strong> ${fmtDate(patient.date_of_birth || patient.dob)}</div>
            <div><strong>Created:</strong> ${fmtDate(patient.created_at)}</div>
          </div>
          ${patient.notes ? `<div class="mt-md"><strong>Notes:</strong> ${patient.notes}</div>` : ''}
        </div>
      </div>`;
  } catch (e) { Toast.error(e.message); }
}


// ===================== STUDIES =====================
async function pageStudies() {
  renderShell('Studies', 'Manage imaging studies',
    '<button class="btn btn-primary" onclick="openAddStudyModal()">+ Add Study</button>', '');

  try {
    const studies = await API.studies.list();
    const list = Array.isArray(studies) ? studies : [];

    document.querySelector('.page-body').innerHTML = renderTable(
      [
        { label: 'ID', render: r => `<code>${shortId(r.id)}</code>` },
        { label: 'Description', key: 'description' },
        { label: 'Modality', key: 'modality' },
        { label: 'Patient', render: r => shortId(r.id_patient || '') },
        { label: 'Created', render: r => fmtDate(r.created_at) },
      ],
      list,
      r => `<button class="btn btn-sm btn-secondary" onclick="Router.navigate('/studies/${r.id}')">View</button>
            <button class="btn btn-sm btn-danger" onclick="deleteStudy('${r.id}')">Del</button>`
    );
  } catch (e) { Toast.error(e.message); }
}

window.openAddStudyModal = async function() {
  let patOpts = '<option value="">Select patient...</option>';
  try {
    const patients = await API.patients.list();
    (Array.isArray(patients) ? patients : []).forEach(p => {
      patOpts += `<option value="${p.id}">${p.name} (${shortId(p.id)})</option>`;
    });
  } catch {}

  Modal.open('Add Study', `
    <div class="form-group">
      <label>Patient</label>
      <select class="form-control" id="m-st-patient">${patOpts}</select>
    </div>
    <div class="form-group">
      <label>Description</label>
      <input class="form-control" id="m-st-desc" placeholder="e.g. Fundus Exam 2024">
    </div>
    <div class="form-group">
      <label>Modality</label>
      <select class="form-control" id="m-st-mod">
        <option value="retinal">Retinal</option>
        <option value="histology">Histology</option>
        <option value="radiology">Radiology</option>
        <option value="spatial">Spatial Transcriptomics</option>
      </select>
    </div>`,
    `<button class="btn btn-secondary" onclick="Modal.close()">Cancel</button>
     <button class="btn btn-primary" onclick="submitStudy()">Create</button>`
  );
};

window.submitStudy = async function() {
  const data = {
    id_patient: document.getElementById('m-st-patient').value,
    description: document.getElementById('m-st-desc').value.trim(),
    modality: document.getElementById('m-st-mod').value,
  };
  if (!data.id_patient) { Toast.warning('Select a patient'); return; }
  try {
    await API.studies.create(data);
    Modal.close();
    Toast.success('Study created');
    pageStudies();
  } catch (e) { Toast.error(e.message); }
};

window.deleteStudy = async function(id) {
  if (!confirm('Delete this study and its data?')) return;
  try {
    await API.studies.remove(id);
    Toast.success('Study deleted');
    pageStudies();
  } catch (e) { Toast.error(e.message); }
};


// ===================== STUDY DETAIL =====================
async function pageStudyDetail(params) {
  renderShell('Study Detail', 'Studies', '', 'Loading...');
  try {
    const [study, seriesList, findingsList] = await Promise.all([
      API.studies.byId(params.id),
      API.studies.series(params.id).catch(() => []),
      API.studies.findings(params.id).catch(() => []),
    ]);
    const sl = Array.isArray(seriesList) ? seriesList : [];
    const fl = Array.isArray(findingsList) ? findingsList : [];

    document.querySelector('.page-body').innerHTML = `
      <div class="card mb-lg">
        <div class="card-header">
          <h3>${study.description || 'Study'}</h3>
          <span class="badge badge-blue">${study.modality || '—'}</span>
        </div>
        <div class="card-body">
          <div class="form-row">
            <div><strong>ID:</strong> <code>${study.id}</code></div>
            <div><strong>Patient:</strong> <code>${shortId(study.id_patient || '')}</code></div>
          </div>
        </div>
      </div>

      <div class="tabs mb-md">
        <div class="tab active" data-tab="series">Series (${sl.length})</div>
        <div class="tab" data-tab="findings">Findings (${fl.length})</div>
      </div>

      <div class="tab-panel active" data-panel="series">
        ${renderTable(
          [
            { label: 'ID', render: r => `<code>${shortId(r.id)}</code>` },
            { label: 'Description', key: 'description' },
            { label: 'Created', render: r => fmtDate(r.created_at) },
          ], sl,
          r => `<button class="btn btn-sm btn-secondary" onclick="Router.navigate('/series/${r.id}')">View</button>`
        )}
      </div>

      <div class="tab-panel" data-panel="findings">
        ${fl.length === 0 ? '<div class="empty-state"><div class="empty-title">No findings</div></div>' :
          fl.map(f => `
            <div class="finding-item">
              <div class="finding-severity ${(f.severity||'low').toLowerCase()}"></div>
              <div class="finding-content">
                <div class="finding-title">${f.label || f.type || 'Finding'}</div>
                <div class="finding-meta">${severityBadge(f.severity)} &middot; Confidence: ${((f.confidence||0)*100).toFixed(0)}%</div>
              </div>
            </div>`).join('')}
      </div>`;

    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.querySelector(`[data-panel="${tab.dataset.tab}"]`).classList.add('active');
      });
    });
  } catch (e) { Toast.error(e.message); }
}


// ===================== IMAGES =====================
async function pageImages() {
  renderShell('Images', 'Browse all images', '', '');
  try {
    const images = await API.images.list();
    const list = Array.isArray(images) ? images : [];

    document.querySelector('.page-body').innerHTML = renderTable(
      [
        { label: 'ID', render: r => `<code>${shortId(r.id)}</code>` },
        { label: 'Filename', key: 'original_filename' },
        { label: 'Format', key: 'format' },
        { label: 'Series', render: r => shortId(r.id_series || '') },
        { label: 'Created', render: r => fmtDate(r.created_at) },
      ],
      list,
      r => `<button class="btn btn-sm btn-secondary" onclick="Router.navigate('/images/${r.id}')">View</button>`
    );
  } catch (e) { Toast.error(e.message); }
}


// ===================== IMAGE DETAIL =====================
async function pageImageDetail(params) {
  renderShell('Image Viewer', 'Images', '', 'Loading...');
  try {
    const img = await API.images.byId(params.id);
    document.querySelector('.page-body').innerHTML = `
      <div class="card mb-lg">
        <div class="card-header"><h3>${img.original_filename || 'Image'}</h3></div>
        <div class="card-body">
          <div class="image-viewer">
            ${img.storage_path
              ? `<img src="/static/repo/${img.storage_path}" alt="Medical image" onerror="this.parentElement.innerHTML='<div class=\\'no-image\\'>Image not available on disk</div>'">`
              : '<div class="no-image">No image file associated</div>'}
          </div>
          <div class="form-row mt-md">
            <div><strong>ID:</strong> <code>${img.id}</code></div>
            <div><strong>Format:</strong> ${img.format || '—'}</div>
          </div>
          <div class="form-row mt-md">
            <div><strong>Series:</strong> <code>${shortId(img.id_series || '')}</code></div>
            <div><strong>Size:</strong> ${img.file_size_bytes ? (img.file_size_bytes / 1024).toFixed(1) + ' KB' : '—'}</div>
          </div>
        </div>
      </div>`;
  } catch (e) { Toast.error(e.message); }
}


// ===================== UPLOAD =====================
function pageUpload() {
  renderShell('Upload Images', 'Upload medical images to the repository',
    '', `
    <div class="card mb-lg">
      <div class="card-header"><h3>Upload Files</h3></div>
      <div class="card-body">
        <div class="upload-zone" id="drop-zone">
          <div class="upload-icon">\u21E7</div>
          <div class="upload-text">Drop files here or click to browse</div>
          <div class="upload-hint">Supports: PNG, JPG, DICOM, NIfTI, SVS, H5AD</div>
          <input type="file" id="file-input" multiple style="display:none"
                 accept=".png,.jpg,.jpeg,.dcm,.nii,.nii.gz,.svs,.tiff,.tif,.h5ad">
        </div>
        <div id="upload-list" class="mt-md"></div>
        <div id="upload-actions" class="mt-md hidden">
          <button class="btn btn-primary btn-lg" id="upload-btn">Upload Files</button>
        </div>
      </div>
    </div>`
  );

  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  let selectedFiles = [];

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    addFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener('change', () => addFiles(fileInput.files));

  function addFiles(files) {
    selectedFiles = [...selectedFiles, ...Array.from(files)];
    renderFileList();
  }

  function renderFileList() {
    const list = document.getElementById('upload-list');
    const actions = document.getElementById('upload-actions');
    if (selectedFiles.length === 0) { list.innerHTML = ''; actions.classList.add('hidden'); return; }

    actions.classList.remove('hidden');
    list.innerHTML = selectedFiles.map((f, i) => `
      <div class="flex items-center justify-between" style="padding:6px 0; border-bottom:1px solid var(--border-light)">
        <span>${f.name} <span class="text-muted">(${(f.size/1024).toFixed(1)} KB)</span></span>
        <button class="btn btn-sm btn-danger" onclick="removeUploadFile(${i})">Remove</button>
      </div>`).join('');
  }

  window.removeUploadFile = function(idx) {
    selectedFiles.splice(idx, 1);
    renderFileList();
  };

  document.getElementById('upload-btn').addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;
    const btn = document.getElementById('upload-btn');
    btn.disabled = true;
    btn.textContent = 'Uploading...';

    const fd = new FormData();
    selectedFiles.forEach(f => fd.append('file', f));

    try {
      const result = await API.uploads.send(fd);
      Toast.success(`Uploaded ${selectedFiles.length} file(s)`);
      selectedFiles = [];
      renderFileList();
      btn.textContent = 'Upload Files';
      btn.disabled = false;
    } catch (e) {
      Toast.error('Upload failed: ' + e.message);
      btn.textContent = 'Upload Files';
      btn.disabled = false;
    }
  });
}


// ===================== ANALYSIS =====================
async function pageAnalysis() {
  let studyOpts = '<option value="">Select a study...</option>';
  try {
    const studies = await API.studies.list();
    (Array.isArray(studies) ? studies : []).forEach(s => {
      studyOpts += `<option value="${s.id}">${s.description || shortId(s.id)} (${s.modality || '?'})</option>`;
    });
  } catch {}

  renderShell('Run Analysis', 'Launch a pipeline analysis on a study', '', `
    <div class="card" style="max-width:600px">
      <div class="card-header"><h3>Launch Analysis Pipeline</h3></div>
      <div class="card-body">
        <div class="form-group">
          <label>Study</label>
          <select class="form-control" id="an-study">${studyOpts}</select>
        </div>
        <div class="form-group">
          <label>Pipeline / Modality</label>
          <select class="form-control" id="an-mod">
            <option value="retinal">Retinal (Fundus / OCT)</option>
            <option value="histology">Histology (H&E / WSI)</option>
            <option value="radiology">Radiology (CT / MRI)</option>
            <option value="spatial">Spatial Transcriptomics</option>
          </select>
        </div>
        <div class="form-group">
          <label>Parameters (optional JSON)</label>
          <textarea class="form-control" id="an-params" rows="3" placeholder='{"key": "value"}'></textarea>
        </div>
      </div>
      <div class="card-footer">
        <button class="btn btn-primary btn-lg" id="an-run">Run Analysis</button>
      </div>
    </div>`
  );

  document.getElementById('an-run').addEventListener('click', async () => {
    const studyId = document.getElementById('an-study').value;
    const modality = document.getElementById('an-mod').value;
    const paramsRaw = document.getElementById('an-params').value.trim();

    if (!studyId) { Toast.warning('Please select a study'); return; }

    let parameters = {};
    if (paramsRaw) {
      try { parameters = JSON.parse(paramsRaw); }
      catch { Toast.error('Invalid JSON in parameters'); return; }
    }

    const btn = document.getElementById('an-run');
    btn.disabled = true;
    btn.textContent = 'Launching...';

    try {
      const result = await API.analysis[modality]({ id_study: studyId, parameters });
      Toast.success(`Analysis queued! Process: ${shortId(result.process_id)}`);
      Router.navigate('/processes');
    } catch (e) {
      Toast.error('Failed: ' + e.message);
      btn.disabled = false;
      btn.textContent = 'Run Analysis';
    }
  });
}


// ===================== FINDINGS =====================
async function pageFindings() {
  renderShell('Findings', 'All diagnostic findings across studies', '', '');
  try {
    const findings = await API.findings.list();
    const list = Array.isArray(findings) ? findings : [];

    document.querySelector('.page-body').innerHTML = renderTable(
      [
        { label: 'ID', render: r => `<code>${shortId(r.id)}</code>` },
        { label: 'Type', render: r => r.label || r.type || '—' },
        { label: 'Severity', render: r => severityBadge(r.severity) },
        { label: 'Confidence', render: r => r.confidence != null ? `${(r.confidence * 100).toFixed(0)}%` : '—' },
        { label: 'Study', render: r => shortId(r.id_study || '') },
        { label: 'Date', render: r => fmtDate(r.created_at || r.timestamp) },
      ],
      list,
      r => `<button class="btn btn-sm btn-secondary" onclick="Router.navigate('/findings/${r.id}')">Detail</button>`
    );
  } catch (e) { Toast.error(e.message); }
}


// ===================== FINDING DETAIL =====================
async function pageFindingDetail(params) {
  renderShell('Finding Detail', 'Findings', '', 'Loading...');
  try {
    const f = await API.findings.byId(params.id);
    const details = f.details || f.metadata || {};

    document.querySelector('.page-body').innerHTML = `
      <div class="card mb-lg">
        <div class="card-header">
          <h3>${f.label || f.type || 'Finding'}</h3>
          ${severityBadge(f.severity)}
        </div>
        <div class="card-body">
          <div class="form-row mb-md">
            <div><strong>ID:</strong> <code>${f.id}</code></div>
            <div><strong>Confidence:</strong> ${f.confidence != null ? (f.confidence * 100).toFixed(1) + '%' : '—'}</div>
          </div>
          <div class="form-row mb-md">
            <div><strong>Study:</strong> <code>${shortId(f.id_study || '')}</code></div>
            <div><strong>Image:</strong> <code>${shortId(f.id_image || '')}</code></div>
          </div>
          ${f.description ? `<div class="mb-md"><strong>Description:</strong><br>${f.description}</div>` : ''}
          ${Object.keys(details).length > 0 ? `
            <div class="mb-md">
              <strong>Details:</strong>
              <pre style="background:var(--bg-input);padding:12px;border-radius:var(--radius-sm);overflow-x:auto;font-size:.8rem;margin-top:6px">${JSON.stringify(details, null, 2)}</pre>
            </div>` : ''}
          ${f.location ? `<div class="mb-md"><strong>Location:</strong> <code>${JSON.stringify(f.location)}</code></div>` : ''}
        </div>
      </div>`;
  } catch (e) { Toast.error(e.message); }
}


// ===================== PROCESSES =====================
async function pageProcesses() {
  renderShell('Processes', 'Pipeline execution monitor', '', '');

  async function refresh() {
    try {
      const processes = await API.processes.list();
      const list = Array.isArray(processes) ? processes : [];

      document.querySelector('.page-body').innerHTML = `
        <div class="mb-md flex justify-between items-center">
          <span class="text-muted">${list.length} process(es)</span>
          <button class="btn btn-sm btn-secondary" onclick="pageProcesses()">Refresh</button>
        </div>` +
        renderTable(
          [
            { label: 'ID', render: r => `<code>${shortId(r.id)}</code>` },
            { label: 'Type', key: 'type' },
            { label: 'Status', render: r => statusBadge(r.status) },
            { label: 'Study', render: r => shortId(r.id_study || '') },
            { label: 'Started', render: r => fmtDateTime(r.created_at || r.started_at) },
            { label: 'User', render: r => shortId(r.id_user || '') },
          ],
          list
        );
    } catch (e) { Toast.error(e.message); }
  }

  await refresh();
}


// ===================== USERS =====================
async function pageUsers() {
  renderShell('Users', 'Manage platform users',
    '<button class="btn btn-primary" onclick="openAddUserModal()">+ Add User</button>', '');

  try {
    const users = await API.users.list();
    const list = Array.isArray(users) ? users : [];

    document.querySelector('.page-body').innerHTML = renderTable(
      [
        { label: 'ID', render: r => `<code>${shortId(r.id)}</code>` },
        { label: 'Username', key: 'username' },
        { label: 'Name', key: 'name' },
        { label: 'Role', render: r => `<span class="badge badge-blue">${r.role || 'user'}</span>` },
        { label: 'Created', render: r => fmtDate(r.created_at) },
      ],
      list,
      r => `<button class="btn btn-sm btn-danger" onclick="deleteUser('${r.id}')">Del</button>`
    );
  } catch (e) { Toast.error(e.message); }
}

window.openAddUserModal = function() {
  Modal.open('Add User', `
    <div class="form-row">
      <div class="form-group">
        <label>Username</label>
        <input class="form-control" id="m-usr-name" placeholder="username">
      </div>
      <div class="form-group">
        <label>Full Name</label>
        <input class="form-control" id="m-usr-full" placeholder="Full name">
      </div>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label>Password</label>
        <input class="form-control" id="m-usr-pass" type="password" placeholder="Password">
      </div>
      <div class="form-group">
        <label>Role</label>
        <select class="form-control" id="m-usr-role">
          <option value="user">User</option>
          <option value="admin">Admin</option>
          <option value="viewer">Viewer</option>
        </select>
      </div>
    </div>`,
    `<button class="btn btn-secondary" onclick="Modal.close()">Cancel</button>
     <button class="btn btn-primary" onclick="submitUser()">Create</button>`
  );
};

window.submitUser = async function() {
  const data = {
    username: document.getElementById('m-usr-name').value.trim(),
    name: document.getElementById('m-usr-full').value.trim(),
    password: document.getElementById('m-usr-pass').value,
    role: document.getElementById('m-usr-role').value,
  };
  if (!data.username || !data.password) { Toast.warning('Username and password required'); return; }
  try {
    await API.users.create(data);
    Modal.close();
    Toast.success('User created');
    pageUsers();
  } catch (e) { Toast.error(e.message); }
};

window.deleteUser = async function(id) {
  if (!confirm('Delete this user?')) return;
  try {
    await API.users.remove(id);
    Toast.success('User deleted');
    pageUsers();
  } catch (e) { Toast.error(e.message); }
};


// ===================== SERIES DETAIL =====================
async function pageSeriesDetail(params) {
  renderShell('Series Detail', 'Studies > Series', '', 'Loading...');
  try {
    const [series, imgs] = await Promise.all([
      API.series.byId(params.id),
      API.series.images(params.id).catch(() => []),
    ]);
    const imgList = Array.isArray(imgs) ? imgs : [];

    document.querySelector('.page-body').innerHTML = `
      <div class="card mb-lg">
        <div class="card-header"><h3>${series.description || 'Series'}</h3></div>
        <div class="card-body">
          <div class="form-row">
            <div><strong>ID:</strong> <code>${series.id}</code></div>
            <div><strong>Study:</strong> <code>${shortId(series.id_study || '')}</code></div>
          </div>
        </div>
      </div>

      <h3 class="mb-md">Images (${imgList.length})</h3>
      ${renderTable(
        [
          { label: 'ID', render: r => `<code>${shortId(r.id)}</code>` },
          { label: 'Filename', key: 'original_filename' },
          { label: 'Format', key: 'format' },
        ], imgList,
        r => `<button class="btn btn-sm btn-secondary" onclick="Router.navigate('/images/${r.id}')">View</button>`
      )}`;
  } catch (e) { Toast.error(e.message); }
}
