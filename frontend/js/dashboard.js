/* ============================================================
   VIDIO Frontend — Enhanced Visual Analytics Dashboard
   Rich data visualization with charts, stats, and real-time
   pipeline monitoring.
   ============================================================ */

// ===================== ANALYTICS DASHBOARD =====================
async function pageAnalyticsDashboard() {
  renderShell('Analytics', 'Visual analytics and insights',
    `<div class="flex gap-sm">
       <button class="btn btn-sm btn-secondary" onclick="pageAnalyticsDashboard()">Refresh</button>
       <button class="btn btn-sm btn-primary" onclick="Router.navigate('/models')">View Models</button>
     </div>`, '<div class="loading">Loading analytics...</div>');

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

    // ── Compute metrics ────────────────────────────────────────
    const severityCounts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
    const categoryCounts = {};
    const modalityCounts = {};
    const monthlyFindings = {};

    fList.forEach(f => {
      const sev = (f.severity || 'LOW').toUpperCase();
      severityCounts[sev] = (severityCounts[sev] || 0) + 1;

      const cat = f.disease_category || f.finding_type || f.type || 'Unknown';
      categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;

      const date = f.created_at || f.timestamp;
      if (date) {
        const month = date.slice(0, 7);
        monthlyFindings[month] = (monthlyFindings[month] || 0) + 1;
      }
    });

    sList.forEach(s => {
      const mod = s.modality || 'Unknown';
      modalityCounts[mod] = (modalityCounts[mod] || 0) + 1;
    });

    const completedProcesses = prList.filter(p => p.status === 'COMPLETED').length;
    const failedProcesses = prList.filter(p => p.status === 'FAILED').length;
    const runningProcesses = prList.filter(p => p.status === 'RUNNING').length;
    const successRate = prList.length > 0 ? (completedProcesses / prList.length) * 100 : 0;

    // ── Render the dashboard ───────────────────────────────────
    document.querySelector('.page-body').innerHTML = `
      <!-- Row 1: Summary KPIs with sparklines -->
      <div class="stats-grid" style="grid-template-columns: repeat(5, 1fr); margin-bottom:20px">
        <div class="stat-card">
          <div class="stat-icon blue">&#9786;</div>
          <div>
            <div class="stat-value">${pList.length}</div>
            <div class="stat-label">Patients</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon green">&#9776;</div>
          <div>
            <div class="stat-value">${sList.length}</div>
            <div class="stat-label">Studies</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon orange">&#9873;</div>
          <div>
            <div class="stat-value">${fList.length}</div>
            <div class="stat-label">Findings</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon ${runningProcesses > 0 ? 'blue' : 'green'}">&#8635;</div>
          <div>
            <div class="stat-value">${prList.length}</div>
            <div class="stat-label">Analyses ${runningProcesses > 0 ? '<span class="badge badge-blue">' + runningProcesses + ' running</span>' : ''}</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon ${severityCounts.CRITICAL > 0 ? 'red' : 'green'}">&#9888;</div>
          <div>
            <div class="stat-value">${severityCounts.CRITICAL}</div>
            <div class="stat-label">Critical Alerts</div>
          </div>
        </div>
      </div>

      <!-- Row 2: Charts -->
      <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; margin-bottom:20px">
        <!-- Severity Distribution -->
        <div class="card">
          <div class="card-header"><h3>Severity Distribution</h3></div>
          <div class="card-body" style="display:flex;justify-content:center;padding:16px">
            <canvas id="chart-severity"></canvas>
          </div>
        </div>

        <!-- Findings by Category -->
        <div class="card">
          <div class="card-header"><h3>Findings by Category</h3></div>
          <div class="card-body" style="padding:16px">
            <canvas id="chart-categories"></canvas>
          </div>
        </div>

        <!-- Pipeline Success Rate -->
        <div class="card">
          <div class="card-header"><h3>Pipeline Performance</h3></div>
          <div class="card-body" style="display:flex; flex-direction:column; align-items:center; gap:16px; padding:20px">
            <canvas id="chart-success"></canvas>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; width:100%; text-align:center">
              <div>
                <div style="font-size:1.3rem;font-weight:700;color:var(--success)">${completedProcesses}</div>
                <div style="font-size:.75rem;color:var(--text-muted)">Completed</div>
              </div>
              <div>
                <div style="font-size:1.3rem;font-weight:700;color:var(--accent)">${runningProcesses}</div>
                <div style="font-size:.75rem;color:var(--text-muted)">Running</div>
              </div>
              <div>
                <div style="font-size:1.3rem;font-weight:700;color:var(--danger)">${failedProcesses}</div>
                <div style="font-size:.75rem;color:var(--text-muted)">Failed</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Row 3: Modality + Recent Activity -->
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px">
        <!-- Studies by Modality -->
        <div class="card">
          <div class="card-header"><h3>Studies by Modality</h3></div>
          <div class="card-body" style="padding:16px">
            <canvas id="chart-modality"></canvas>
          </div>
        </div>

        <!-- Available SOTA Models -->
        <div class="card">
          <div class="card-header">
            <h3>Pretrained SOTA Models</h3>
            <span class="badge badge-blue">State of the Art</span>
          </div>
          <div class="card-body" style="padding:0;max-height:300px;overflow-y:auto">
            ${renderModelList()}
          </div>
        </div>
      </div>

      <!-- Row 4: Recent Findings Feed -->
      <div class="card">
        <div class="card-header">
          <h3>Recent Findings</h3>
          <button class="btn btn-sm btn-secondary" onclick="Router.navigate('/findings')">View All</button>
        </div>
        <div class="card-body" style="padding:8px;max-height:350px;overflow-y:auto">
          ${fList.length === 0
            ? '<div class="empty-state" style="padding:30px"><div class="empty-title">No findings yet</div><div class="empty-desc">Run an analysis pipeline to generate findings.</div></div>'
            : fList.slice(0, 12).map(f => `
              <div class="finding-item" style="cursor:pointer" onclick="Router.navigate('/findings/${f.id}')">
                <div class="finding-severity ${(f.severity||'low').toLowerCase()}"></div>
                <div class="finding-content">
                  <div class="finding-title">${f.label || f.finding_type || f.type || 'Finding'}</div>
                  <div class="finding-meta">
                    ${severityBadge(f.severity)}
                    &middot; ${f.disease_category || ''}
                    &middot; Confidence: ${f.confidence != null ? (f.confidence * 100).toFixed(0) + '%' : '---'}
                    &middot; ${fmtDate(f.created_at || f.timestamp)}
                  </div>
                </div>
              </div>`).join('')}
        </div>
      </div>
    `;

    // ── Render charts after DOM is ready ────────────────────────
    setTimeout(() => {
      // Severity donut
      Charts.donut('chart-severity', [
        { label: 'Critical', value: severityCounts.CRITICAL, color: '#f85149' },
        { label: 'High', value: severityCounts.HIGH, color: '#f0883e' },
        { label: 'Medium', value: severityCounts.MEDIUM, color: '#d29922' },
        { label: 'Low', value: severityCounts.LOW, color: '#3fb950' },
      ], { centerLabel: 'Findings', size: 220 });

      // Category horizontal bar
      const catData = Object.entries(categoryCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([label, value], i) => ({ label, value, color: Charts.CATEGORY_COLORS[i] }));
      Charts.horizontalBar('chart-categories', catData, { title: '', width: 360 });

      // Success rate ring
      Charts.progressRing('chart-success', successRate, {
        size: 120,
        label: 'Success Rate',
      });

      // Modality bar chart
      const modData = Object.entries(modalityCounts)
        .map(([label, value]) => ({ label: label.charAt(0).toUpperCase() + label.slice(1), value }));
      Charts.bar('chart-modality', modData, { title: '', width: 380, height: 200 });

    }, 50);

  } catch (e) {
    document.querySelector('.page-body').innerHTML = `
      <div class="empty-state">
        <div class="empty-title">Error loading analytics</div>
        <div class="empty-desc">${e.message}</div>
      </div>`;
  }
}


// ── Pretrained Models List (Static) ──────────────────────────
function renderModelList() {
  const models = [
    {
      key: 'efficientnet_b4_retinal',
      name: 'EfficientNet-B4',
      task: 'Retinal Disease Classification (8 classes)',
      classes: 'DR, Glaucoma, AMD, Cataract, etc.',
      source: 'ImageNet pretrained',
      icon: '&#128065;',
    },
    {
      key: 'resnet50_tumor_classifier',
      name: 'ResNet-50',
      task: 'Histopathology Tumor Classification',
      classes: 'Tumor, Normal, Stroma, Necrosis',
      source: 'ImageNet V2 pretrained',
      icon: '&#129516;',
    },
    {
      key: 'densenet121_pathology',
      name: 'DenseNet-121',
      task: 'Cancer Detection (Binary)',
      classes: 'Benign, Malignant',
      source: 'ImageNet pretrained',
      icon: '&#128300;',
    },
    {
      key: 'vit_b16_pathology',
      name: 'Vision Transformer B/16',
      task: 'Pathology Feature Extraction',
      classes: 'Benign, Malignant',
      source: 'ImageNet pretrained (Self-Attention)',
      icon: '&#129504;',
    },
    {
      key: 'densenet121_chest_xray',
      name: 'DenseNet-121 (CheXpert)',
      task: 'Chest X-ray Pathology (14 classes)',
      classes: 'Pneumonia, Cardiomegaly, Nodule, etc.',
      source: 'CheXpert-style head',
      icon: '&#129729;',
    },
    {
      key: 'unet_vessel_segmentation',
      name: 'U-Net 2D',
      task: 'Retinal Vessel Segmentation',
      classes: 'Background, Vessel',
      source: 'MONAI architecture',
      icon: '&#129657;',
    },
    {
      key: 'unet3d_brain_segmentation',
      name: 'U-Net 3D',
      task: 'Brain Tumor Segmentation',
      classes: 'Background, Tumor',
      source: 'MONAI 3D architecture',
      icon: '&#129504;',
    },
  ];

  return models.map(m => `
    <div class="process-item" style="padding:12px 16px">
      <span style="font-size:1.5rem">${m.icon}</span>
      <div class="process-info" style="flex:1">
        <div class="process-name">${m.name}</div>
        <div class="process-detail">${m.task}</div>
        <div style="font-size:.7rem;color:var(--text-muted);margin-top:2px">
          ${m.source} &middot; Classes: ${m.classes}
        </div>
      </div>
      <span class="badge badge-green" style="font-size:.7rem">SOTA</span>
    </div>`).join('');
}


// ===================== MODELS PAGE =====================
async function pageModels() {
  renderShell('Pretrained Models', 'State-of-the-art models for biomedical image analysis',
    '', '');

  const models = [
    {
      key: 'efficientnet_b4_retinal',
      name: 'EfficientNet-B4',
      arch: 'EfficientNet',
      task: 'Multi-label retinal disease classification',
      classes: ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertensive Retinopathy', 'Pathological Myopia', 'Other'],
      input: '3 x 380 x 380',
      source: 'torchvision (ImageNet pretrained)',
      ref: 'Tan & Le, 2019',
      modality: 'Retinal',
      color: '#58a6ff',
    },
    {
      key: 'resnet50_tumor_classifier',
      name: 'ResNet-50',
      arch: 'ResNet',
      task: 'Histopathology tile classification',
      classes: ['Tumor', 'Normal', 'Stroma', 'Necrosis'],
      input: '3 x 224 x 224',
      source: 'torchvision (ImageNet V2)',
      ref: 'He et al., 2016',
      modality: 'Histology',
      color: '#3fb950',
    },
    {
      key: 'densenet121_pathology',
      name: 'DenseNet-121',
      arch: 'DenseNet',
      task: 'Cancer detection (benign vs malignant)',
      classes: ['Benign', 'Malignant'],
      input: '3 x 224 x 224',
      source: 'torchvision (ImageNet pretrained)',
      ref: 'Huang et al., 2017',
      modality: 'Histology',
      color: '#d29922',
    },
    {
      key: 'vit_b16_pathology',
      name: 'Vision Transformer B/16',
      arch: 'ViT',
      task: 'Pathology feature extraction via self-attention',
      classes: ['Benign', 'Malignant'],
      input: '3 x 224 x 224',
      source: 'torchvision (ImageNet pretrained)',
      ref: 'Dosovitskiy et al., 2021',
      modality: 'Histology',
      color: '#bc8cff',
    },
    {
      key: 'densenet121_chest_xray',
      name: 'DenseNet-121 (CheXpert)',
      arch: 'DenseNet',
      task: 'Chest X-ray multi-label pathology detection',
      classes: ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax'],
      input: '3 x 224 x 224',
      source: 'CheXpert-style head',
      ref: 'Rajpurkar et al., 2017',
      modality: 'Radiology',
      color: '#f0883e',
    },
    {
      key: 'unet_vessel_segmentation',
      name: 'U-Net 2D (MONAI)',
      arch: 'U-Net',
      task: 'Binary retinal vessel segmentation',
      classes: ['Background', 'Vessel'],
      input: '1 x 512 x 512',
      source: 'MONAI UNet',
      ref: 'Ronneberger et al., 2015',
      modality: 'Retinal',
      color: '#f778ba',
    },
    {
      key: 'unet3d_brain_segmentation',
      name: 'U-Net 3D (MONAI)',
      arch: 'U-Net 3D',
      task: '3D brain tumor volumetric segmentation',
      classes: ['Background', 'Tumor'],
      input: '1 x 128 x 128 x 128',
      source: 'MONAI 3D UNet',
      ref: 'Cicek et al., 2016',
      modality: 'Radiology',
      color: '#39d353',
    },
  ];

  document.querySelector('.page-body').innerHTML = `
    <div class="mb-lg" style="max-width:700px">
      <p style="color:var(--text-secondary);line-height:1.7">
        VIDIO integrates <strong>state-of-the-art pretrained models</strong> from the research community.
        All models use <strong>ImageNet-pretrained backbones</strong> with task-specific classification heads,
        following the <strong>transfer learning</strong> paradigm that is standard in medical imaging.
        No custom training is required &mdash; models are downloaded and cached automatically on first use.
      </p>
    </div>

    <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:16px">
      ${models.map(m => `
        <div class="card" style="border-left:3px solid ${m.color}">
          <div class="card-header" style="padding:14px 16px">
            <div>
              <h3 style="margin-bottom:2px">${m.name}</h3>
              <div style="font-size:.75rem;color:var(--text-muted)">${m.arch} &middot; ${m.ref}</div>
            </div>
            <span class="badge badge-blue">${m.modality}</span>
          </div>
          <div class="card-body" style="padding:14px 16px">
            <div style="margin-bottom:10px;font-size:.875rem;color:var(--text-secondary)">${m.task}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:.8rem">
              <div><span style="color:var(--text-muted)">Input:</span> <code style="font-size:.75rem">${m.input}</code></div>
              <div><span style="color:var(--text-muted)">Classes:</span> ${m.classes.length}</div>
            </div>
            <div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:4px">
              ${m.classes.slice(0, 6).map(c => `<span class="badge badge-gray" style="font-size:.65rem">${c}</span>`).join('')}
              ${m.classes.length > 6 ? `<span class="badge badge-gray" style="font-size:.65rem">+${m.classes.length - 6} more</span>` : ''}
            </div>
            <div style="margin-top:10px;font-size:.7rem;color:var(--text-muted)">${m.source}</div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}
