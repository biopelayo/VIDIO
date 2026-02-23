/* ============================================================
   VIDIO Frontend — Chart Components (Pure Canvas, No Dependencies)
   Renders interactive charts for the visual analytics dashboard.
   ============================================================ */

/**
 * VIDIO Charts Library
 * ====================
 * Pure JavaScript + Canvas 2D chart rendering. No external charting
 * libraries required. Supports:
 *   - Donut/Pie charts (severity distribution)
 *   - Bar charts (findings by project/modality)
 *   - Line/Area charts (timeline trends)
 *   - Horizontal bar charts (model performance)
 *   - Heatmap grids (correlation matrices)
 */
const Charts = (() => {

  // ── Color Palette (matches dark medical theme) ──────────────
  const COLORS = {
    accent: '#58a6ff',
    success: '#3fb950',
    warning: '#d29922',
    danger: '#f85149',
    info: '#79b8ff',
    purple: '#bc8cff',
    pink: '#f778ba',
    orange: '#f0883e',
    teal: '#39d353',
    gray: '#8b949e',
  };

  const SEVERITY_COLORS = {
    CRITICAL: '#f85149',
    HIGH: '#f0883e',
    MEDIUM: '#d29922',
    LOW: '#3fb950',
  };

  const CATEGORY_COLORS = [
    '#58a6ff', '#3fb950', '#d29922', '#f85149',
    '#bc8cff', '#f778ba', '#f0883e', '#39d353',
    '#79b8ff', '#8b949e',
  ];

  // ── Utility: Get device pixel ratio for sharp rendering ─────
  function getScale(ctx) {
    return window.devicePixelRatio || 1;
  }

  function setupCanvas(canvas, width, height) {
    const scale = getScale();
    canvas.width = width * scale;
    canvas.height = height * scale;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(scale, scale);
    return ctx;
  }

  // ── Donut Chart ─────────────────────────────────────────────
  /**
   * Draw a donut chart with labels.
   * @param {string} canvasId - Canvas element ID.
   * @param {Array<{label: string, value: number, color: string}>} data
   * @param {Object} opts - {title, size, innerRadius}
   */
  function donut(canvasId, data, opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const size = opts.size || 260;
    const ctx = setupCanvas(canvas, size, size + 60);

    const total = data.reduce((s, d) => s + d.value, 0);
    if (total === 0) {
      ctx.fillStyle = '#8b949e';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No data', size / 2, size / 2);
      return;
    }

    const cx = size / 2;
    const cy = size / 2;
    const outerR = size / 2 - 20;
    const innerR = opts.innerRadius || outerR * 0.55;

    let startAngle = -Math.PI / 2;

    // Draw segments
    data.forEach((d, i) => {
      const sliceAngle = (d.value / total) * Math.PI * 2;
      const endAngle = startAngle + sliceAngle;

      ctx.beginPath();
      ctx.arc(cx, cy, outerR, startAngle, endAngle);
      ctx.arc(cx, cy, innerR, endAngle, startAngle, true);
      ctx.closePath();
      ctx.fillStyle = d.color || CATEGORY_COLORS[i % CATEGORY_COLORS.length];
      ctx.fill();

      startAngle = endAngle;
    });

    // Center text
    ctx.fillStyle = '#e6edf3';
    ctx.font = 'bold 28px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(total.toString(), cx, cy - 8);

    ctx.fillStyle = '#8b949e';
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText(opts.centerLabel || 'Total', cx, cy + 14);

    // Legend below
    const legendY = size + 10;
    const legendItemW = size / Math.min(data.length, 3);
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';

    data.forEach((d, i) => {
      const row = Math.floor(i / 3);
      const col = i % 3;
      const lx = col * (size / 3) + 8;
      const ly = legendY + row * 18;

      ctx.fillStyle = d.color || CATEGORY_COLORS[i % CATEGORY_COLORS.length];
      ctx.fillRect(lx, ly, 10, 10);
      ctx.fillStyle = '#e6edf3';
      ctx.fillText(`${d.label} (${d.value})`, lx + 14, ly + 9);
    });
  }


  // ── Bar Chart ───────────────────────────────────────────────
  /**
   * Draw a vertical bar chart.
   * @param {string} canvasId
   * @param {Array<{label: string, value: number, color?: string}>} data
   * @param {Object} opts - {title, width, height, barColor}
   */
  function bar(canvasId, data, opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const width = opts.width || 400;
    const height = opts.height || 220;
    const ctx = setupCanvas(canvas, width, height);

    const padding = { top: 30, right: 20, bottom: 40, left: 50 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    if (!data || data.length === 0) {
      ctx.fillStyle = '#8b949e';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No data', width / 2, height / 2);
      return;
    }

    const maxVal = Math.max(...data.map(d => d.value), 1);
    const barW = Math.min(chartW / data.length * 0.7, 50);
    const gap = (chartW - barW * data.length) / (data.length + 1);

    // Title
    if (opts.title) {
      ctx.fillStyle = '#e6edf3';
      ctx.font = 'bold 13px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(opts.title, padding.left, 18);
    }

    // Y-axis
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 1;
    const ySteps = 4;
    for (let i = 0; i <= ySteps; i++) {
      const y = padding.top + chartH - (i / ySteps) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();

      ctx.fillStyle = '#6e7681';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(Math.round(maxVal * i / ySteps), padding.left - 6, y + 3);
    }

    // Bars
    data.forEach((d, i) => {
      const x = padding.left + gap + i * (barW + gap);
      const barH = (d.value / maxVal) * chartH;
      const y = padding.top + chartH - barH;

      // Bar fill with gradient
      const grad = ctx.createLinearGradient(x, y, x, y + barH);
      const color = d.color || opts.barColor || COLORS.accent;
      grad.addColorStop(0, color);
      grad.addColorStop(1, color + '80');
      ctx.fillStyle = grad;

      // Rounded top corners
      const r = Math.min(4, barW / 2);
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + barW - r, y);
      ctx.quadraticCurveTo(x + barW, y, x + barW, y + r);
      ctx.lineTo(x + barW, y + barH);
      ctx.lineTo(x, y + barH);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.fill();

      // Value on top
      ctx.fillStyle = '#e6edf3';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(d.value, x + barW / 2, y - 5);

      // Label below
      ctx.fillStyle = '#8b949e';
      ctx.font = '10px Inter, sans-serif';
      ctx.save();
      ctx.translate(x + barW / 2, height - padding.bottom + 14);
      ctx.rotate(-0.3);
      ctx.fillText(d.label.length > 10 ? d.label.slice(0, 10) + '..' : d.label, 0, 0);
      ctx.restore();
    });
  }


  // ── Horizontal Bar Chart ────────────────────────────────────
  /**
   * Draw a horizontal bar chart (good for categories).
   * @param {string} canvasId
   * @param {Array<{label: string, value: number, color?: string}>} data
   * @param {Object} opts
   */
  function horizontalBar(canvasId, data, opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const width = opts.width || 380;
    const barH = 28;
    const gap = 6;
    const height = Math.max(data.length * (barH + gap) + 50, 100);
    const ctx = setupCanvas(canvas, width, height);

    const padding = { left: 120, right: 20, top: 30 };
    const chartW = width - padding.left - padding.right;

    if (!data || data.length === 0) return;

    const maxVal = Math.max(...data.map(d => d.value), 1);

    // Title
    if (opts.title) {
      ctx.fillStyle = '#e6edf3';
      ctx.font = 'bold 13px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(opts.title, 10, 18);
    }

    data.forEach((d, i) => {
      const y = padding.top + i * (barH + gap);
      const w = (d.value / maxVal) * chartW;
      const color = d.color || CATEGORY_COLORS[i % CATEGORY_COLORS.length];

      // Label
      ctx.fillStyle = '#e6edf3';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(d.label.length > 16 ? d.label.slice(0, 16) + '..' : d.label,
                    padding.left - 8, y + barH / 2 + 4);

      // Background track
      ctx.fillStyle = '#21262d';
      ctx.beginPath();
      ctx.roundRect(padding.left, y, chartW, barH, 4);
      ctx.fill();

      // Bar
      if (w > 0) {
        const grad = ctx.createLinearGradient(padding.left, y, padding.left + w, y);
        grad.addColorStop(0, color);
        grad.addColorStop(1, color + '99');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(padding.left, y, Math.max(w, 8), barH, 4);
        ctx.fill();
      }

      // Value
      ctx.fillStyle = '#e6edf3';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(d.value, padding.left + w + 6, y + barH / 2 + 4);
    });
  }


  // ── Sparkline (Mini Line Chart) ─────────────────────────────
  /**
   * Draw a compact sparkline chart.
   * @param {string} canvasId
   * @param {Array<number>} values
   * @param {Object} opts - {width, height, color, fillColor}
   */
  function sparkline(canvasId, values, opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const width = opts.width || 120;
    const height = opts.height || 40;
    const ctx = setupCanvas(canvas, width, height);

    if (!values || values.length < 2) return;

    const padding = 4;
    const chartW = width - padding * 2;
    const chartH = height - padding * 2;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    const color = opts.color || COLORS.accent;

    // Area fill
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = padding + (i / (values.length - 1)) * chartW;
      const y = padding + chartH - ((v - min) / range) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    // Close the area
    ctx.lineTo(padding + chartW, padding + chartH);
    ctx.lineTo(padding, padding + chartH);
    ctx.closePath();
    ctx.fillStyle = (opts.fillColor || color) + '20';
    ctx.fill();

    // Line
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = padding + (i / (values.length - 1)) * chartW;
      const y = padding + chartH - ((v - min) / range) * chartH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Last point dot
    const lastX = padding + chartW;
    const lastY = padding + chartH - ((values[values.length - 1] - min) / range) * chartH;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }


  // ── Progress Ring ───────────────────────────────────────────
  /**
   * Draw a progress ring (percentage indicator).
   * @param {string} canvasId
   * @param {number} value - Percentage 0-100.
   * @param {Object} opts - {size, color, label}
   */
  function progressRing(canvasId, value, opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const size = opts.size || 100;
    const ctx = setupCanvas(canvas, size, size + 20);

    const cx = size / 2;
    const cy = size / 2;
    const radius = size / 2 - 10;
    const lineW = 8;

    // Background ring
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = lineW;
    ctx.stroke();

    // Progress arc
    const endAngle = -Math.PI / 2 + (value / 100) * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, -Math.PI / 2, endAngle);
    ctx.strokeStyle = opts.color || (value > 75 ? COLORS.success : value > 40 ? COLORS.warning : COLORS.danger);
    ctx.lineWidth = lineW;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Center text
    ctx.fillStyle = '#e6edf3';
    ctx.font = 'bold 20px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${Math.round(value)}%`, cx, cy);

    // Label below
    if (opts.label) {
      ctx.fillStyle = '#8b949e';
      ctx.font = '10px Inter, sans-serif';
      ctx.fillText(opts.label, cx, size + 10);
    }
  }


  return {
    donut,
    bar,
    horizontalBar,
    sparkline,
    progressRing,
    COLORS,
    SEVERITY_COLORS,
    CATEGORY_COLORS,
  };
})();
