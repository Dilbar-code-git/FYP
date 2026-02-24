// ============================================
// SalesIQ Dashboard — Fixed JavaScript
// ============================================

let charts   = {};
let appData  = {};

// ── NAVIGATION ──────────────────────────────
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
        e.preventDefault();
        switchTab(item.dataset.tab);
    });
});

function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const tab = document.getElementById('tab-' + tabId);
    const nav = document.querySelector(`[data-tab="${tabId}"]`);
    if (tab) tab.classList.add('active');
    if (nav) nav.classList.add('active');
    const titles = { upload:'Upload Data', analysis:'Sales Analysis',
                     training:'Model Training', results:'Prediction Results',
                     insights:'Business Insights' };
    document.getElementById('pageTitle').textContent = titles[tabId] || tabId;
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
}

// ── FILE UPLOAD ──────────────────────────────
const fileInput  = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');

uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', ()  => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) handleUpload(f);
});
fileInput.addEventListener('change', e => { if (e.target.files[0]) handleUpload(e.target.files[0]); });

function handleUpload(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['csv','xlsx','xls'].includes(ext)) {
        return showToast('Invalid file type. Please upload CSV or Excel.', 'error');
    }
    const fd = new FormData();
    fd.append('file', file);
    showLoading('Uploading and reading your file...');
    fetch('/upload', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
            hideLoading();
            if (data.error) return showToast('Upload error: ' + data.error, 'error');
            onUploadSuccess(data);
            showToast('File uploaded successfully!', 'success');
        })
        .catch(err => { hideLoading(); showToast('Upload failed: ' + err.message, 'error'); });
}

document.getElementById('sampleBtn').addEventListener('click', () => {
    showLoading('Loading sample dataset...');
    fetch('/sample-data')
        .then(r => r.json())
        .then(data => {
            hideLoading();
            if (data.error) return showToast(data.error, 'error');
            onUploadSuccess(data);
            showToast(data.message || 'Sample data loaded!', 'success');
        })
        .catch(err => { hideLoading(); showToast('Failed: ' + err.message, 'error'); });
});

function onUploadSuccess(data) {
    appData.uploadData = data;

    // Shape badge
    document.getElementById('shapeBadge').textContent =
        `${data.shape.rows} rows × ${data.shape.cols} cols`;
    document.getElementById('filePreviewCard').style.display = 'block';
    document.getElementById('colConfigCard').style.display   = 'block';

    // Build HTML preview table
    const cols = data.columns;
    let html = '<table><thead><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
    (data.preview || []).forEach(row => {
        html += '<tr>' + cols.map(c => `<td>${row[c] ?? ''}</td>`).join('') + '</tr>';
    });
    html += '</tbody></table>';
    document.getElementById('previewTable').innerHTML = html;

    // Populate column selectors
    const dateEl  = document.getElementById('dateColSelect');
    const salesEl = document.getElementById('salesColSelect');
    dateEl.innerHTML  = cols.map(c => `<option value="${c}">${c}</option>`).join('');
    salesEl.innerHTML = cols.map(c => `<option value="${c}">${c}</option>`).join('');

    if (data.date_candidates?.length)  dateEl.value  = data.date_candidates[0];
    if (data.sales_candidates?.length) salesEl.value = data.sales_candidates[0];

    updateStatus('File loaded ✓', true);
}

// ── ANALYZE ──────────────────────────────────
document.getElementById('analyzeBtn').addEventListener('click', doAnalyze);

function doAnalyze() {
    const dateCol  = document.getElementById('dateColSelect').value;
    const salesCol = document.getElementById('salesColSelect').value;
    if (!dateCol || !salesCol) return showToast('Please select both columns.', 'error');

    // Save to appData so train can use them
    appData.dateCol  = dateCol;
    appData.salesCol = salesCol;

    showLoading('Analyzing your data...');
    fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date_column: dateCol, sales_column: salesCol })
    })
    .then(r => r.json())
    .then(data => {
        hideLoading();
        if (data.error) return showToast('Analysis error: ' + data.error, 'error');
        appData.analysis = data;
        renderAnalysis(data);
        switchTab('analysis');
        showToast('Analysis complete!', 'success');
    })
    .catch(err => { hideLoading(); showToast('Analysis failed: ' + err.message, 'error'); });
}

function renderAnalysis(data) {
    const s = data.stats;
    document.getElementById('kpiTotal').textContent  = fmt(s.total_sales);
    document.getElementById('kpiAvg').textContent    = fmt(s.avg_sales);
    document.getElementById('kpiGrowth').textContent =
        (s.overall_growth >= 0 ? '+' : '') + s.overall_growth.toFixed(1) + '%';
    document.getElementById('kpiCount').textContent  = s.count;

    // Growth KPI colour
    const growthEl = document.getElementById('kpiGrowth');
    growthEl.style.color = s.overall_growth >= 0 ? '#10b981' : '#ef4444';

    // Time series
    destroyChart('timeSeriesChart');
    chartInstances('timeSeriesChart', 'line', {
        labels: data.chart_data.labels,
        datasets: [{ label: 'Sales', data: data.chart_data.values,
            borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.1)',
            fill: true, tension: 0.4, pointRadius: 2 }]
    });

    // Seasonal bar
    destroyChart('seasonalChart');
    const maxSeas = Math.max(...data.seasonal.values);
    chartInstances('seasonalChart', 'bar', {
        labels: data.seasonal.labels,
        datasets: [{ label: 'Avg Monthly Sales', data: data.seasonal.values,
            backgroundColor: data.seasonal.values.map(v =>
                v === maxSeas ? 'rgba(16,185,129,0.75)' : 'rgba(99,102,241,0.5)'),
            borderRadius: 6 }]
    }, { plugins: { legend: { display: false } } });

    // Annual bar
    destroyChart('annualChart');
    chartInstances('annualChart', 'bar', {
        labels: data.yearly.labels,
        datasets: [{ label: 'Annual Sales', data: data.yearly.values,
            backgroundColor: 'rgba(99,102,241,0.6)', borderRadius: 8 }]
    }, { plugins: { legend: { display: false } } });
}

// ── TRAIN ─────────────────────────────────────
document.getElementById('trainBtn').addEventListener('click',  doTrain);
document.getElementById('trainBtn2').addEventListener('click', doTrain);

function doTrain() {
    // Ensure we have column info (re-read from selectors if available)
    const dateEl  = document.getElementById('dateColSelect');
    const salesEl = document.getElementById('salesColSelect');
    if (dateEl.value)  appData.dateCol  = dateEl.value;
    if (salesEl.value) appData.salesCol = salesEl.value;

    if (!appData.dateCol || !appData.salesCol) {
        showToast('Please upload and analyze data first.', 'error');
        return switchTab('upload');
    }

    const periods = parseInt(document.getElementById('forecastPeriods').value) || 12;
    switchTab('training');
    startProgressAnim();

    fetch('/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            date_column:      appData.dateCol,
            sales_column:     appData.salesCol,
            forecast_periods: periods
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            showToast('Training error: ' + data.error, 'error');
            logLine('❌ Error: ' + data.error);
            return;
        }
        appData.results = data;
        finishProgressAnim(data);
        renderResults(data);
        renderInsights(data);
        showToast(`✅ Done! Best model: ${data.best_model}`, 'success');
    })
    .catch(err => {
        showToast('Training failed: ' + err.message, 'error');
        logLine('❌ Network error: ' + err.message);
    });
}

function startProgressAnim() {
    document.getElementById('trainingContainer').style.display  = 'none';
    document.getElementById('trainingProgress').style.display   = 'block';
    document.getElementById('trainingLog').innerHTML = '> Initialising training pipeline...\n';

    const list = [
        ['lr',   'Linear Regression'],
        ['rf',   'Random Forest'],
        ['xgb',  'XGBoost'],
        ['arima','ARIMA']
    ];
    list.forEach(([key, name], i) => {
        setTimeout(() => {
            setProgStatus(key, 'Training...', '#f59e0b');
            setProgFill(key, 55);
            logLine(`> Training ${name}...`);
        }, i * 350 + 200);
    });
}

function finishProgressAnim(data) {
    const map = { lr:'Linear Regression', rf:'Random Forest', xgb:'XGBoost', arima:'ARIMA' };
    Object.entries(map).forEach(([key, name], i) => {
        const m = data.metrics[name];
        setTimeout(() => {
            if (m) {
                const best = name === data.best_model;
                setProgFill(key, 100);
                setProgStatus(key,
                    `RMSE: ${m.RMSE.toLocaleString()} | R²: ${m.R2} ${best ? '🏆' : ''}`,
                    best ? '#10b981' : '#94a3b8');
                logLine(`> ${name}: MAE=${m.MAE.toLocaleString()}, RMSE=${m.RMSE.toLocaleString()}, R²=${m.R2}, MAPE=${m.MAPE}%`);
            } else {
                setProgFill(key, 100, '#475569');
                setProgStatus(key, 'Not available', '#64748b');
                logLine(`> ${name}: skipped (library not installed)`);
            }
        }, i * 250 + 100);
    });
    setTimeout(() => {
        logLine(`\n> ✅ Best model selected: ${data.best_model}`);
        logLine(`> Generated ${data.future_forecasts.length}-month forecast`);
        logLine('> All done!');
    }, 1400);
}

function setProgStatus(key, text, color) {
    const el = document.querySelector(`#prog-${key} .prog-status`);
    if (el) { el.textContent = text; el.style.color = color; }
}
function setProgFill(key, pct, color) {
    const el = document.getElementById(`fill-${key}`);
    if (el) { el.style.width = pct + '%'; if (color) el.style.background = color; }
}
function logLine(text) {
    const log = document.getElementById('trainingLog');
    if (log) { log.innerHTML += text + '\n'; log.scrollTop = log.scrollHeight; }
}

// ── RENDER RESULTS ────────────────────────────
const MODEL_COLORS = {
    'Linear Regression': '#6366f1',
    'Random Forest':     '#10b981',
    'XGBoost':           '#f59e0b',
    'ARIMA':             '#ec4899'
};

function renderResults(data) {
    // Best model banner
    const banner = document.getElementById('bestModelBanner');
    banner.style.display = 'block';
    document.getElementById('bestModelName').textContent = data.best_model;
    const bm = data.metrics[data.best_model];
    document.getElementById('bestModelMetrics').innerHTML =
        `<div><label>RMSE</label><span>${bm.RMSE.toLocaleString()}</span></div>
         <div><label>R² Score</label><span>${bm.R2}</span></div>
         <div><label>MAPE</label><span>${bm.MAPE}%</span></div>`;

    // Metrics table
    const tbody = document.getElementById('metricsBody');
    tbody.innerHTML = '';
    Object.entries(data.metrics)
          .sort((a, b) => a[1].RMSE - b[1].RMSE)
          .forEach(([name, m]) => {
        const best  = name === data.best_model;
        const r2cls = m.R2 > 0.8 ? 'badge-r2-good' : (m.R2 > 0.5 ? 'badge-r2-ok' : 'badge-r2-bad');
        tbody.innerHTML += `
            <tr style="${best ? 'background:rgba(16,185,129,0.06)' : ''}">
                <td>${best ? '🏆 ' : ''}<strong>${name}</strong></td>
                <td>${m.MAE.toLocaleString()}</td>
                <td>${m.RMSE.toLocaleString()}</td>
                <td class="${r2cls}">${m.R2}</td>
                <td>${m.MAPE}%</td>
                <td>${best ? '<span class="badge-best">✓ Best</span>' : '—'}</td>
            </tr>`;
    });

    // Actual vs Predicted
    destroyChart('predictionChart');
    const pvDatasets = [{
        label: 'Actual', data: data.test_actual,
        borderColor: '#fff', backgroundColor: 'transparent',
        borderWidth: 2.5, pointRadius: 5, tension: 0
    }];
    Object.entries(data.test_predictions).forEach(([name, preds]) => {
        pvDatasets.push({
            label: name, data: preds,
            borderColor: MODEL_COLORS[name] || '#94a3b8',
            backgroundColor: 'transparent',
            borderWidth: 2, tension: 0.3, pointRadius: 3,
            borderDash: name === data.best_model ? [] : [5, 3]
        });
    });
    chartInstances('predictionChart', 'line',
        { labels: data.test_dates, datasets: pvDatasets });

    // RMSE comparison bar
    destroyChart('rmseChart');
    const mNames = Object.keys(data.metrics).sort((a,b) => data.metrics[a].RMSE - data.metrics[b].RMSE);
    chartInstances('rmseChart', 'bar', {
        labels: mNames,
        datasets: [{
            label: 'RMSE (lower = better)',
            data: mNames.map(n => data.metrics[n].RMSE),
            backgroundColor: mNames.map(n =>
                n === data.best_model ? 'rgba(16,185,129,0.75)' : 'rgba(99,102,241,0.5)'),
            borderRadius: 8
        }]
    }, { plugins: { legend: { display: false } } });

    // Forecast chart
    destroyChart('forecastChart');
    const hDates = data.historical.map(h => h.date);
    const hVals  = data.historical.map(h => h.value);
    const fDates = data.future_forecasts.map(f => f.date);
    const fVals  = data.future_forecasts.map(f => f.value);
    // Bridge: repeat last historical value as first forecast point for visual continuity
    const bridgeF = hVals.length ? [hVals[hVals.length-1], ...fVals] : fVals;
    const bridgeD = hDates.length ? [hDates[hDates.length-1], ...fDates] : fDates;
    chartInstances('forecastChart', 'line', {
        labels: [...hDates, ...fDates],
        datasets: [
            { label: 'Historical', data: [...hVals, ...Array(fDates.length).fill(null)],
              borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.08)',
              fill: true, tension: 0.4, pointRadius: 2 },
            { label: `Forecast (${data.best_model})`,
              data: [...Array(hDates.length > 0 ? hDates.length - 1 : 0).fill(null), ...bridgeF],
              borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,0.08)',
              fill: true, tension: 0.4, borderDash: [6,3], pointRadius: 3 }
        ]
    });

    // Feature importance
    if (data.feature_importance?.length) {
        document.getElementById('featureImportanceCard').style.display = 'block';
        destroyChart('featureChart');
        const fi = data.feature_importance.slice(0, 10);
        chartInstances('featureChart', 'bar', {
            labels: fi.map(f => f.feature),
            datasets: [{ label: 'Importance', data: fi.map(f => f.importance),
                backgroundColor: 'rgba(99,102,241,0.65)', borderRadius: 6 }]
        }, { indexAxis: 'y', plugins: { legend: { display: false } } });
    }
}

// ── RENDER INSIGHTS ───────────────────────────
function renderInsights(data) {
    const grid = document.getElementById('insightsGrid');
    if (!data.insights?.length) return;
    grid.innerHTML = data.insights.map(ins => `
        <div class="insight-card">
            <div class="insight-icon">${ins.icon}</div>
            <div class="insight-title">${ins.title}</div>
            <div class="insight-text">${ins.text}</div>
        </div>`).join('');

    // Forecast detail table
    if (data.future_forecasts?.length) {
        const last3avg = data.historical.length >= 3
            ? data.historical.slice(-3).reduce((s,h) => s + h.value, 0) / 3
            : (data.historical[0]?.value || 0);

        document.getElementById('forecastTableCard').style.display = 'block';
        document.getElementById('forecastTableBody').innerHTML = data.future_forecasts.map((f, i) => {
            const diff = last3avg ? (f.value - last3avg) / last3avg * 100 : 0;
            const cls  = diff >= 0 ? 'badge-r2-good' : 'badge-r2-bad';
            return `<tr>
                <td>${i + 1}</td>
                <td>${f.date}</td>
                <td><strong>${fmt(f.value)}</strong></td>
                <td class="${cls}">${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%</td>
            </tr>`;
        }).join('');
    }
}

// ── CHART FACTORY ─────────────────────────────
function chartInstances(id, type, chartData, extraOpts = {}) {
    const ctx  = document.getElementById(id).getContext('2d');
    const base = baseOpts();
    charts[id] = new Chart(ctx, {
        type,
        data: chartData,
        options: mergeDeep(base, extraOpts)
    });
}

function destroyChart(id) {
    if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

function baseOpts() {
    return {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: { labels: { color: '#94a3b8', font: { family: 'Inter' }, boxWidth: 12 } },
            tooltip: {
                backgroundColor: '#1e293b', titleColor: '#f1f5f9',
                bodyColor: '#94a3b8', borderColor: 'rgba(99,102,241,0.3)', borderWidth: 1,
                callbacks: { label: ctx => ' ' + (ctx.dataset.label || '') + ': ' + fmt(ctx.raw) }
            }
        },
        scales: {
            x: { ticks: { color:'#64748b', font:{ size:10 } }, grid: { color:'rgba(255,255,255,0.04)' } },
            y: { ticks: { color:'#64748b', font:{ size:10 }, callback: v => fmt(v) },
                 grid: { color:'rgba(255,255,255,0.04)' } }
        }
    };
}

function mergeDeep(target, source) {
    const out = Object.assign({}, target);
    for (const key of Object.keys(source)) {
        if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
            out[key] = mergeDeep(target[key] || {}, source[key]);
        } else {
            out[key] = source[key];
        }
    }
    return out;
}

// ── HELPERS ───────────────────────────────────
function fmt(n) {
    if (n === null || n === undefined || isNaN(n)) return '—';
    n = parseFloat(n);
    if (n >= 1e6) return '$' + (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
    return '$' + n.toFixed(2);
}

function showLoading(text) {
    document.getElementById('loadingText').textContent = text || 'Processing...';
    document.getElementById('uploadLoading').style.display = 'flex';
}
function hideLoading() {
    document.getElementById('uploadLoading').style.display = 'none';
}

function updateStatus(text, loaded) {
    const b = document.getElementById('statusBadge');
    b.className = 'status-badge' + (loaded ? ' loaded' : '');
    b.innerHTML = `<i class="fas fa-circle"></i> ${text}`;
}

function showToast(message, type = 'info') {
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
    document.getElementById('toastContainer').appendChild(t);
    setTimeout(() => t.remove(), 5000);
}
