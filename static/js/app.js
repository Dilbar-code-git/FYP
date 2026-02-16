let uploadedFilepath = null;

document.addEventListener('DOMContentLoaded', () => {
  const uploadBtn = document.getElementById('uploadBtn');
  const previewBtn = document.getElementById('previewBtn');
  const trainBtn = document.getElementById('trainBtn');

  uploadBtn.addEventListener('click', uploadFile);
  previewBtn.addEventListener('click', previewData);
  trainBtn.addEventListener('click', trainModels);
});

async function uploadFile() {
  const input = document.getElementById('fileInput');
  const file = input.files[0];
  const result = document.getElementById('uploadResult');
  result.textContent = '';

  if (!file) { result.textContent = 'Please select a file.'; return; }

  const form = new FormData();
  form.append('file', file);

  try {
    const resp = await fetch('/api/upload/file', { method: 'POST', body: form });
    const data = await resp.json();
    if (resp.ok) {
      uploadedFilepath = data.filepath.replaceAll('\\\\', '/').replaceAll('\\', '/');
      result.innerHTML = `<strong>Uploaded:</strong> ${data.filename}`;
    } else {
      result.textContent = data.error || 'Upload failed';
    }
  } catch (err) {
    result.textContent = 'Upload error: ' + err.message;
  }
}

async function previewData() {
  const previewArea = document.getElementById('previewArea');
  previewArea.textContent = '';
  if (!uploadedFilepath) { previewArea.textContent = 'No file uploaded yet.'; return; }

  try {
    const resp = await fetch('/api/upload/preview', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ filepath: uploadedFilepath, rows: 5 }) });
    const data = await resp.json();
    if (resp.ok) {
      const table = renderTable(data.preview);
      previewArea.appendChild(table);
    } else {
      previewArea.textContent = data.error || 'Preview failed';
    }
  } catch (err) {
    previewArea.textContent = 'Preview error: ' + err.message;
  }
}

async function trainModels() {
  const trainResult = document.getElementById('trainResult');
  trainResult.textContent = '';
  const target = document.getElementById('targetColumn').value || 'sales';
  if (!uploadedFilepath) { trainResult.textContent = 'Upload a file first.'; return; }

  try {
    const payload = { filepath: uploadedFilepath, target_column: target, test_size: 0.2 };
    const resp = await fetch('/api/predict/train', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const data = await resp.json();
    if (resp.ok) {
      const best = data.best_model || {};
      const modelName = best.model_name || 'N/A';
      const metrics = best.metrics || {};
      const performance = metrics.performance || {};
      const errors = metrics.errors || {};
      const quality = metrics.quality_score || 0;
      
      // Build results HTML
      let html = `<div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0;">`;
      html += `<h3 style="color: #2c3e50; margin-top: 0;">🏆 Best Model: <strong>${modelName}</strong></h3>`;
      html += `<div style="background: white; padding: 12px; border-radius: 5px; margin: 10px 0;">`;
      html += `<p style="font-size: 18px; color: #e74c3c;"><strong>Quality Score: ${quality}</strong> / 100</p>`;
      html += `</div>`;
      
      // Performance Metrics
      html += `<h4 style="color: #27ae60;">📊 Performance Metrics:</h4>`;
      html += `<div style="background: #ecf0f1; padding: 10px; border-left: 4px solid #27ae60; border-radius: 4px;">`;
      html += `<p><strong>Accuracy:</strong> ${performance.accuracy || 'N/A'}%</p>`;
      html += `<p><strong>R² Score:</strong> ${performance.r2_score || 'N/A'}</p>`;
      html += `<p><strong>Adjusted R²:</strong> ${performance.adjusted_r2 || 'N/A'}</p>`;
      html += `<p><strong>Correlation:</strong> ${performance.correlation || 'N/A'}</p>`;
      html += `<p><strong>RMSE %:</strong> ${performance.rmse_percentage || 'N/A'}%</p>`;
      html += `</div>`;
      
      // Error Metrics
      html += `<h4 style="color: #e67e22;">❌ Error Metrics:</h4>`;
      html += `<div style="background: #ecf0f1; padding: 10px; border-left: 4px solid #e67e22; border-radius: 4px;">`;
      html += `<p><strong>MSE:</strong> ${errors.mse || 'N/A'}</p>`;
      html += `<p><strong>RMSE:</strong> ${errors.rmse || 'N/A'}</p>`;
      html += `<p><strong>MAE:</strong> ${errors.mae || 'N/A'}</p>`;
      html += `<p><strong>MAPE:</strong> ${errors.mape || 'N/A'}%</p>`;
      html += `</div>`;
      
      html += `</div>`;
      trainResult.innerHTML = html;
      
      // Show all models performance
      if (data.model_performance) {
        const allModelsDiv = document.createElement('div');
        allModelsDiv.style.marginTop = '20px';
        allModelsDiv.innerHTML = '<h4 style="color: #2c3e50;">📈 All Models Performance:</h4>';
        
        let modelsHtml = '<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">';
        for (const [modelName, modelMetrics] of Object.entries(data.model_performance)) {
          const perf = modelMetrics.performance || {};
          modelsHtml += `<div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #dee2e6;">`;
          modelsHtml += `<p style="margin: 0; font-weight: bold;">${modelName}</p>`;
          modelsHtml += `<p style="margin: 5px 0; font-size: 12px;"><strong>Accuracy:</strong> ${perf.accuracy || 'N/A'}%</p>`;
          modelsHtml += `<p style="margin: 5px 0; font-size: 12px;"><strong>R²:</strong> ${perf.r2_score || 'N/A'}</p>`;
          modelsHtml += `<p style="margin: 5px 0; font-size: 12px; color: #e74c3c;"><strong>Score:</strong> ${modelMetrics.quality_score || 0}</p>`;
          modelsHtml += `</div>`;
        }
        modelsHtml += '</div>';
        allModelsDiv.innerHTML += modelsHtml;
        trainResult.appendChild(allModelsDiv);
      }
    } else {
      trainResult.textContent = data.error || 'Training failed';
    }
  } catch (err) {
    trainResult.textContent = 'Training error: ' + err.message;
  }
}

function renderTable(rows) {
  const table = document.createElement('table');
  table.style.width = '100%';
  table.style.borderCollapse = 'collapse';
  if (!rows || rows.length === 0) { const n = document.createElement('div'); n.textContent='No preview available'; return n; }

  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  Object.keys(rows[0]).forEach(k => { const th = document.createElement('th'); th.textContent = k; th.style.textAlign='left'; th.style.padding='6px'; headerRow.appendChild(th); });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  rows.forEach(r => {
    const tr = document.createElement('tr');
    Object.values(r).forEach(v => { const td = document.createElement('td'); td.textContent = v; td.style.padding='6px'; tr.appendChild(td); });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  return table;
}
