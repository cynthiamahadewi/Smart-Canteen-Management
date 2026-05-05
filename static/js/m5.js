let currentMode = 'upload';

function switchTab(mode) {
  currentMode = mode;
  document.getElementById('mode-upload').style.display =
    mode === 'upload' ? 'block' : 'none';
  document.getElementById('mode-preset').style.display =
    mode === 'preset' ? 'block' : 'none';
  document.getElementById('tab-upload').classList.toggle('active', mode === 'upload');
  document.getElementById('tab-preset').classList.toggle('active', mode === 'preset');
}

// ── Upload mode ──────────────────────────────────────────────────────────────

function handleUpload(files) {
  if (!files || !files.length) return;
  const file = files[0];

  // show preview
  const url = URL.createObjectURL(file);
  document.getElementById('upload-img').src = url;
  document.getElementById('upload-preview').style.display = 'block';

  // send to model
  const form = new FormData();
  form.append('image', file);
  runPredict(form);
}

// drag & drop
const dz = document.getElementById('dropzone');
dz.addEventListener('dragover', e => {
  e.preventDefault();
  dz.style.background = '#e6f1fb';
});
dz.addEventListener('dragleave', () => dz.style.background = '');
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.style.background = '';
  handleUpload(e.dataTransfer.files);
});

// ── Preset mode ──────────────────────────────────────────────────────────────

function previewPreset() {
  const name = document.getElementById('presetSelect').value;
  if (!name) {
    document.getElementById('preset-preview').style.display = 'none';
    return;
  }
  document.getElementById('before-img').src = `/static/presets/${name}_before.jpg`;
  document.getElementById('after-img').src  = `/static/presets/${name}.jpg`;
  document.getElementById('preset-preview').style.display = 'block';
}

function runPreset() {
  const name = document.getElementById('presetSelect').value;
  if (!name) return;
  const form = new FormData();
  form.append('preset', name);
  runPredict(form);
}

// ── Shared predict call ──────────────────────────────────────────────────────

function runPredict(formData) {
  // hide idle, show loading state
  document.getElementById('result-idle').style.display = 'none';
  document.getElementById('result-body').style.display = 'none';

  fetch('/api/predict', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.error) { alert('Error: ' + data.error); return; }
      displayResult(data);
    })
    .catch(err => alert('Request failed: ' + err));
}

function displayResult(data) {
  document.getElementById('result-idle').style.display = 'none';
  document.getElementById('result-body').style.display = 'block';

  // consumption ratio
  const pct = Math.round(data.pred_r * 100);
  document.getElementById('res-ratio').textContent = pct + '%';
  document.getElementById('res-bar').style.width = pct + '%';

  // bin pill
  const pill = document.getElementById('res-bin');
  pill.textContent = data.bin;
  pill.className = 'bin-pill';
  if (data.pred_r <= 0.2)      pill.classList.add('low');
  else if (data.pred_r <= 0.6) pill.classList.add('mid');

  // CORAL thresholds
  const grid = document.getElementById('coral-grid');
  grid.innerHTML = '';
  if (data.thresholds && data.thresholds.length) {
    data.thresholds.forEach(t => {
      const pct = Math.round(t.value * 100);
      grid.innerHTML += `
        <div class="coral-card">
          <div class="coral-label">${t.label}</div>
          <div class="coral-val">${pct}%</div>
          <div class="coral-bar">
            <div class="coral-fill" style="width:${pct}%"></div>
          </div>
        </div>`;
    });
  }

  // Top-3 food
  const list = document.getElementById('top3-list');
  list.innerHTML = '';
  data.top3_food.forEach(item => {
    const pct = Math.round(item.prob * 100);
    list.innerHTML += `
      <div class="top3-row">
        <span>${item.name}</span>
        <div class="top3-prob-wrap">
          <div class="top3-bar">
            <div class="top3-fill" style="width:${pct}%"></div>
          </div>
          <span class="top3-pct">${pct}%</span>
        </div>
      </div>`;
  });
}