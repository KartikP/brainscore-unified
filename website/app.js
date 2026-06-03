(function () {
  const D = window.BSU_DATA;
  const $ = (id) => document.getElementById(id);

  const LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#c6d0e6', family: 'Inter, sans-serif', size: 13 },
    margin: { l: 56, r: 24, t: 18, b: 70 }, showlegend: false,
    xaxis: { gridcolor: '#1d2740', zerolinecolor: '#1d2740' },
    yaxis: { gridcolor: '#1d2740', zerolinecolor: '#1d2740' },
  };
  const CFG = { displayModeBar: false, responsive: true };

  // ---- hero ----
  $('hero-sub').textContent = D.meta.subtitle;
  $('hero-note').textContent = D.meta.note;
  $('provenance').textContent = D.meta.provenance;

  // ---- capability cards ----
  const CAPS = [
    ['Neural encoding', 'process(StimulusSet)', 'Predict V4/IT, language, or whole-cortex responses from model features.'],
    ['Behavior', 'start_task(...)', 'Logistic readout or instruction-following generation, scored against human accuracy.'],
    ['State change', 'process(StateChange)', 'Lesion a unit population, observe the effect, restore bit-for-bit.'],
    ['Embodied', 'process(EnvironmentStep)', 'Close the loop: the model acts, the environment responds, repeat.'],
    ['Temporal · multimodal', 'synchronize_modalities(...)', 'Align per-frame / per-token / per-sample features onto the brain’s TR grid.'],
    ['Layer mapping', "start_recording('all')", 'Standard region→layer, whole-brain, or a CompositeSelector across layers.'],
    ['Topographic metric', 'TopographicMetric()', 'Score a model’s spatial unit layout against cortical topography (TDANN / TopoLM).'],
    ['Visualization', 'cortical_surface_map(...)', 'Map per-parcel scores onto the inflated cortex — publication-style figures.'],
  ];
  $('cap-grid').innerHTML = CAPS.map(([t, api, d]) =>
    `<div class="cap-card"><div class="tag">capability</div><h3>${t}</h3>
     <p>${d}</p><div class="api"><code>${api}</code></div></div>`).join('');

  // ---- input -> brain response ----
  const toggles = $('input-toggles');
  D.inputs.forEach((inp, i) => {
    const b = document.createElement('button');
    b.className = 'tog' + (i === 0 ? ' active' : '');
    b.textContent = inp.type;
    b.onclick = () => selectInput(i, b);
    toggles.appendChild(b);
  });
  function slug(t){ return t.replace(/[^a-z]/gi, '').toLowerCase(); }
  function selectInput(i, btn) {
    document.querySelectorAll('.tog').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    const inp = D.inputs[i];
    $('resp-type').textContent = inp.type + ' input';
    $('resp-desc').textContent = inp.desc;
    $('resp-bench').textContent = inp.benchmark;
    $('resp-example').textContent = inp.example;
    const img = $('cortex-img');
    const candidate = 'assets/cortex_' + slug(inp.type) + '.png';
    img.onerror = () => { img.onerror = null; img.src = 'assets/cortex_demo.png'; };
    img.src = candidate;
  }
  selectInput(0, document.querySelector('.tog'));

  // ---- layer contribution per modality (MIRAGE Fig 4 analogue) ----
  if (D.layer_contribution) {
    const lc = D.layer_contribution;
    $('lc-title').textContent = lc.title;
    $('lc-sub').textContent = lc.subtitle;
    $('lc-reading').textContent = lc.reading;
    const maxlen = Math.max(...lc.order.map(m => lc.values[m].length));
    const z = lc.order.map(m => {
      const v = lc.values[m]; const mx = Math.max.apply(null, v);
      const norm = v.map(x => x / mx);
      while (norm.length < maxlen) norm.push(null);
      return norm;
    });
    const heat = {
      z: z, y: lc.labels, x: Array.from({ length: maxlen }, (_, i) => i),
      type: 'heatmap', colorscale: 'Magma', colorbar: { title: 'rel.', thickness: 12 },
      hovertemplate: '%{y}, layer %{x}: %{z:.2f}<extra></extra>',
    };
    const lay = Object.assign({}, LAYOUT, {
      margin: { l: 110, r: 24, t: 12, b: 50 },
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'layer', dtick: 2 }),
      yaxis: { gridcolor: '#1d2740', automargin: true },
    });
    Plotly.react('lc-plot', [heat], lay, CFG);
  }

  // ---- benchmark mechanics ----
  if (D.benchmark_mechanics) {
    const m = D.benchmark_mechanics;
    $('mech-title').textContent = m.title;
    $('mech-sub').textContent = m.subtitle;
    $('mech-task').textContent = m.task;
    $('mech-paths').innerHTML = m.paths.map(p =>
      `<div class="mech-path"><div class="mech-path-name">${p.name}</div>
       <div class="mech-path-models">${p.models}</div>
       <div class="mech-path-how">${p.how}</div></div>`).join('');
    $('mech-answer').textContent = m.answer;
    const bar = {
      x: m.floors.map(f => f.label), y: m.floors.map(f => f.value), type: 'bar',
      marker: { color: ['#9aa0a6', '#9aa0a6', '#3b7dd8'] },
      hovertemplate: '%{x}: %{y:.2f}<extra></extra>',
    };
    const lay = Object.assign({}, LAYOUT, {
      margin: { l: 44, r: 16, t: 10, b: 70 },
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'ROAR (raw)', range: [0.45, 0.75] }),
    });
    Plotly.react('mech-floors-plot', [bar], lay, CFG);
  }

  // ---- all models, stratified by input type x output path ----
  if (D.all_paths) {
    const ap = D.all_paths;
    $('allpaths-title').textContent = ap.title;
    $('allpaths-sub').textContent = ap.subtitle;
    $('allpaths-reading').textContent = ap.reading;
    // one trace per path (colour = path), points at x=model, y=score
    const traces = Object.keys(ap.pathColors).map(path => {
      const rows = ap.rows.filter(r => r.path === path && r.score != null);
      return {
        x: rows.map(r => r.model), y: rows.map(r => r.score),
        text: rows.map(r => r.input), type: 'scatter', mode: 'markers', name: path,
        marker: { color: ap.pathColors[path], size: 15, line: { color: '#0a0e17', width: 1 } },
        hovertemplate: '%{x} · %{text} · ' + path + ': %{y:.2f}<extra></extra>',
      };
    });
    const lay = Object.assign({}, LAYOUT, {
      showlegend: true,
      legend: { orientation: 'h', x: 0, y: 1.12, font: { size: 10 }, bgcolor: 'rgba(0,0,0,0)' },
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'ROAR raw accuracy', range: [0.42, 1.08] }),
      xaxis: Object.assign({}, LAYOUT.xaxis, { categoryorder: 'array', categoryarray: ap.models,
        title: 'model  (worst → best)' }),
      shapes: [
        { type: 'line', x0: -0.5, x1: ap.models.length - 0.5, y0: ap.chance, y1: ap.chance,
          line: { color: '#888', width: 1, dash: 'dot' } },
        { type: 'line', x0: -0.5, x1: ap.models.length - 0.5, y0: ap.null_floor, y1: ap.null_floor,
          line: { color: '#d8483b', width: 1, dash: 'dot' } }],
      annotations: [
        { x: ap.models.length - 1, y: ap.chance, yanchor: 'top', xanchor: 'right',
          text: 'chance', showarrow: false, font: { color: '#888', size: 10 } },
        { x: ap.models.length - 1, y: ap.null_floor, yanchor: 'bottom', xanchor: 'right',
          text: 'random-feature floor', showarrow: false, font: { color: '#d8483b', size: 10 } }],
    });
    Plotly.react('allpaths-plot', traces, lay, CFG);
    // table
    const fmt = (r) => `<tr><td>${r.model}</td><td>${r.input}</td>`
      + `<td><span class="path-chip" style="background:${ap.pathColors[r.path] || '#666'}">${r.path}</span></td>`
      + `<td>${r.score == null ? '—' : r.score.toFixed(3)}</td></tr>`;
    $('allpaths-table').innerHTML =
      '<tr><th>model</th><th>input</th><th>output path</th><th>raw</th></tr>'
      + ap.rows.map(fmt).join('');
  }

  // ---- scaling curves ----
  const scKeys = Object.keys(D.scaling);
  const scTabs = $('scaling-tabs');
  scKeys.forEach((k, i) => {
    const b = document.createElement('button');
    b.className = 'tog' + (i === 0 ? ' active' : '');
    b.textContent = D.scaling[k].capability.split('—')[0].trim();
    b.onclick = () => drawScaling(k, b);
    scTabs.appendChild(b);
  });
  function drawScaling(key, btn) {
    document.querySelectorAll('#scaling-tabs .tog').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    const s = D.scaling[key];
    const curve = {
      x: s.models, y: s.scores, type: 'scatter', mode: 'lines+markers',
      line: { color: '#5b8cff', width: 3 }, marker: { size: 11, color: '#7c5bff' },
      hovertemplate: '%{x}: %{y:.3f}<extra></extra>',
    };
    const floor = {
      x: s.models, y: s.models.map(() => s.null_floor), type: 'scatter', mode: 'lines',
      line: { color: '#d8483b', width: 1.6, dash: 'dash' }, hoverinfo: 'skip',
    };
    const lay = Object.assign({}, LAYOUT, {
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'score', rangemode: 'tozero' }),
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'model  (worse → better)' }),
      annotations: [{ x: s.models.length - 1, y: s.null_floor, xanchor: 'right',
        yanchor: 'bottom', text: 'null floor', showarrow: false,
        font: { color: '#d8483b', size: 11 } }],
    });
    Plotly.react('scaling-plot', [floor, curve], lay, CFG);
    $('scaling-reading').textContent = s.reading;
  }
  drawScaling(scKeys[0], document.querySelector('#scaling-tabs .tog'));

  // ---- models glossary ----
  if (D.models_glossary) {
    const mg = D.models_glossary;
    $('models-title').textContent = mg.title;
    $('models-sub').textContent = mg.subtitle;
    $('model-grid').innerHTML = mg.models.map(x =>
      `<div class="model-card"><div class="model-kind">${x.kind}</div>`
      + `<h3>${x.name}</h3><p>${x.desc}</p></div>`).join('');
  }

  // ---- embodied VLM game ----
  if (D.embodied_game) {
    const g = D.embodied_game;
    $('game-title').textContent = g.title;
    $('game-sub').textContent = g.subtitle;
    $('game-reading').textContent = g.reading;
    const bar = {
      x: g.models, y: g.success, type: 'bar', marker: { color: g.colors },
      hovertemplate: '%{x}: %{y:.2f} success<extra></extra>',
    };
    const lay = Object.assign({}, LAYOUT, {
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'success rate', range: [0, 1.08] }),
      shapes: [{ type: 'line', x0: -0.5, x1: g.models.length - 0.5, y0: g.null_floor, y1: g.null_floor,
        line: { color: '#888', width: 1, dash: 'dot' } }],
      annotations: [{ x: g.models.length - 1, y: g.null_floor, yanchor: 'bottom', xanchor: 'right',
        text: 'random floor', showarrow: false, font: { color: '#888', size: 10 } }],
    });
    Plotly.react('game-plot', [bar], lay, CFG);
  }

  // ---- ablation (Honarmand dissociation) ----
  (function () {
    const a = D.ablation;
    const vwf = {
      x: a.mask_pct, y: a.vwf_roar, type: 'scatter', mode: 'lines+markers',
      name: 'VWF-selective ablation', line: { color: '#d8483b', width: 3 },
      marker: { size: 8 },
      error_y: { type: 'data', array: a.vwf_roar_sd, color: '#d8483b', thickness: 1 },
      hovertemplate: 'VWF, %{x}%: ROAR %{y:.2f}<extra></extra>',
    };
    const rnd = {
      x: a.mask_pct, y: a.random_roar, type: 'scatter', mode: 'lines+markers',
      name: 'random ablation', line: { color: '#9aa0a6', width: 3, dash: 'dot' },
      marker: { size: 8 },
      error_y: { type: 'data', array: a.random_roar_sd, color: '#9aa0a6', thickness: 1 },
      hovertemplate: 'random, %{x}%: ROAR %{y:.2f}<extra></extra>',
    };
    const lay = Object.assign({}, LAYOUT, {
      showlegend: true,
      legend: { x: 0.02, y: 0.12, font: { size: 10 }, bgcolor: 'rgba(0,0,0,0)' },
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'ROAR accuracy', range: [0.45, 1.05] }),
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'units ablated (% of MLP gate_proj)',
        range: [-1.5, Math.max(...a.mask_pct) + 2.5], autorange: false }),
      shapes: [{ type: 'line', x0: -1, x1: Math.max(...a.mask_pct) + 2, y0: a.threshold, y1: a.threshold,
        line: { color: '#e0a13b', width: 1.5, dash: 'dash' } }],
      annotations: [{ x: Math.max(...a.mask_pct), y: a.threshold, yanchor: 'bottom', xanchor: 'right',
        text: 'dyslexia threshold (0.65)', showarrow: false, font: { color: '#e0a13b', size: 10 } }],
    });
    Plotly.react('ablation-plot', [vwf, rnd], lay, CFG);
    if (a.protocol && document.getElementById('ablation-protocol'))
      $('ablation-protocol').textContent = a.protocol;
    $('ablation-reading').textContent = a.reading;
    if (a.brain_caption && document.getElementById('lesion-brain-cap'))
      $('lesion-brain-cap').textContent = a.brain_caption;
  })();

  // ---- selection ----
  (function () {
    const s = D.selection;
    const frac = s.selected_counts.map(c => c / s.units_per_layer);
    const bar = {
      x: s.layers, y: s.selected_counts, type: 'bar',
      marker: { color: '#6a3d9a' },
      hovertemplate: '%{x}: %{y} units<extra></extra>',
    };
    const lay = Object.assign({}, LAYOUT, {
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'selected units' }),
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'layer' }),
    });
    Plotly.react('selection-plot', [bar], lay, CFG);
    $('selection-reading').textContent = s.reading;
  })();

  // ---- temporal-shift validation (real BOLD) ----
  if (D.temporal_shift_validation) {
    const tv = D.temporal_shift_validation;
    $('shift-title').textContent = tv.title;
    $('shift-sub').textContent = tv.subtitle;
    $('shift-reading').textContent = tv.reading;
    const curve = {
      x: tv.shifts, y: tv.scores, type: 'scatter', mode: 'lines+markers',
      line: { color: '#3bb273', width: 3 }, marker: { size: 9, color: '#3bb273' },
      hovertemplate: 'delay %{x} TRs: r=%{y:.3f}<extra></extra>',
    };
    const floor = {
      x: tv.shifts, y: tv.shifts.map(() => tv.shuffle_floor), type: 'scatter',
      mode: 'lines', line: { color: '#d8483b', width: 1.5, dash: 'dash' }, hoverinfo: 'skip',
    };
    const lay = Object.assign({}, LAYOUT, {
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'median per-parcel r', rangemode: 'tozero' }),
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'HRF delay applied (TRs) — true ≈ +3' }),
      shapes: [{ type: 'line', x0: tv.true_delay, x1: tv.true_delay, y0: 0, y1: Math.max(...tv.scores),
        line: { color: '#5b8cff', width: 1.5, dash: 'dot' } }],
      annotations: [
        { x: tv.true_delay, y: Math.max(...tv.scores), yanchor: 'bottom', text: 'true delay',
          showarrow: false, font: { color: '#5b8cff', size: 11 } },
        { x: tv.shifts[tv.shifts.length - 1], y: tv.shuffle_floor, yanchor: 'bottom', xanchor: 'right',
          text: 'shuffle floor', showarrow: false, font: { color: '#d8483b', size: 11 } },
      ],
    });
    Plotly.react('shift-plot', [floor, curve], lay, CFG);
  }

  // ---- limitations ----
  if (D.limitations) {
    $('lim-title').textContent = D.limitations.title;
    $('lim-list').innerHTML = D.limitations.items
      .map(t => `<li>${t}</li>`).join('');
  }

  // ---- nulls table ----
  const nt = $('nulls-table');
  nt.innerHTML = '<tr><th>capability</th><th>matched null</th><th>what it catches</th></tr>' +
    D.nulls.entries.map(e =>
      `<tr><td>${e.capability}</td><td><code>${e.null}</code></td><td>${e.what_it_catches}</td></tr>`).join('');

  function mean(a){ return a.reduce((x, y) => x + y, 0) / a.length; }
  function sem(a){ const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length) / Math.sqrt(a.length); }

  // ---- force all Plotly charts to fill their containers ----
  // Plotly can compute a too-narrow width when it renders before CSS grid
  // layout settles (charts ending up in the left half of the panel). Resize
  // every plot after layout, again shortly after, and on window resize.
  function resizeAllPlots() {
    document.querySelectorAll('.plot').forEach(el => {
      if (el && el.classList.contains('js-plotly-plot')) {
        try { Plotly.Plots.resize(el); } catch (e) { /* not yet drawn */ }
      }
    });
  }
  window.addEventListener('resize', resizeAllPlots);
  // run after the current layout pass, and again once fonts/CDN settle
  requestAnimationFrame(resizeAllPlots);
  setTimeout(resizeAllPlots, 150);
  setTimeout(resizeAllPlots, 700);
})();
