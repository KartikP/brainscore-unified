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

  // ---- ablation ----
  (function () {
    const a = D.ablation;
    const names = Object.keys(a.conditions);
    const means = names.map(n => mean(a.conditions[n]));
    const sems = names.map(n => sem(a.conditions[n]));
    const colors = names.map(n => ({
      'baseline': '#3b7dd8', 'lesioned': '#d8483b',
      'random control': '#9aa0a6', 'restored': '#3bb273',
    }[n] || '#6c6c6c'));
    const bar = {
      x: names, y: means, type: 'bar',
      error_y: { type: 'data', array: sems, color: '#c6d0e6' },
      marker: { color: colors },
      hovertemplate: '%{x}: %{y:.3f}<extra></extra>',
    };
    const lay = Object.assign({}, LAYOUT, {
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'ROAR score', rangemode: 'tozero' }),
      shapes: [{ type: 'line', x0: -0.5, x1: names.length - 0.5, y0: a.chance, y1: a.chance,
        line: { color: '#888', width: 1, dash: 'dot' } }],
      annotations: [{ x: names.length - 1, y: a.chance, yanchor: 'bottom', xanchor: 'right',
        text: 'chance', showarrow: false, font: { color: '#888', size: 11 } }],
    });
    Plotly.react('ablation-plot', [bar], lay, CFG);
    $('ablation-reading').textContent = a.reading;
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

  // ---- nulls table ----
  const nt = $('nulls-table');
  nt.innerHTML = '<tr><th>capability</th><th>matched null</th><th>what it catches</th></tr>' +
    D.nulls.entries.map(e =>
      `<tr><td>${e.capability}</td><td><code>${e.null}</code></td><td>${e.what_it_catches}</td></tr>`).join('');

  function mean(a){ return a.reduce((x, y) => x + y, 0) / a.length; }
  function sem(a){ const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length) / Math.sqrt(a.length); }
})();
