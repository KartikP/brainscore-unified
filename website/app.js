(function () {
  const D = window.BSU_DATA;
  const $ = (id) => document.getElementById(id);

  // Highlight take-homes: **text** in a reading becomes an accented <b>.
  // HTML-escaped first so reading strings stay safe to inject.
  const _esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const mark = s => _esc(s).replace(/\*\*(.+?)\*\*/g, '<b class="hl">$1</b>');
  // Set a reading element from a string that may contain **highlights**.
  const setReading = (id, s) => { const el = $(id); if (el) el.innerHTML = mark(s || ''); };

  const LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#1b2333', family: 'Inter, sans-serif', size: 13 },
    margin: { l: 56, r: 24, t: 18, b: 70 }, showlegend: false,
    xaxis: { gridcolor: '#dde3ee', zerolinecolor: '#dde3ee' },
    yaxis: { gridcolor: '#dde3ee', zerolinecolor: '#dde3ee' },
  };
  const CFG = { displayModeBar: false, responsive: true };

  // ---- hero ----
  $('hero-sub').textContent = D.meta.subtitle;
  $('hero-note').textContent = D.meta.note;
  $('provenance').textContent = D.meta.provenance;

  // ---- hero: swipe through registrations to show the call is invariant ----
  // Most entries share one skeleton (only the model/benchmark/comment change);
  // the embodied and perturbation entries carry their own `lines` because they
  // run through a different process() variant — that contrast IS the point.
  (function () {
    const rot = D.hero_rotation;
    const body = $('hero-code-body');
    if (!rot || !rot.length || !body) return;
    const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const hl = line => {
      let s = esc(line);
      s = s.replace(/("[^"]*")/g, '<span class="s">$1</span>');     // strings
      s = s.replace(/\b(from|import)\b/g, '<span class="k">$1</span>'); // keywords
      s = s.replace(/(#.*)$/, '<span class="c">$1</span>');         // trailing comment
      return s;
    };
    // Standard entries share ONE fixed skeleton; only the model/benchmark/comment
    // tokens change, so we animate just those (the surrounding call stays put).
    const skeleton = e =>
      '<span class="k">from</span> brainscore <span class="k">import</span> load_model, load_benchmark\n' +
      'model = load_model(<span class="s hero-tok" id="hero-model">"' + esc(e.model) + '"</span>)\n' +
      'score = load_benchmark(<span class="s hero-tok" id="hero-bench">"' + esc(e.benchmark) + '"</span>)(model)\n' +
      '<span class="c hero-tok" id="hero-comment">' + esc(e.comment) + '</span>';
    // Custom entries (embodied / perturbation) have a different call shape, so
    // the whole block changes — those swipe as one.
    const renderFull = e => { body.innerHTML = e.lines ? e.lines.map(hl).join('\n') : skeleton(e); };
    renderFull(rot[0]);
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce || rot.length < 2) return;          // respect reduced-motion: no cycling
    let i = 0;
    setInterval(() => {
      const prev = rot[i];
      i = (i + 1) % rot.length;
      const e = rot[i];
      const toks = [$('hero-model'), $('hero-bench'), $('hero-comment')];
      if (!e.lines && !prev.lines && toks.every(Boolean)) {
        // standard → standard: animate ONLY the changed tokens
        toks.forEach(t => t.classList.add('swap-out'));
        setTimeout(() => {
          $('hero-model').textContent = '"' + e.model + '"';
          $('hero-bench').textContent = '"' + e.benchmark + '"';
          $('hero-comment').textContent = e.comment;
          toks.forEach(t => { t.classList.remove('swap-out'); t.classList.add('swap-in'); });
          setTimeout(() => toks.forEach(t => t.classList.remove('swap-in')), 440);
        }, 300);
      } else {
        // to/from a custom-shape entry: swipe the whole block
        body.classList.add('swap-out');
        setTimeout(() => {
          renderFull(e);
          body.classList.remove('swap-out'); body.classList.add('swap-in');
          setTimeout(() => body.classList.remove('swap-in'), 440);
        }, 300);
      }
    }, 3200);
  })();

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
    const candidate = 'assets/cortex_' + slug(inp.type) + '.png?v=' + (window.ASSET_V || '3');
    img.onerror = () => { img.onerror = null; img.src = 'assets/cortex_demo.png?v=' + (window.ASSET_V || '3'); };
    img.src = candidate;
  }
  selectInput(0, document.querySelector('.tog'));

  // ---- layer contribution per modality (MIRAGE Fig 4 analogue) ----
  if (D.layer_contribution) {
    const lc = D.layer_contribution;
    $('lc-title').textContent = lc.title;
    $('lc-sub').textContent = lc.subtitle;
    setReading('lc-reading', lc.reading);
    const maxlen = Math.max(...lc.order.map(m => lc.values[m].length));
    const z = lc.order.map(m => {
      const v = lc.values[m]; const mx = Math.max.apply(null, v);
      const norm = v.map(x => x / mx);
      while (norm.length < maxlen) norm.push(null);
      return norm;
    });
    const heat = {
      z: z, y: lc.labels, x: Array.from({ length: maxlen }, (_, i) => i),
      type: 'heatmap', colorscale: 'Turbo', colorbar: { title: 'rel.', thickness: 12 },
      hovertemplate: '%{y}, layer %{x}: %{z:.2f}<extra></extra>',
    };
    const lay = Object.assign({}, LAYOUT, {
      margin: { l: 110, r: 24, t: 12, b: 50 },
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'layer', dtick: 2 }),
      yaxis: { gridcolor: '#dde3ee', automargin: true },
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
    setReading('mech-answer', m.answer);
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
    setReading('allpaths-reading', ap.reading);
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
  // Distinct tab labels — several capabilities are "Neural encoding"; show the
  // specific target (IT cortex / language / video) so the buttons aren't identical.
  function scTabLabel(cap) {
    const parts = cap.split('—').map(s => s.trim());
    if (parts.length < 2) return parts[0];
    const spec = parts[1].split('(')[0].split(',')[0].trim();
    return parts[0].startsWith('Neural') ? 'Neural · ' + spec : parts[0];
  }
  scKeys.forEach((k, i) => {
    const b = document.createElement('button');
    b.className = 'tog' + (i === 0 ? ' active' : '');
    b.textContent = scTabLabel(D.scaling[k].capability);
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
    setReading('scaling-reading', s.reading);
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
    setReading('game-reading', g.reading);
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
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'reading accuracy (ROAR test)', range: [0.45, 1.05] }),
      // autorange so the points fill the width (the explicit range was squeezing
      // them into the left third); de-jargoned axis title.
      xaxis: Object.assign({}, LAYOUT.xaxis, {
        title: "word-recognition units switched off (% of the layer's filters)",
        type: 'linear', nticks: 7, autorange: true }),
      // threshold is a horizontal reference — span the FULL width via paper coords
      // so it never depends on the data range.
      shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: a.threshold, y1: a.threshold,
        line: { color: '#e0a13b', width: 1.5, dash: 'dash' } }],
      annotations: [{ xref: 'paper', x: 0.98, y: a.threshold, yanchor: 'bottom', xanchor: 'right',
        text: 'dyslexia threshold (0.65)', showarrow: false, font: { color: '#e0a13b', size: 10 } }],
    });
    Plotly.react('ablation-plot', [vwf, rnd], lay, CFG);
    if (a.protocol && document.getElementById('ablation-protocol'))
      $('ablation-protocol').textContent = a.protocol;
    setReading('ablation-reading', a.reading);
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
    setReading('selection-reading', s.reading);
  })();

  // ---- temporal-shift validation (real BOLD) ----
  if (D.temporal_shift_validation) {
    const tv = D.temporal_shift_validation;
    $('shift-title').textContent = tv.title;
    $('shift-sub').textContent = tv.subtitle;
    setReading('shift-reading', tv.reading);
    const curve = {
      x: tv.shifts, y: tv.scores, type: 'scatter', mode: 'lines+markers',
      line: { color: '#3bb273', width: 3 }, marker: { size: 9, color: '#3bb273' },
      hovertemplate: 'delay %{x} TRs: r=%{y:.3f}<extra></extra>',
    };
    const lay = Object.assign({}, LAYOUT, {
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'brain-prediction accuracy (median r)', rangemode: 'tozero' }),
      // linear axis so the delays sit at their true spacing and span the full width
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'time-shift applied to the model features (brain scans; 1 scan ≈ 1.5 s)',
        type: 'linear', nticks: 9 }),
      shapes: [
        // shuffle floor: horizontal reference spanning the FULL width (paper coords)
        { type: 'line', xref: 'paper', x0: 0, x1: 1, y0: tv.shuffle_floor, y1: tv.shuffle_floor,
          line: { color: '#d8483b', width: 1.5, dash: 'dash' } },
        // true delay: vertical line at the real brain lag (lands on the peak)
        { type: 'line', x0: tv.true_delay, x1: tv.true_delay, y0: 0, y1: Math.max(...tv.scores),
          line: { color: '#5b8cff', width: 1.5, dash: 'dot' } },
      ],
      annotations: [
        { x: tv.true_delay, y: Math.max(...tv.scores), yanchor: 'bottom', text: 'true brain lag (+3)',
          showarrow: false, font: { color: '#5b8cff', size: 11 } },
        { xref: 'paper', x: 0.98, y: tv.shuffle_floor, yanchor: 'bottom', xanchor: 'right',
          text: 'shuffle floor', showarrow: false, font: { color: '#d8483b', size: 11 } },
      ],
    });
    Plotly.react('shift-plot', [curve], lay, CFG);
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

  // ---- Witness: watch any benchmark run ----
  if (D.witness) {
    const w = D.witness;
    $('witness-title').textContent = w.title;
    $('witness-sub').textContent = w.subtitle;
    setReading('witness-reading', w.reading);
    $('witness-modes').innerHTML = w.modes.map(m => `<span class="wmode">${m}</span>`).join('');
    $('witness-grid').innerHTML = w.panels.map(p =>
      `<figure class="witness-card"><img src="${p.img}?v=1" alt="witness panel" />` +
      `<figcaption>${p.caption}</figcaption></figure>`).join('');
  }

  // ---- PerceptWindow: what the model actually saw (tabbed) ----
  if (D.percept && D.percept.tabs) {
    const p = D.percept;
    $('percept-title').textContent = p.title;
    $('percept-sub').textContent = p.subtitle;
    const tabsEl = $('percept-tabs');
    const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    function imgPanel(src, cap, cls) {
      return `<figure class="${cls || ''}"><img src="${src}?v=2" alt="${cap}"/><figcaption>${cap}</figcaption></figure>`;
    }
    function textPanel(txt, cap, cls) {
      return `<div class="percept-textcard mono ${cls || ''}"><div class="pt">${esc(txt)}</div>` +
        `<figcaption>${cap}</figcaption></div>`;
    }
    function drawPercept(tab, btn) {
      document.querySelectorAll('#percept-tabs .tog').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');
      const c = tab.columns;
      $('percept-grid').innerHTML = tab.rows.map(r => {
        const panels = r.kind === 'text'
          ? textPanel(r.presented, c[0]) + `<span class="percept-arrow">→</span>` +
            textPanel(r.tensor, c[1]) + `<span class="percept-arrow">→</span>` +
            textPanel(r.percept, c[2], 'percept-final')
          : imgPanel(r.presented, c[0]) + `<span class="percept-arrow">→</span>` +
            imgPanel(r.tensor, c[1]) + `<span class="percept-arrow">→</span>` +
            imgPanel(r.percept, c[2], 'percept-final');
        return `<div class="percept-row"><div class="percept-rowlabel">${r.label}</div>` +
          `<div class="percept-trip">${panels}</div>` +
          `<p class="percept-note">${r.note}</p></div>`;
      }).join('');
      setReading('percept-reading', tab.reading);
      $('percept-caveat').innerHTML = tab.caveat
        ? '<h3 class="caveat-h">What this does NOT show</h3>' + `<div class="raj-caveat">${tab.caveat}</div>`
        : '';
    }
    const want = (location.hash.match(/ptab=([\w-]+)/) || [])[1];
    // Display order: Rajalingham 2-AFC first, then multimodal, then resize & crop.
    const tabOrder = ['rajalingham', 'multimodal', 'crop'];
    const tabs = p.tabs.slice().sort(
      (a, b) => tabOrder.indexOf(a.id) - tabOrder.indexOf(b.id));
    tabs.forEach((tab, i) => {
      const b = document.createElement('button');
      b.className = 'tog';
      b.textContent = tab.label;
      b.onclick = () => drawPercept(tab, b);
      tabsEl.appendChild(b);
      const isDefault = want ? tab.id === want : i === 0;
      if (isDefault) drawPercept(tab, b);
    });
  }

  // ---- Rajalingham 2-AFC: same behavior, several ways ----
  if (D.rajalingham) {
    const raj = D.rajalingham;
    $('raj-title').textContent = raj.title;
    $('raj-sub').textContent = raj.subtitle;
    setReading('raj-reading', raj.reading);
    $('raj-montage-imgs').innerHTML = raj.montages.map(m =>
      `<img src="${m}?v=1" alt="2-AFC montage" />`).join('');
    // Group by model family (modes kept adjacent), families ordered by scale with
    // the null floor at the bottom — so within-model elicitation effects (direct vs
    // CoT vs few-shot) AND the cross-scale trend are both readable. Colour still
    // encodes the paradigm (kind), so 'all direct bars' etc. remain scannable.
    const famOrder = ['random null', 'CLIP', 'Qwen-VL-3B', 'Qwen-VL-7B', 'Gemma-4-12B'];
    const modeRank = { null: 0, feature: 0, direct: 1, cot: 2, fewshot: 3 };
    const fam = m => { const i = famOrder.indexOf(m); return i < 0 ? 99 : i; };
    const rr = raj.rows.slice().sort((a, b) =>
      fam(a.model) - fam(b.model) || (modeRank[a.kind] ?? 9) - (modeRank[b.kind] ?? 9));
    const labels = rr.map(r => `${r.model} · ${r.mode}`);
    const bar = {
      type: 'bar', orientation: 'h', y: labels, x: rr.map(r => r.i2n),
      marker: { color: rr.map(r => raj.kindColors[r.kind] || '#888') },
      hovertemplate: '%{y}: i2n %{x:.3f}<extra></extra>',
    };
    const lay = Object.assign({}, LAYOUT, {
      height: 430, margin: { l: 195, r: 28, t: 16, b: 44 }, showlegend: false,
      yaxis: Object.assign({}, LAYOUT.yaxis, { automargin: true }),
      // autorange to fit the bars (max ~0.16) so they fill the width; ~8 clean
      // auto-ticks. The 0.33 binary-chooser ceiling lives in the caption instead
      // of a plot line — drawing it forced the axis out to 0.33 and squeezed the
      // bars into the left third.
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'i2n (raw, vs human pool)',
        type: 'linear', nticks: 8, tickformat: '.2f', zeroline: true, zerolinecolor: '#c2cadb' }),
    });
    Plotly.react('raj-plot', [bar], lay, CFG);
    $('raj-table').innerHTML =
      '<tr><th>model</th><th>mode</th><th>acc</th><th>frac-L</th><th>i2n</th></tr>' +
      raj.rows.map(r => `<tr><td>${r.model}</td>`
        + `<td><span class="path-chip" style="background:${raj.kindColors[r.kind] || '#888'}">${r.mode}</span></td>`
        + `<td>${r.acc.toFixed(3)}</td><td>${r.frac_left == null ? '—' : r.frac_left.toFixed(2)}</td>`
        + `<td><b>${r.i2n.toFixed(3)}</b></td></tr>`).join('');
    $('raj-findings').innerHTML = raj.findings.map(f => `<div class="raj-finding">${f}</div>`).join('');
    if (D.gemma_scorecard) {
      const g = D.gemma_scorecard;
      $('gsc-title').textContent = g.title;
      $('gsc-sub').textContent = g.subtitle;
      setReading('gsc-reading', g.reading);
      const badge = s => s === 'done'
        ? '<span style="color:#1f9d57">✓ done</span>'
        : '<span style="color:#c6810f">⏳ running</span>';
      $('gsc-table').innerHTML =
        '<tr><th>capability</th><th>benchmark</th><th>metric</th><th>score</th><th>status</th><th>note</th></tr>'
        + g.rows.map(r => `<tr><td>${r.capability}</td><td>${r.benchmark}</td><td>${r.metric}</td>`
          + `<td><b>${r.score}</b></td><td>${badge(r.status)}</td>`
          + `<td style="font-size:.85em;color:var(--muted)">${r.note}</td></tr>`).join('');
    }
    if (raj.caveats) {
      $('raj-caveats').innerHTML = '<h3 class="caveat-h">What this does NOT yet establish</h3>' +
        raj.caveats.map(c => `<div class="raj-caveat">${c}</div>`).join('');
    }

    // ---- sequential vs simultaneous (Witness-style trace + score bars) ----
    if (raj.sequential) {
      const sq = raj.sequential;
      $('raj-seq-title').textContent = sq.title;
      $('raj-seq-sub').textContent = sq.subtitle;
      setReading('raj-seq-reading', sq.reading);
      $('raj-seq-trace').innerHTML = sq.trace.map((s, i) => {
        const card = s.kind === 'text'
          ? `<div class="seq-card seq-textcard"><div class="pt">${s.text}</div><figcaption>${s.cap}</figcaption></div>`
          : `<figure class="seq-card"><img src="${s.img}?v=1" alt="${s.cap}"/><figcaption>${s.cap}</figcaption></figure>`;
        return (i ? '<span class="percept-arrow">→</span>' : '') + card;
      }).join('');
      // Declared semantic order (null floor → by sample-availability: describe,
      // then recall ≈ simultaneous), NOT value-sorted — groups the two
      // "sample-available" conditions together so `describe` visibly stands apart.
      const sc = sq.conditions.slice();
      const bar = {
        type: 'bar', orientation: 'h', y: sc.map(c => c.label), x: sc.map(c => c.i2n),
        marker: { color: sc.map(c => sq.kindColors[c.kind] || '#888') },
        hovertemplate: '%{y}: i2n %{x:.3f}<extra></extra>',
      };
      const lay = Object.assign({}, LAYOUT, {
        height: 300, margin: { l: 230, r: 28, t: 14, b: 40 }, showlegend: false,
        yaxis: Object.assign({}, LAYOUT.yaxis, { automargin: true }),
        xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'i2n (raw)',
          type: 'linear', nticks: 7, tickformat: '.2f', zeroline: true, zerolinecolor: '#c2cadb' }),
      });
      Plotly.react('raj-seq-plot', [bar], lay, CFG);
      if (sq.caveat) {
        $('raj-seq-caveat').innerHTML = '<div class="raj-caveat">' + sq.caveat + '</div>';
      }
    }
  }

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
