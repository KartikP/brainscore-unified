(function () {
  const D = window.BSU_DATA;
  const $ = (id) => document.getElementById(id);

  // Reading markup (HTML-escaped first, so reading strings stay safe to inject):
  //   **text**  → accented <b>   (use sparingly)
  //   `code`    → <code>          (identifiers, method names)
  //   blank line → paragraph break
  const _esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const mark = s => _esc(s)
    .replace(/\*\*(.+?)\*\*/g, '<b class="hl">$1</b>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n\n/g, '<br><br>');
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
  // run through a different process() variant; that contrast IS the point.
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
    // the whole block changes, so those swipe as one.
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
    ['Predicting brain activity', 'Turn the model’s internal features into a prediction of real brain activity (in vision areas, the language network, or across the whole brain) and check it against actual recordings.'],
    ['Matching human behavior', 'Have the model make the same kind of choice a person would (e.g. “real word or fake?”) and score how closely its pattern of right and wrong answers matches theirs.'],
    ['Switching neurons off', 'Turn off a chosen group of the model’s neurons, watch what breaks, then switch them back on exactly. This is how we give a model “dyslexia” further down.'],
    ['Acting in a world', 'Close the loop: the model sees a scene, makes a move, the world responds, repeat (a built-in grid-game test).'],
    ['Models talking to models', 'One model’s reply becomes the next model’s input, with no special glue code, so you can study models interacting.'],
    ['Lining the senses up in time', 'Stitch features from picture, sound, and words onto the brain’s own clock, so they can be compared moment by moment.'],
    ['Matching layers to brain areas', 'Find which of the model’s internal layers best matches each brain area: one layer, the whole brain, or a hand-picked mix across layers.'],
    ['Brain-like layout', 'Score whether the model’s neurons are arranged in space like the brain’s, not just whether they compute the same thing.'],
    ['Cortical visualizations', 'Project any of these scores onto an inflated cortical surface, as they appear in published papers.'],
  ];
  $('cap-grid').innerHTML = CAPS.map(([t, d]) =>
    `<div class="cap-card"><div class="tag">capability</div><h3>${t}</h3>
     <p>${d}</p></div>`).join('');

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

  // ---- layer contribution per modality (per-layer ridge brain-prediction r) ----
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
      type: 'heatmap', colorscale: 'Turbo', colorbar: { title: 'brain-<br>match', thickness: 12 },
      hovertemplate: '%{y}, layer %{x}: %{z:.2f}<extra></extra>',
    };
    const lay = Object.assign({}, LAYOUT, {
      margin: { l: 110, r: 24, t: 12, b: 50 },
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'layer (input → output)', dtick: 2 }),
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
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'real-vs-fake-word accuracy', range: [0.45, 0.75] }),
    });
    Plotly.react('mech-floors-plot', [bar], lay, CFG);
  }

  // ---- all models, stratified by input type x output path ----
  if (D.all_paths) {
    const ap = D.all_paths;
    $('allpaths-title').textContent = ap.title;
    $('allpaths-sub').textContent = ap.subtitle;
    setReading('allpaths-reading', ap.reading);
    // friendly display names for the three ways a model can answer
    const pathName = { 'readout': 'read its features', 'generation': 'write an answer',
      'instr-readout': 'features after thinking' };
    const pn = p => pathName[p] || p;
    // one trace per path (colour = path), points at x=model, y=score
    const traces = Object.keys(ap.pathColors).map(path => {
      const rows = ap.rows.filter(r => r.path === path && r.score != null);
      return {
        x: rows.map(r => r.model), y: rows.map(r => r.score),
        text: rows.map(r => r.input), type: 'scatter', mode: 'markers', name: pn(path),
        marker: { color: ap.pathColors[path], size: 15, line: { color: '#0a0e17', width: 1 } },
        hovertemplate: '%{x} · %{text} · ' + pn(path) + ': %{y:.2f}<extra></extra>',
      };
    });
    const lay = Object.assign({}, LAYOUT, {
      showlegend: true,
      legend: { orientation: 'h', x: 0, y: 1.12, font: { size: 10 }, bgcolor: 'rgba(0,0,0,0)' },
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'real-vs-fake-word accuracy', range: [0.42, 1.08] }),
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
      + `<td><span class="path-chip" style="background:${ap.pathColors[r.path] || '#666'}">${pn(r.path)}</span></td>`
      + `<td>${r.score == null ? '—' : r.score.toFixed(3)}</td></tr>`;
    $('allpaths-table').innerHTML =
      '<tr><th>model</th><th>shown</th><th>how it answered</th><th>score</th></tr>'
      + ap.rows.map(fmt).join('');
  }

  // ---- scaling curves ----
  const scKeys = Object.keys(D.scaling);
  const scTabs = $('scaling-tabs');
  // Distinct tab labels. Several capabilities are "Neural encoding"; show the
  // specific target (IT cortex / language / video) so the buttons aren't identical.
  function scTabLabel(cap) {
    // Capability labels read "Main: specific" (or, historically, "Main — specific").
    // Split on either delimiter and keep the specific half so the tabs aren't identical.
    const parts = cap.split(/\s*[:—]\s*/).map(s => s.trim());
    if (parts.length < 2) return parts[0];
    const spec = parts[1].split('(')[0].split(',')[0].trim();
    return spec.charAt(0).toUpperCase() + spec.slice(1);
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
        yanchor: 'bottom', text: 'null baseline', showarrow: false,
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
      hovertemplate: '%{x}: %{y:.0%} of games won<extra></extra>',
    };
    const lay = Object.assign({}, LAYOUT, {
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'games won', range: [0, 1.08] }),
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
      name: 'switch off word-selective units', line: { color: '#d8483b', width: 3 },
      marker: { size: 8 },
      error_y: { type: 'data', array: a.vwf_roar_sd, color: '#d8483b', thickness: 1 },
      hovertemplate: 'word-selective units, %{x}%: accuracy %{y:.2f}<extra></extra>',
    };
    const rnd = {
      x: a.mask_pct, y: a.random_roar, type: 'scatter', mode: 'lines+markers',
      name: 'switch off random neurons', line: { color: '#9aa0a6', width: 3, dash: 'dot' },
      marker: { size: 8 },
      error_y: { type: 'data', array: a.random_roar_sd, color: '#9aa0a6', thickness: 1 },
      hovertemplate: 'random neurons, %{x}%: accuracy %{y:.2f}<extra></extra>',
    };
    const lay = Object.assign({}, LAYOUT, {
      showlegend: true,
      legend: { x: 0.02, y: 0.12, font: { size: 10 }, bgcolor: 'rgba(0,0,0,0)' },
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'reading accuracy', range: [0.45, 1.05] }),
      // autorange so the points fill the width (the explicit range was squeezing
      // them into the left third); de-jargoned axis title.
      xaxis: Object.assign({}, LAYOUT.xaxis, {
        title: "word-selective units switched off (% of the layer)",
        type: 'linear', nticks: 7, autorange: true }),
      // threshold is a horizontal reference: span the FULL width via paper coords
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

  // ---- tools section (auto-setup + layer-mapping schematics) ----
  if (D.tools) {
    const t = D.tools;
    $('tools-title').textContent = t.title;
    $('tools-sub').textContent = t.subtitle;
    if (t.autoreg) {
      $('tools-ar-tag').textContent = t.autoreg.tag;
      $('tools-ar-name').textContent = t.autoreg.name;
      $('tools-ar-cap').textContent = t.autoreg.caption;
    }
    if (t.layermap) {
      $('tools-lm-tag').textContent = t.layermap.tag;
      $('tools-lm-name').textContent = t.layermap.name;
      $('tools-lm-cap').textContent = t.layermap.caption;
    }
  }

  // ---- layer-mapping approaches (V-JEPA2 sweep + 4-approach comparison) ----
  if (D.layer_mapping) {
    const lm = D.layer_mapping;
    $('lm-title').textContent = lm.title;
    $('lm-sub').textContent = lm.subtitle;
    $('lm-appr-title').textContent = lm.approaches_title || 'Four ways to map the region';
    setReading('lm-reading', lm.reading);
    if (lm.heatmap_img) $('lm-heatmap-img').src = lm.heatmap_img + '?v=' + (lm.cachebust || '1');
    if (lm.heatmap_caption) $('lm-heatmap-cap').innerHTML = lm.heatmap_caption;

    // (1) per-layer sweep curve, with current vs best layer marked
    const xs = lm.per_layer_r.map((_, i) => i);
    const curve = {
      x: xs, y: lm.per_layer_r, type: 'scatter', mode: 'lines+markers',
      line: { color: '#6a3d9a', width: 3 }, marker: { size: 7, color: '#6a3d9a' },
      hovertemplate: 'layer %{x}: r=%{y:.3f}<extra></extra>',
    };
    const best = { x: [lm.best_layer], y: [lm.per_layer_r[lm.best_layer]], type: 'scatter',
      mode: 'markers', marker: { size: 16, color: '#1f9d57', symbol: 'star',
      line: { color: '#0a0e17', width: 1 } }, hovertemplate: 'best: layer %{x} r=%{y:.3f}<extra></extra>' };
    const cur = { x: [lm.current_layer], y: [lm.per_layer_r[lm.current_layer]], type: 'scatter',
      mode: 'markers', marker: { size: 13, color: '#d8483b', symbol: 'circle-open',
      line: { width: 3 } }, hovertemplate: 'previous default: layer %{x} r=%{y:.3f}<extra></extra>' };
    const sweepLay = Object.assign({}, LAYOUT, {
      margin: { l: 52, r: 16, t: 28, b: 46 },
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'Brain-Score', rangemode: 'tozero' }),
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'layer (input → output)', dtick: 4 }),
      annotations: [
        { x: lm.best_layer, y: lm.per_layer_r[lm.best_layer], yanchor: 'bottom', xanchor: 'center',
          text: 'best (layer ' + lm.best_layer + ')', showarrow: false, yshift: 12,
          font: { color: '#1f9d57', size: 11 } },
        { x: lm.current_layer, y: lm.per_layer_r[lm.current_layer], yanchor: 'top', xanchor: 'center',
          text: 'the layer the test uses (16)', showarrow: false, yshift: -10,
          font: { color: '#d8483b', size: 11 } }],
    });
    Plotly.react('lm-sweep-plot', [curve, best, cur], sweepLay, CFG);

    // (2) the value of each CHOICE = the gap it opens over a same-size random
    // pick. This is the ONLY fair comparison: absolute scores across strategies
    // are confounded by neuron count (fixed-strength readout flatters smaller
    // sets), so we plot the gap, not the level.
    if ($('lm-metric-note')) $('lm-metric-note').textContent = lm.metric_note || '';

    // (2) budget-matched curve: score vs # neurons, RidgeCV (per-voxel alpha) +
    // ceiling-normalized, so heights are comparable. Two selection strategies
    // (within one layer / pooled across top layers), each shadowed by its
    // same-size random null, plus a horizontal reference for the whole layer.
    const bc = lm.budget_curve;
    if (bc) {
      const K = bc.budgets;
      const mk = (y, name, color, dash, width) => ({
        x: K, y, name, type: 'scatter', mode: 'lines+markers',
        line: { color, width: width || 2.5, dash: dash || 'solid' },
        marker: { size: dash ? 0 : 6, color },
        hovertemplate: name + ': %{y:.3f} at %{x} neurons<extra></extra>',
      });
      const refLine = (val, name, color) => ({
        x: [K[0], K[K.length - 1]], y: [val, val], name, type: 'scatter', mode: 'lines',
        line: { color, width: 2, dash: 'dot' },
        hovertemplate: name + ': %{y:.3f}<extra></extra>',
      });
      const traces = [
        refLine(bc.whole_layer, 'whole best layer (standard), best', '#1f9d57'),
        refLine(bc.several_layers, 'several whole layers', '#c6810f'),
        mk(bc.within, 'best neurons, one layer', '#2f6bff'),
        mk(bc.pooled, 'best neurons, pooled across layers', '#7c4dff'),
        mk(bc.within_random, 'picked at random', '#9aa0a6', 'dash'),
      ];
      const bLay = Object.assign({}, LAYOUT, {
        margin: { l: 54, r: 14, t: 10, b: 70 },
        legend: { orientation: 'h', x: 0, y: -0.32, font: { size: 10.5 },
          bgcolor: 'rgba(0,0,0,0)' },
        showlegend: true,
        xaxis: Object.assign({}, LAYOUT.xaxis, { title: bc.x_title, type: 'log',
          tickvals: K, ticktext: K.map(String), tickfont: { size: 10 } }),
        yaxis: Object.assign({}, LAYOUT.yaxis, { title: bc.y_title, rangemode: 'tozero' }),
        annotations: bc.crossover_k ? [{
          x: Math.log10(bc.crossover_k), y: bc.within[bc.budgets.indexOf(bc.crossover_k)],
          text: 'by ~' + bc.crossover_k + ' neurons,<br>best ≈ random',
          showarrow: true, arrowhead: 0, ax: -10, ay: -38,
          font: { size: 10, color: 'var(--muted)' }, align: 'center' }] : [],
      });
      Plotly.react('lm-budget-plot', traces, bLay, CFG);
    }

    // ranked four-strategy table, now an apples-to-apples comparison (RidgeCV +
    // ceiling), so heights ARE directly comparable.
    if ($('lm-table-title')) $('lm-table-title').textContent = lm.table_title || '';
    const st = lm.strategies_table || [];
    const t = $('lm-appr-table');
    if (t) t.innerHTML = '<tr><th>way of reading the model</th><th># neurons</th>'
      + '<th>Brain-Score</th></tr>' +
      st.map(s => `<tr><td>${s.name}${s.best ? ' &nbsp;<b style="color:var(--good)">← best</b>' : ''}</td>`
        + `<td>${s.features.toLocaleString()}</td>`
        + `<td${s.best ? ' style="color:var(--good);font-weight:600"' : ''}>${s.score.toFixed(3)}</td></tr>`).join('');
    if ($('lm-table-caveat')) $('lm-table-caveat').innerHTML =
      'All four now use the properly-tuned readout and the same explainable-signal '
      + 'scale, so these numbers are directly comparable: no hidden penalty for using more neurons.';

    // (3) deep-dive: the stress-tests that explain WHY one layer is enough
    const dd = lm.deep_dive;
    if (dd) {
      if ($('dd-title')) $('dd-title').textContent = dd.title;
      if ($('dd-lead')) $('dd-lead').textContent = dd.lead;
      if ($('dd-cards')) $('dd-cards').innerHTML = (dd.cards || []).map(c =>
        `<div class="dd-card"><div class="dd-stat">${c.stat}</div>`
        + `<h4>${c.title}</h4><p>${c.body}</p></div>`).join('');
      const pc = dd.pc_chart;
      if (pc && $('dd-pc-plot')) {
        if ($('dd-pc-title')) $('dd-pc-title').textContent = pc.title;
        if ($('dd-pc-cap')) $('dd-pc-cap').textContent = pc.caption;
        const bar = {
          x: pc.labels, y: pc.r, type: 'bar',
          marker: { color: pc.r.map(v => v >= 0.5 ? '#1f9d57' : (v >= 0.2 ? '#7c4dff' : '#9aa0a6')) },
          hovertemplate: 'pattern %{x}: %{y:.2f}<extra></extra>',
        };
        const lay = Object.assign({}, LAYOUT, {
          margin: { l: 50, r: 12, t: 8, b: 48 },
          xaxis: Object.assign({}, LAYOUT.xaxis, { title: pc.x_title, dtick: 2,
            tickfont: { size: 10 } }),
          yaxis: Object.assign({}, LAYOUT.yaxis, { title: pc.y_title, range: [0, 1] }),
        });
        Plotly.react('dd-pc-plot', [bar], lay, CFG);
      }
    }
  }

  // ---- does it generalize? cross-benchmark synthesis ----
  if (D.generalize) {
    const g = D.generalize;
    $('gen-title').textContent = g.title;
    $('gen-sub').textContent = g.subtitle;
    $('gen-dim-title').textContent = g.dim_title;
    $('gen-dim-note').textContent = g.dim_note;
    $('gen-hier-title').textContent = g.hier_title;
    $('gen-hier-note').textContent = g.hier_note;
    setReading('gen-reading', g.reading);

    // (1) effective-dim across benchmarks (sorted by #stimuli), stays flat/low
    const db = g.dim_bars;
    const dimBar = {
      x: db.map(b => b.label), y: db.map(b => b.dim), type: 'bar',
      marker: { color: '#7c4dff' },
      text: db.map(b => b.dim.toFixed(0)), textposition: 'outside', textfont: { size: 11 },
      customdata: db.map(b => b.n),
      hovertemplate: '%{x}: ~%{y:.0f} patterns<br>%{customdata} stimuli<extra></extra>',
    };
    const dimLay = Object.assign({}, LAYOUT, {
      margin: { l: 46, r: 12, t: 10, b: 86 },
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'effective dimensions', range: [0, 34] }),
      xaxis: Object.assign({}, LAYOUT.xaxis, { tickangle: -28, tickfont: { size: 9.5 },
        title: 'brain dataset (← fewer / more stimuli →)' }),
    });
    Plotly.react('gen-dim-plot', [dimBar], dimLay, CFG);

    // (2) brain-area hierarchy vs model-layer depth
    const lines = g.hier_lines.map(L => ({
      x: g.hier_rois, y: L.depth, name: L.name, type: 'scatter', mode: 'lines+markers',
      line: { color: L.color, width: 3 }, marker: { size: 8, color: L.color },
      hovertemplate: L.name + ' · %{x}: depth %{y:.2f}<extra></extra>',
    }));
    const hierLay = Object.assign({}, LAYOUT, {
      margin: { l: 52, r: 12, t: 10, b: 70 }, showlegend: true,
      legend: { orientation: 'h', x: 0, y: -0.22, font: { size: 10 }, bgcolor: 'rgba(0,0,0,0)' },
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'model layer depth (0→1)',
        range: [-0.05, 1.05] }),
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'brain area (early → late)' }),
    });
    Plotly.react('gen-hier-plot', lines, hierLay, CFG);
  }

  // ---- public leaderboard (Algonauts 2025 submission) ----
  if (D.leaderboard) {
    const lb = D.leaderboard;
    $('lb-title').textContent = lb.title;
    $('lb-sub').textContent = lb.subtitle;
    $('lb-indist-title').textContent = lb.indist_title;
    $('lb-indist-note').textContent = lb.indist_note;
    $('lb-ood-title').textContent = lb.ood_title;
    $('lb-ood-note').textContent = lb.ood_note;
    setReading('lb-reading', lb.reading);
    if (lb.caption) { const c = $('lb-caption'); if (c) c.textContent = lb.caption; }

    // (1) in-distribution: ours vs baseline vs best published
    const ib = lb.indist_bars;
    const inBar = {
      x: ib.map(b => b.label), y: ib.map(b => b.v), type: 'bar',
      marker: { color: ib.map(b => b.ours ? '#7c4dff' : '#b9c2d6') },
      text: ib.map(b => b.v.toFixed(2)), textposition: 'outside', textfont: { size: 12 },
      hovertemplate: '%{x}: %{y:.3f}<extra></extra>',
    };
    const inLay = Object.assign({}, LAYOUT, {
      margin: { l: 48, r: 12, t: 10, b: 52 },
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'brain-prediction score', range: [0, 0.36] }),
      xaxis: Object.assign({}, LAYOUT.xaxis, { tickfont: { size: 11 } }),
    });
    Plotly.react('lb-indist-plot', [inBar], inLay, CFG);

    // (2) out-of-distribution: per-film bars + our-average reference line
    const ob = lb.ood_bars;
    const oodBar = {
      x: ob.map(b => b.label), y: ob.map(b => b.v), type: 'bar',
      marker: { color: '#5b8def' },
      text: ob.map(b => b.v.toFixed(2)), textposition: 'outside', textfont: { size: 10 },
      hovertemplate: '%{x}: %{y:.3f}<extra></extra>',
    };
    const oodLay = Object.assign({}, LAYOUT, {
      margin: { l: 48, r: 12, t: 10, b: 100 },
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'brain-prediction score', range: [0, 0.26] }),
      xaxis: Object.assign({}, LAYOUT.xaxis, { tickangle: -30, tickfont: { size: 9.5 } }),
      shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: lb.ood_avg, y1: lb.ood_avg,
        line: { color: '#7c4dff', width: 1.5, dash: 'dash' } }],
      annotations: [{ xref: 'paper', x: 1, y: lb.ood_avg, xanchor: 'right',
        text: 'our avg ' + lb.ood_avg.toFixed(2), showarrow: false, yshift: 9,
        font: { size: 10, color: '#7c4dff' } }],
    });
    Plotly.react('lb-ood-plot', [oodBar], oodLay, CFG);
  }

  // ---- native vs post-hoc fusion (MIRAGE vs TRIBEv2) ----
  if (D.fusion_compare) {
    const fc = D.fusion_compare;
    $('fusion-title').textContent = fc.title;
    $('fusion-sub').textContent = fc.subtitle;
    $('fusion-appr-title').textContent = fc.appr_title || 'Encoding accuracy by feature source';
    setReading('fusion-reading', fc.reading);
    if (fc.caveat) $('fusion-caveat').innerHTML = fc.caveat;
    const kindColor = { native: '#d8483b', posthoc: '#3b7dd8', modality: '#9aa0a6' };
    const bar = {
      x: fc.bars.map(b => b.name), y: fc.bars.map(b => b.r), type: 'bar',
      marker: { color: fc.bars.map(b => kindColor[b.kind] || '#9aa0a6') },
      text: fc.bars.map(b => b.r.toFixed(3)), textposition: 'outside', textfont: { size: 11 },
      hovertemplate: '%{x}: r=%{y:.3f}<extra></extra>',
    };
    const ys = fc.bars.map(b => b.r);
    const lay = Object.assign({}, LAYOUT, {
      margin: { l: 48, r: 12, t: 16, b: 84 },
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'Brain-Score',
        range: [Math.max(0, Math.min.apply(null, ys) - 0.04), Math.max.apply(null, ys) + 0.04] }),
      xaxis: Object.assign({}, LAYOUT.xaxis, { tickangle: -18, tickfont: { size: 10 } }),
    });
    Plotly.react('fusion-plot', [bar], lay, CFG);
    const tb = $('fusion-table');
    if (tb) tb.innerHTML = '<tr><th>where the features come from</th><th>how the senses combine</th><th># features</th><th>Brain-Score</th><th>time to run (min)</th></tr>' +
      fc.bars.map(b => `<tr><td>${b.name}</td>`
        + `<td><span class="path-chip" style="background:${kindColor[b.kind] || '#9aa0a6'}">${b.fusion || b.kind}</span></td>`
        + `<td>${b.dim != null ? b.dim.toLocaleString() : '—'}</td><td>${b.r.toFixed(3)}</td>`
        + `<td class="muted">${b.extract_min != null ? b.extract_min.toFixed(1) : '—'}</td></tr>`).join('');

    // within-Qwen fusion curve: per-layer r, layer 0 = fusion OFF, peak = fusion ON
    const wq = fc.within_qwen;
    if (wq && document.getElementById('fusion-curve-plot')) {
      if (fc.curve_caption) $('fusion-curve-cap').innerHTML = fc.curve_caption;
      const curve = {
        x: wq.layers, y: wq.r, type: 'scatter', mode: 'lines+markers',
        line: { color: '#d8483b', width: 3 }, marker: { size: 6, color: '#d8483b' },
        hovertemplate: 'layer %{x}: r=%{y:.3f}<extra></extra>',
      };
      const off = { x: [0], y: [wq.r[wq.layers.indexOf(0)]], type: 'scatter', mode: 'markers',
        marker: { size: 13, color: '#9aa0a6', symbol: 'circle-open', line: { width: 3 } },
        hovertemplate: 'fusion OFF (layer 0): r=%{y:.3f}<extra></extra>' };
      const pk = { x: [wq.peak_layer], y: [wq.r[wq.layers.indexOf(wq.peak_layer)]], type: 'scatter',
        mode: 'markers', marker: { size: 15, color: '#1f9d57', symbol: 'star', line: { color: '#0a0e17', width: 1 } },
        hovertemplate: 'fusion ON (peak layer %{x}): r=%{y:.3f}<extra></extra>' };
      const lay = Object.assign({}, LAYOUT, {
        margin: { l: 48, r: 14, t: 26, b: 44 },
        yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'Brain-Score', rangemode: 'tozero' }),
        xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'depth inside the native multimodal model (0 = before the senses mix)', dtick: 8 }),
        annotations: [
          { x: 0, y: wq.r[wq.layers.indexOf(0)], yanchor: 'top', yshift: -8, text: 'fusion off', showarrow: false, font: { color: '#777', size: 10 } },
          { x: wq.peak_layer, y: wq.r[wq.layers.indexOf(wq.peak_layer)], yanchor: 'bottom', yshift: 10, text: 'fusion on (peak)', showarrow: false, font: { color: '#1f9d57', size: 10 } }],
      });
      Plotly.react('fusion-curve-plot', [curve, off, pk], lay, CFG);
    }
  }

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
      yaxis: Object.assign({}, LAYOUT.yaxis, { title: 'Brain-Score', rangemode: 'tozero' }),
      // linear axis so the delays sit at their true spacing and span the full width
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'how far we nudge the AI in time vs. the brain (scans; 1 scan ≈ 1.5 s)',
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
        { x: tv.true_delay, y: Math.max(...tv.scores), yanchor: 'bottom', text: "the brain's real delay (+3)",
          showarrow: false, font: { color: '#5b8cff', size: 11 } },
        { xref: 'paper', x: 0.98, y: tv.shuffle_floor, yanchor: 'bottom', xanchor: 'right',
          text: 'no-signal floor', showarrow: false, font: { color: '#d8483b', size: 11 } },
      ],
    });
    Plotly.react('shift-plot', [curve], lay, CFG);
  }

  // ---- real Algonauts clip -> model-predicted fMRI, evolving on the cortex ----
  if (D.movie_brain) {
    const mb = D.movie_brain;
    $('mb-title').textContent = mb.title;
    $('mb-sub').textContent = mb.subtitle;
    setReading('mb-reading', mb.reading);
    if (mb.note) $('mb-note').innerHTML = mb.note;
    if (mb.caption) $('mb-caption').innerHTML =
      '<h3 class="caveat-h">How to read this</h3><div class="raj-caveat">' + mb.caption + '</div>';
    const pad = i => String(i).padStart(3, '0');
    const bust = '?v=' + (mb.cachebust || '1');
    const rEl = $('mb-rnow');
    // Tabs: one encoder per tab (TRIBEv2 / Qwen). Each carries its own model
    // brain prefix + per-TR r + clip-mean. Backward-compatible with the single
    // mb.model/per_tr_r/mean_r shape.
    const tabs = mb.tabs || [{ id: 'm', label: '', model: mb.model, per_tr_r: mb.per_tr_r, mean_r: mb.mean_r }];
    let active = tabs[0];
    let curFrame = 0;
    function setFrame(i) {
      curFrame = i = Math.max(0, Math.min(mb.n - 1, i));
      $('mb-human').src = mb.human + pad(i) + '.png' + bust;
      $('mb-model').src = active.model + pad(i) + '.png' + bust;
      $('mb-transcript').innerHTML = mb.transcript.map((w, j) =>
        `<span class="${j === i ? 'mb-word-on' : ''}">${w}</span>`).join(' ');
      if (rEl && active.per_tr_r) {
        const r = active.per_tr_r[i];
        const pct = Math.max(0, Math.min(100, (r / 0.5) * 100));   // bar scaled to r∈[0,0.5]
        rEl.innerHTML =
          `<div class="mb-rnow-lab">how well the two brains match, right now ` +
          `<span class="mb-rnow-sub">(0 = chance, ~0.4 = today's ceiling)</span></div>` +
          `<div class="mb-rnow-bar"><span style="width:${pct}%"></span></div>` +
          `<div class="mb-rnow-val">match = ${r.toFixed(2)}` +
          (active.mean_r ? ` <span class="mb-rnow-sub">· whole-clip average ${active.mean_r.toFixed(2)}</span>` : '') +
          `</div>`;
      }
    }
    // render tab selector
    const tabsEl = $('mb-tabs');
    if (tabsEl && mb.tabs) {
      tabsEl.innerHTML = tabs.map((t, k) =>
        `<button class="mb-tab${k === 0 ? ' on' : ''}" data-k="${k}">${t.label}</button>`).join('');
      tabsEl.querySelectorAll('.mb-tab').forEach(btn => btn.addEventListener('click', () => {
        active = tabs[+btn.dataset.k];
        tabsEl.querySelectorAll('.mb-tab').forEach(b => b.classList.toggle('on', b === btn));
        setFrame(curFrame);
      }));
    }
    setFrame(0);
    // Both brain montages + transcript follow the video's playhead (1 frame per TR).
    const vid = $('mb-video');
    if (vid) {
      vid.src = mb.video + bust;
      const sync = () => setFrame(Math.floor(vid.currentTime / mb.tr_sec));
      vid.addEventListener('timeupdate', sync);
      vid.addEventListener('seeked', sync);
    }
  }

  // ---- limitations ----
  if (D.limitations) {
    $('lim-title').textContent = D.limitations.title;
    $('lim-list').innerHTML = D.limitations.items
      .map(t => `<li>${t}</li>`).join('');
  }

  // ---- nulls table ----
  const nt = $('nulls-table');
  nt.innerHTML = '<tr><th>kind of test</th><th>the "from nothing" version</th><th>the mistake it catches</th></tr>' +
    D.nulls.entries.map(e =>
      `<tr><td>${e.capability}</td><td>${e.null}</td><td>${e.what_it_catches}</td></tr>`).join('');

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
    // Group by model family (modes kept adjacent), families ordered by scale with
    // the null floor at the bottom, so within-model elicitation effects (direct vs
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
      // of a plot line. Drawing it forced the axis out to 0.33 and squeezed the
      // bars into the left third.
      xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'human–model consistency (i2n) (how human-like the pattern of mistakes is)',
        type: 'linear', nticks: 8, tickformat: '.2f', zeroline: true, zerolinecolor: '#c2cadb' }),
    });
    Plotly.react('raj-plot', [bar], lay, CFG);
    $('raj-table').innerHTML =
      '<tr><th>model</th><th>how asked</th><th>accuracy</th><th>% chose left</th><th>match-to-human</th></tr>' +
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
        '<tr><th>capability</th><th>test</th><th>what\'s measured</th><th>score</th><th>status</th><th>note</th></tr>'
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
      // then recall ≈ simultaneous), NOT value-sorted: groups the two
      // "sample-available" conditions together so `describe` visibly stands apart.
      const sc = sq.conditions.slice();
      const bar = {
        type: 'bar', orientation: 'h', y: sc.map(c => c.label), x: sc.map(c => c.i2n),
        marker: { color: sc.map(c => sq.kindColors[c.kind] || '#888') },
        hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
      };
      const lay = Object.assign({}, LAYOUT, {
        height: 300, margin: { l: 230, r: 28, t: 14, b: 40 }, showlegend: false,
        yaxis: Object.assign({}, LAYOUT.yaxis, { automargin: true }),
        xaxis: Object.assign({}, LAYOUT.xaxis, { title: 'human–model consistency (i2n)',
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
