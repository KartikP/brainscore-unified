/* Interactive architecture explorer for the Brain-Score Unified Model Interface.
 *
 * Mirrors the real dispatch in core/brainscore_core/model_interface.py::
 * BrainScoreModel.process() and the compatibility contract in compatibility.py.
 * Pick a model archetype + an input event (+ a behavioral task), and it resolves
 * the exact call flow, the OutputEvent type, and any pre-flight failure — honestly,
 * including the incompatible combinations.
 */
(function () {
  'use strict';

  // MODALITY_PRIORITY tiebreak (single-modality dispatch), verbatim from core.
  const MODALITY_PRIORITY = ['vision', 'text', 'audio', 'video'];

  // ---- model archetypes: the capability profile each registration exposes ----
  // has: which dispatch slots / wrappers are present.
  //   activations = has an activations_model (internal layers readable)
  //   generation  = has generation_fn (instruction-following text out)
  //   readout     = has behavioral_readout_layer (fit a logistic head on a layer)
  //   action      = has action_fn (embodied closed loop)
  //   state_change= has state_change_fn (lesion/perturb)
  const MODELS = [
    { id: 'vision', name: 'Vision model', eg: 'CLIP ViT-B/32 · ResNet-50 · random-ViT (null)',
      wrappers: ['PytorchWrapper'], preprocessors: ['vision'], available: ['vision'], required: [],
      has: { activations: true, generation: false, readout: false, action: false, state_change: false },
      blurb: 'A CNN/ViT wrapped by PytorchWrapper; forward hooks expose any layer.' },
    { id: 'language', name: 'Language model', eg: 'GPT-2',
      wrappers: ['TextWrapper'], preprocessors: ['text'], available: ['text'], required: ['text'],
      has: { activations: true, generation: false, readout: false, action: false, state_change: false },
      blurb: 'TextWrapper tokenizes + runs the causal LM; required={text} hard-gates non-text benchmarks.' },
    { id: 'vlm', name: 'Vision-language model (VLM)', eg: 'Qwen2.5-VL · BLIP-2',
      wrappers: ['VLMVisionWrapper', 'TextWrapper'], preprocessors: ['vision', 'text'], available: ['vision', 'text'], required: [],
      has: { activations: true, generation: true, readout: true, action: true, state_change: true },
      blurb: 'Two towers (flattened-patch vision + causal text). Often also instruction-following (generation_fn) and lesionable (state_change_fn).' },
    { id: 'video', name: 'Video model', eg: 'V-JEPA v1/v2 · VideoMAE',
      wrappers: ['VideoWrapper'], preprocessors: ['video'], available: ['video'], required: ['video'],
      has: { activations: true, generation: false, readout: false, action: false, state_change: false },
      blurb: 'VideoWrapper consumes (B,T,C,H,W); returns time-resolved features.' },
    { id: 'audio', name: 'Audio model', eg: 'Wav2Vec2 · HuBERT · Whisper',
      wrappers: ['AudioWrapper'], preprocessors: ['audio'], available: ['audio'], required: ['audio'],
      has: { activations: true, generation: false, readout: false, action: false, state_change: false },
      blurb: 'AudioWrapper resamples + runs an HF audio encoder; mean-time or time-series features.' },
    { id: 'av', name: 'Multimodal A+V model', eg: 'V-JEPA + Wav2Vec2',
      wrappers: ['VideoWrapper', 'AudioWrapper'], preprocessors: ['video', 'audio'], available: ['video', 'audio'], required: [],
      has: { activations: true, generation: false, readout: false, action: false, state_change: false },
      regionModalityMap: true,
      blurb: 'Two preprocessors + region_modality_map routes each region to its tower. Needs multi_modality=True to use both at once.' },
    { id: 'api', name: 'Closed-weight API model', eg: 'Claude · GPT-4 · DeepSeek · OpenRouter',
      wrappers: [], preprocessors: ['vision', 'text'], available: ['vision', 'text'], required: [],
      has: { activations: false, generation: true, readout: false, action: true, state_change: false },
      blurb: 'Output-only: generation_fn closure, activations_model=None. No internal activations → behavioral / embodied only.' },
    { id: 'embodied', name: 'Embodied policy', eg: 'game player · π0 robot policy',
      wrappers: [], preprocessors: [], available: [], required: [],
      has: { activations: false, generation: false, readout: false, action: true, state_change: false },
      blurb: 'action_fn(EnvironmentStep) -> EnvironmentResponse. Drives the closed loop; no perceptual extraction.' }
  ];

  // ---- input events (what you hand to process) ----
  const INPUTS = [
    { id: 'image', name: 'Images', event: 'StimulusSet', modality: 'vision', column: 'image_file_name', perceptual: true },
    { id: 'text', name: 'Sentences', event: 'StimulusSet', modality: 'text', column: 'sentence', perceptual: true },
    { id: 'audio', name: 'Audio clips', event: 'StimulusSet', modality: 'audio', column: 'audio_path', perceptual: true },
    { id: 'video', name: 'Video clips', event: 'StimulusSet', modality: 'video', column: 'video_path', perceptual: true },
    { id: 'movie', name: 'Movie (video+audio+text)', event: 'MultimodalStimulusSet', modality: ['video', 'audio', 'text'], column: 'video_path / audio_path / sentence', perceptual: true },
    { id: 'lesion', name: 'Lesion spec', event: 'StateChange', modality: null, perceptual: false },
    { id: 'gamestep', name: 'Game step', event: 'EnvironmentStep', modality: null, perceptual: false }
  ];

  // ---- behavioral task modifier (only meaningful for perceptual inputs) ----
  const TASKS = [
    { id: 'neural', name: 'Neural encoding', desc: 'start_recording(region) → predict brain responses' },
    { id: 'readout', name: 'Behavioral readout', desc: 'fit a logistic head on a layer (fitting_stimuli)' },
    { id: 'generation', name: 'Behavioral generation', desc: 'TaskContext.instruction → generate → parse a label' }
  ];

  const NEURAL_BENCH = {
    vision: 'MajajHong2015 V4 / IT (vision · neural)',
    text: 'Pereira2018 sentences (language · neural)',
    audio: 'Lahner2024 auditory-ROI (audio · neural)',
    video: 'Lahner2024 BOLDMoments (video · neural)'
  };

  // ---- routing engine: mirror of process() + the pre-flight compatibility check ----
  const S = (kind, label, sub, contract) => ({ kind, label, sub, contract });
  const ok = (steps, output, bench, model) =>
    ({ ok: true, steps, output, bench, model });
  const err = (etype, msg, steps) => ({ ok: false, etype, msg, steps: steps || [] });

  function pickByPriority(mods) {
    for (const m of MODALITY_PRIORITY) if (mods.includes(m)) return m;
    return mods[0];
  }

  function route(model, input, task, multiModality) {
    const C = model.has;
    const procStep = S('event', 'process(input_event)', `you hand the model a ${input.event}`, 'UnifiedModel.process');

    // 1 — StateChange (lesion / perturbation)
    if (input.event === 'StateChange') {
      if (!C.state_change) {
        return err('NotImplementedError',
          `'${model.name}' has no state_change_fn registered, so it can't be lesioned/perturbed.`,
          [procStep, S('decision', 'isinstance(StateChange)', 'routes to state_change_fn — but none is registered')]);
      }
      return ok([
        procStep,
        S('decision', 'isinstance(StateChange) → state_change_fn', 'first dispatch branch'),
        S('fn', 'state_change_fn(state_change)', 'resolve Selection (which units) → apply Perturbation (zero / scale / replace)', 'BrainScoreModel.state_change_fn'),
        S('output', 'PerturbationApplied', 'carries a handle_id; process(StateChange(kind="reset", handle_id=…)) restores bit-for-bit')
      ], 'PerturbationApplied', 'Yeatman2021-induced_dyslexia (lesion the word-form units)', model);
    }

    // 2 — EnvironmentStep (embodied closed loop)
    if (input.event === 'EnvironmentStep') {
      if (!C.action) {
        return err('NotImplementedError',
          `'${model.name}' has no action_fn registered; embodied evaluation needs action_fn(env_step) -> EnvironmentResponse.`,
          [procStep, S('decision', 'isinstance(EnvironmentStep)', 'routes to action_fn — but none is registered')]);
      }
      return ok([
        procStep,
        S('decision', 'isinstance(EnvironmentStep) → action_fn', 'embodied dispatch branch'),
        S('fn', 'action_fn(env_step)', 'see the rendered frame + legal actions, reason, pick a move', 'BrainScoreModel.action_fn'),
        S('output', 'EnvironmentResponse', 'action index fed back to the env; loop repeats (no reset between ticks)')
      ], 'EnvironmentResponse', 'GridGame-reach / MiniGrid (closed-loop)', model);
    }

    // 3 — perceptual input (StimulusSet / MultimodalStimulusSet)
    const inMods = Array.isArray(input.modality) ? input.modality : [input.modality];
    const supported = inMods.filter(m => model.available.includes(m));

    // 3a — behavioral generation path
    if (task === 'generation') {
      if (!C.generation) {
        return err('path unavailable',
          `'${model.name}' has no generation_fn, so the generation path can't fire. It would fall through to readout (if a readout layer exists) or neural encoding.`,
          [procStep, S('decision', 'TaskContext.instruction + generation_fn?', 'no generation_fn → skip this branch')]);
      }
      if (supported.length === 0) {
        return err('CompatibilityError',
          `pre-flight check_compatibility fails: this input is {${inMods.join(', ')}} but the model only reads {${model.available.join(', ')}}.`,
          [procStep, S('decision', 'check_compatibility()', 'required modality ⊄ model.available')]);
      }
      return ok([
        procStep,
        S('check', 'check_compatibility() · check_memory()', 'pre-flight: modality/region subset, then probe-based memory estimate'),
        S('decision', '_use_generation_for_task & instruction present', 'generation wins (prefer_path="auto")'),
        S('fn', '_generate_predictions(stimuli)', `read the ${pickByPriority(supported)} column → generation_fn(row, instruction, label_set) → parse a label`, 'BrainScoreModel._generate_predictions'),
        S('output', 'BehavioralAssembly', 'one-hot over label_set per stimulus')
      ], 'BehavioralAssembly', 'ROAR / Yeatman2021 (generation) · Rajalingham 2-AFC', model);
    }

    // 3b — behavioral readout path
    if (task === 'readout') {
      if (!C.activations) {
        return err('NotImplementedError',
          `'${model.name}' exposes no internal activations (activations_model=None), and a readout fits a logistic head on a layer. Use behavioral generation instead.`,
          [procStep, S('decision', 'behavioral_readout_layer + fitting_stimuli?', 'no activations to read from')]);
      }
      if (!C.readout) {
        return err('path unavailable',
          `'${model.name}' has no behavioral_readout_layer set, so the readout path can't fire (it would fall through to neural encoding).`,
          [procStep, S('decision', 'behavioral_readout_layer present?', 'none set → skip this branch')]);
      }
      if (supported.length === 0) {
        return err('CompatibilityError',
          `pre-flight check_compatibility fails: input {${inMods.join(', ')}} ⊄ model {${model.available.join(', ')}}.`,
          [procStep]);
      }
      const m = pickByPriority(supported);
      return ok([
        procStep,
        S('check', 'check_compatibility() · check_memory()', 'pre-flight'),
        S('decision', '_readout_classifier set & task needs readout', 'readout path'),
        S('fn', '_predict_probabilities(stimuli)', `${m} preprocessor → activations_model at behavioral_readout_layer → fitted logistic head`, 'BrainScoreModel._predict_probabilities'),
        S('output', 'BehavioralAssembly', 'probabilities over label_set')
      ], 'BehavioralAssembly', 'ROAR / Yeatman2021 (readout)', model);
    }

    // 3c — neural encoding (default perceptual path)
    if (!C.activations) {
      return err('NotImplementedError',
        `'${model.name}' is output-only (activations_model=None) — it exposes no layer activations, so it can't run activation-based neural encoding. Score it behaviorally (generation) instead.`,
        [procStep, S('check', 'check_compatibility()', 'passes modality, but there is no activations_model to extract from')]);
    }
    if (supported.length === 0) {
      return err('CompatibilityError',
        `pre-flight check_compatibility fails: the benchmark requires {${inMods.join(', ')}} but model.available = {${model.available.join(', ')}}. The contract is benchmark.required ⊆ model.available.`,
        [procStep, S('check', 'check_compatibility()', 'required ⊄ available → fail fast, before any compute')]);
    }

    const useMulti = multiModality && supported.length > 1;
    const towers = useMulti ? supported : [pickByPriority(supported)];
    const steps = [
      procStep,
      S('check', 'check_compatibility() · check_memory()', 'pre-flight: modality+region subset, then a one-stimulus memory probe'),
      S('decision', `_detect_modalities() → {${supported.join(', ')}}`,
        useMulti ? 'multi_modality=True → fan out to every supported tower'
                 : (supported.length > 1 ? `multi_modality=False → MODALITY_PRIORITY picks "${towers[0]}"` : 'single modality'))
    ];
    towers.forEach(mod => {
      const wrapperName = wrapperFor(model, mod);
      steps.push(S('fn', `preprocessors['${mod}'](stimuli)`, 'modality-specific transform (resize/normalize · tokenize · resample)', 'preprocessor callable'));
      steps.push(S('wrapper', `${wrapperName}(layers=region_layer_map)`, 'one forward pass, hooks capture the recording layer(s)', wrapperName));
    });
    steps.push(S('output', 'NeuroidAssembly',
      useMulti ? '(presentation, neuroid) — concat across towers with a per-neuroid modality coord'
               : '(presentation, neuroid) with layer (+ region) coords'));

    const benchName = useMulti ? 'Algonauts2025 / Lahner multimodal (banded ridge over towers)'
      : (input.id === 'movie' ? `${NEURAL_BENCH[towers[0]]} — only the "${towers[0]}" tower ran` : NEURAL_BENCH[towers[0]]);
    return ok(steps, useMulti ? 'NeuroidAssembly (multi-tower)' : 'NeuroidAssembly', benchName, model);
  }

  function wrapperFor(model, modality) {
    const map = { vision: model.id === 'vlm' ? 'VLMVisionWrapper' : 'PytorchWrapper',
      text: 'TextWrapper', audio: 'AudioWrapper', video: 'VideoWrapper' };
    return map[modality] || 'PytorchWrapper';
  }

  // ---- example code snippet for the resolved combo ----
  function snippet(model, input, task, result) {
    const id = `'${model.id}-model'`;
    if (input.event === 'StateChange') {
      return `m = load_model('qwen2.5-vl-3b')   # has a state_change_fn\n` +
        `handle = m.process(StateChange(target=sel, perturbation=Perturbation('zero')))\n` +
        `# … observe deficit … then restore:\n` +
        `m.process(StateChange(kind='reset', handle_id=handle.handle_id))`;
    }
    if (input.event === 'EnvironmentStep') {
      return `m = BrainScoreModel(${id}, action_fn=policy, …)\n` +
        `resp = m.process(EnvironmentStep(observation=obs, step_num=t))\n` +
        `env.step(int(resp.action))   # closed loop`;
    }
    if (!result.ok) {
      return `m = load_model(${id})\n` +
        `m.start_recording('IT')\n` +
        `m.process(stimulus_set)   # -> ${result.etype}`;
    }
    if (task === 'generation') {
      return `m = load_model(${id})\n` +
        `m.start_task(TaskContext(task_type='probabilities',\n` +
        `             instruction='Which object?', label_set=[...]))\n` +
        `assembly = m.process(stimulus_set)   # -> BehavioralAssembly`;
    }
    if (task === 'readout') {
      return `m = load_model(${id})   # behavioral_readout_layer set\n` +
        `m.start_task(TaskContext(task_type='probabilities', fitting_stimuli=train))\n` +
        `assembly = m.process(stimulus_set)   # -> BehavioralAssembly`;
    }
    const multi = (input.id === 'movie');
    return `m = load_model(${id})\n` +
      `m.start_recording(${model.id === 'vlm' ? "['V4','IT']" : "'IT'"})\n` +
      `assembly = m.process(stimulus_set${multi ? ', multi_modality=True' : ''})   # -> NeuroidAssembly`;
  }

  // ============================ rendering ============================
  const state = { model: 'vlm', input: 'image', task: 'neural', multi: false };
  const $ = sel => document.querySelector(sel);
  const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };

  function renderControls() {
    const mWrap = $('#arch-models'); mWrap.innerHTML = '';
    MODELS.forEach(m => {
      const b = el('button', 'arch-chip' + (m.id === state.model ? ' on' : ''),
        `<b>${m.name}</b><span>${m.eg}</span>`);
      b.onclick = () => { state.model = m.id; renderAll(); };
      mWrap.appendChild(b);
    });
    const iWrap = $('#arch-inputs'); iWrap.innerHTML = '';
    INPUTS.forEach(inp => {
      const b = el('button', 'arch-chip' + (inp.id === state.input ? ' on' : ''),
        `<b>${inp.name}</b><span>${inp.event}</span>`);
      b.onclick = () => { state.input = inp.id; renderAll(); };
      iWrap.appendChild(b);
    });
  }

  function renderTaskRow() {
    const input = INPUTS.find(i => i.id === state.input);
    const row = $('#arch-taskrow');
    if (!input.perceptual) { row.style.display = 'none'; return; }
    row.style.display = '';
    const tWrap = $('#arch-tasks'); tWrap.innerHTML = '';
    TASKS.forEach(t => {
      const b = el('button', 'arch-task' + (t.id === state.task ? ' on' : ''),
        `${t.name}<small>${t.desc}</small>`);
      b.onclick = () => { state.task = t.id; renderAll(); };
      tWrap.appendChild(b);
    });
    const mm = $('#arch-multi');
    mm.style.display = (input.id === 'movie' && state.task === 'neural') ? 'inline-flex' : 'none';
    $('#arch-multi-cb').checked = state.multi;
  }

  function renderFlow() {
    const model = MODELS.find(m => m.id === state.model);
    const input = INPUTS.find(i => i.id === state.input);
    const task = input.perceptual ? state.task : null;
    const result = route(model, input, task, state.multi);

    // verdict
    const v = $('#arch-verdict');
    if (result.ok) {
      v.className = 'arch-verdict ok';
      v.innerHTML = `<span class="vbadge">✓ routes</span> resolves to <code>${result.output}</code>`;
    } else {
      v.className = 'arch-verdict bad';
      v.innerHTML = `<span class="vbadge">✗ ${result.etype}</span> ${result.msg}`;
    }

    // flow chain
    const flow = $('#arch-flow'); flow.innerHTML = '';
    result.steps.forEach((st, i) => {
      const box = el('div', 'flow-box ' + st.kind);
      box.innerHTML = `<div class="fb-label">${st.label}</div>` +
        (st.sub ? `<div class="fb-sub">${st.sub}</div>` : '') +
        (st.contract ? `<div class="fb-contract">${st.contract}</div>` : '');
      flow.appendChild(box);
      if (i < result.steps.length - 1) flow.appendChild(el('div', 'flow-arrow', '↓'));
    });
    if (!result.ok) {
      const e = el('div', 'flow-box error');
      e.innerHTML = `<div class="fb-label">${result.etype}</div><div class="fb-sub">${result.msg}</div>`;
      flow.appendChild(el('div', 'flow-arrow', '↓'));
      flow.appendChild(e);
    }

    // meta: example benchmark + model blurb
    $('#arch-meta').innerHTML =
      `<div class="kv"><span>example test</span><b>${result.bench || '—'}</b></div>` +
      `<div class="kv"><span>this model is</span><b>${model.blurb}</b></div>` +
      `<div class="kv"><span>declares</span><b>available={${model.available.join(', ') || '∅'}}` +
      `${model.required.length ? ` · required={${model.required.join(', ')}}` : ''}` +
      ` · wrappers={${model.wrappers.join(', ') || 'none'}}</b></div>`;

    // code
    $('#arch-code').textContent = snippet(model, input, task, result);
  }

  function renderMatrix() {
    const wrap = $('#arch-matrix'); if (!wrap) return;
    let html = '<table class="arch-mtx"><thead><tr><th></th>';
    INPUTS.forEach(i => html += `<th>${i.name}</th>`);
    html += '</tr></thead><tbody>';
    MODELS.forEach(m => {
      html += `<tr><th>${m.name}</th>`;
      INPUTS.forEach(inp => {
        const task = inp.perceptual ? 'neural' : null;
        const r = route(m, inp, task, false);
        const cls = r.ok ? 'cm-ok' : (r.etype === 'CompatibilityError' ? 'cm-incompat' : 'cm-na');
        const out = r.ok ? r.output.replace(' (multi-tower)', '') : r.etype;
        html += `<td class="${cls}" data-m="${m.id}" data-i="${inp.id}" title="${(r.ok ? 'routes to ' + r.output : r.msg).replace(/"/g, '&quot;')}">${out}</td>`;
      });
      html += '</tr>';
    });
    html += '</tbody></table>';
    wrap.innerHTML = html;
    wrap.querySelectorAll('td[data-m]').forEach(td => {
      td.onclick = () => {
        state.model = td.dataset.m; state.input = td.dataset.i;
        if (INPUTS.find(i => i.id === state.input).perceptual) state.task = 'neural';
        renderAll();
        $('#arch-explorer').scrollIntoView({ behavior: 'smooth', block: 'start' });
      };
    });
  }

  function renderAll() {
    renderControls();
    renderTaskRow();
    renderFlow();
    renderMatrix();
  }

  if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
      const cb = document.querySelector('#arch-multi-cb');
      if (cb) cb.addEventListener('change', e => { state.multi = e.target.checked; renderFlow(); });
      renderAll();
    });
  }

  // Expose the pure routing engine for tests (no DOM required).
  const __api = { route, MODELS, INPUTS, TASKS };
  if (typeof module !== 'undefined' && module.exports) module.exports = __api;
  if (typeof window !== 'undefined') window.__ARCH = __api;
})();
