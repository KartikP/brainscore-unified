export type Mode = 'benchmark' | 'stimuli' | 'session' | 'calibration';
interface SubjectInfo { identifier: string; in_channels: string[]; out_channels: string[]; calibration?: boolean; schemas?: Record<string, { payload: string; addressing?: string; json_input?: boolean }> }
interface Info { phase: string; token: string; profile: SubjectInfo | null; catalog_ready?: boolean; run?: { request?: unknown } | null }
interface Callbacks { state: () => Info; post: (path: string, data: object) => Promise<unknown>; changed: () => void; loading: () => void; error: (message: string) => void }
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
let mode: Mode = 'calibration';
let subjectID = '';
let catalogLoaded = false;
let catalogPending = false;
let callbacks: Callbacks;
let uploading = false;
let restoredForm = false;
let modelEntries: { identifier: string; source: string }[] = [];
let matchingModelCount = 0;
export const currentMode = () => mode;
export const isUploading = () => uploading;
export const customIntervention = () => mode !== 'benchmark' && mode !== 'calibration' && $<HTMLInputElement>('custom-intervention').checked;
export const requestedOutput = () => $<HTMLSelectElement>(mode === 'session' ? 'session-output' : 'output-channel').value;

function chooseMode(value: Mode): void {
  mode = value;
  callbacks.changed();
}

function options(id: string, values: string[], previous = ''): void {
  const select = $<HTMLSelectElement>(id); select.replaceChildren();
  for (const value of values) { const option = document.createElement('option'); option.value = value; option.textContent = value; select.append(option); }
  if (values.includes(previous)) select.value = previous;
}

export function parseCSV(source: string): Record<string, string>[] {
  const rows: string[][] = []; let row: string[] = [], cell = '', quoted = false;
  for (let i = 0; i < source.length; i++) {
    const char = source[i];
    if (char === '"') { if (quoted && source[i + 1] === '"') { cell += '"'; i++; } else quoted = !quoted; }
    else if (char === ',' && !quoted) { row.push(cell); cell = ''; }
    else if (char === '\n' && !quoted) { row.push(cell.replace(/\r$/, '')); rows.push(row); row = []; cell = ''; }
    else cell += char;
  }
  if (quoted) throw new Error('CSV contains an unclosed quoted field');
  if (cell || row.length) { row.push(cell.replace(/\r$/, '')); rows.push(row); }
  const headers = rows.shift()?.map(header => header.trim()) ?? [];
  if (!headers.length || headers.some(header => !header) || new Set(headers).size !== headers.length) throw new Error('CSV needs distinct, nonempty column headers');
  return rows.filter(row => row.some(cell => cell.trim())).map(row => {
    if (row.length !== headers.length) throw new Error('Each CSV row must have the same number of columns as its header');
    return Object.fromEntries(headers.map((header, i) => [header, row[i]]));
  });
}

function filterModelChoices(selected = $<HTMLSelectElement>('model-id').value): void {
  const search = $<HTMLInputElement>('model-search').value.trim();
  const normalize = (value: string) => value.toLowerCase().replace(/[^a-z0-9]/g, '');
  const words = search.toLowerCase().split(/\s+/).map(normalize).filter(Boolean);
  const matching = modelEntries.filter(entry => words.every(word => normalize(`${entry.identifier} ${entry.source}`).includes(word)));
  matchingModelCount = matching.length;
  const select = $<HTMLSelectElement>('model-id');
  select.replaceChildren();
  if (!search) select.append(new Option('Select a registered model', ''));
  if (!matching.length) select.append(new Option('No matching models', ''));
  const sources = [...new Set(matching.map(entry => entry.source))];
  sources.sort((a, b) => a === 'unified' ? -1 : b === 'unified' ? 1 : a.localeCompare(b));
  for (const source of sources) {
    const group = document.createElement('optgroup'); group.label = `${source[0].toUpperCase()}${source.slice(1)}`;
    const entries = matching.filter(entry => entry.source === source);
    entries.sort((a, b) => {
      const prefixA = words.length > 0 && normalize(a.identifier).startsWith(words[0]);
      const prefixB = words.length > 0 && normalize(b.identifier).startsWith(words[0]);
      return Number(prefixB) - Number(prefixA) || a.identifier.localeCompare(b.identifier);
    });
    for (const entry of entries) group.append(new Option(entry.identifier, entry.identifier));
    select.append(group);
  }
  select.size = search ? Math.min(8, Math.max(3, matching.length + sources.length)) : 1;
  if (matching.some(entry => entry.identifier === selected)) select.value = selected;
  else select.value = search && matching.length ? select.options[0].value : '';
  $('model-search-clear').hidden = !search;
  $('model-search-status').textContent = search
    ? matching.length ? `${matching.length} matching model${matching.length === 1 ? '' : 's'} / ${modelEntries.length}. Select one below.` : 'No matching models. Try a shorter name or clear the search.'
    : `${modelEntries.length} models. Type part of a name to filter instantly.`;
}

async function readCatalog(): Promise<void> {
  catalogPending = true;
  try {
    const response = await fetch('/api/catalog'); if (!response.ok) throw new Error('Registry catalog unavailable');
    const result = await response.json() as { models: { identifier: string; source: string }[]; benchmarks: { identifier: string; source: string }[]; stimulus_sets: string[] };
    const select = $<HTMLSelectElement>('model-id');
    const selected = select.value || (callbacks.state().profile?.calibration ? '' : callbacks.state().profile?.identifier ?? '');
    modelEntries = result.models;
    if (selected && !modelEntries.some(entry => entry.identifier === selected)) modelEntries.push({ identifier: selected, source: 'loaded subject' });
    filterModelChoices(selected);
    const benchmarkList = $('benchmark-options'); benchmarkList.replaceChildren();
    for (const entry of result.benchmarks) { const option = document.createElement('option'); option.value = entry.identifier; option.label = entry.source; benchmarkList.append(option); }
    const stimuli = $('stimulus-options'); stimuli.replaceChildren();
    for (const name of result.stimulus_sets) { const option = document.createElement('option'); option.value = name; stimuli.append(option); }
    $('catalog-count').textContent = `${result.models.length} model IDs / ${result.benchmarks.length} benchmark IDs. Loading may require weights or data.`;
    catalogLoaded = true;
    callbacks.changed();
  } catch (problem) { callbacks.error(String(problem)); }
  finally { catalogPending = false; }
}

function sessionExample(notify = true): void {
  const channel = $<HTMLSelectElement>('session-input').value;
  const payload = channel === 'generation_request' ? { messages: [{ role: 'user', content: 'Describe what an experiment measures.' }], temperature: 0, max_tokens: 64 }
    : channel === 'conditional_logprob_request' ? { contexts: ['The capital of France is', 'The capital of France is'], targets: [' Paris', ' London'], reduction: 'sum' }
      : 'A stimulus to present to the subject.';
  $<HTMLTextAreaElement>('session-events').value = JSON.stringify([{ channel, payload, t_ms: null }], null, 2);
  if (notify) callbacks.changed();
}

function toCSV(rows: Record<string, unknown>[]): string {
  const headers = [...new Set(rows.flatMap(row => Object.keys(row)))];
  const quote = (value: unknown) => `"${String(value ?? '').replaceAll('"', '""')}"`;
  return [headers.join(','), ...rows.map(row => headers.map(header => quote(row[header])).join(','))].join('\n');
}

function restoreRequest(value: unknown): void {
  const saved = value as { mode: Mode; benchmark: string; rows: Record<string, unknown>[]; fitting_rows: Record<string, unknown>[]; stimulus_set: string; output: string; instruction: string; labels: string[]; prefer_path: string; events: unknown[]; intervention?: { target: string; indices?: number[]; operation?: string; value?: number } };
  if (!saved || !['benchmark', 'stimuli', 'session'].includes(saved.mode)) return;
  mode = saved.mode;
  $<HTMLInputElement>('benchmark-id').value = saved.benchmark ?? '';
  $<HTMLSelectElement>('output-channel').value = saved.output;
  $<HTMLSelectElement>('session-output').value = saved.output;
  if (saved.stimulus_set) {
    $<HTMLSelectElement>('stimulus-format').value = 'registered';
    $<HTMLInputElement>('stimulus-set-id').value = saved.stimulus_set;
  } else if (saved.rows?.length) {
    const lines = saved.rows.every(row => Object.keys(row).length === 1 && typeof row.sentence === 'string');
    $<HTMLSelectElement>('stimulus-format').value = lines ? 'lines' : 'csv';
    $<HTMLTextAreaElement>('stimulus-text').value = lines ? saved.rows.map(row => String(row.sentence)).join('\n') : toCSV(saved.rows);
  }
  $<HTMLTextAreaElement>('instruction').value = saved.instruction ?? '';
  $<HTMLInputElement>('labels').value = saved.labels?.join(',') ?? '';
  $<HTMLSelectElement>('behavior-path').value = saved.prefer_path ?? 'auto';
  $<HTMLTextAreaElement>('fitting-text').value = saved.fitting_rows?.length ? toCSV(saved.fitting_rows) : '';
  if (saved.events?.length) $<HTMLTextAreaElement>('session-events').value = JSON.stringify(saved.events, null, 2);
  $<HTMLInputElement>('custom-intervention').checked = Boolean(saved.intervention);
  if (saved.intervention) {
    $<HTMLInputElement>('intervention-target').value = saved.intervention.target;
    $<HTMLInputElement>('intervention-indices').value = saved.intervention.indices?.join(',') ?? '';
    $<HTMLSelectElement>('intervention-operation').value = saved.intervention.operation ?? 'zero';
    $<HTMLInputElement>('intervention-value').value = String(saved.intervention.value ?? 0);
  }
}

export function updateWorkbench(state: Info, busy: boolean): void {
  if (state.catalog_ready && !catalogLoaded && !catalogPending) void readCatalog();
  const profile = state.profile;
  if (profile && profile.identifier !== subjectID) {
    subjectID = profile.identifier;
    if (mode === 'calibration' && !profile.calibration) mode = 'stimuli';
    const modelSelect = $<HTMLSelectElement>('model-id');
    if (!profile.calibration && ![...modelSelect.options].some(option => option.value === profile.identifier)) modelSelect.append(new Option(profile.identifier, profile.identifier));
    modelSelect.value = profile.calibration ? '' : profile.identifier;
    const outputs = profile.out_channels.filter(channel => channel === 'behavior' || channel.startsWith('neural:') || channel === 'response_trace' && profile.in_channels.includes('generation_request'));
    if (outputs.includes('response_trace')) { outputs.splice(outputs.indexOf('response_trace'), 1); outputs.unshift('response_trace'); }
    options('output-channel', outputs, requestedOutput());
    options('session-output', profile.out_channels, profile.out_channels.includes('response_trace') ? 'response_trace' : '');
    options('session-input', profile.in_channels.filter(channel => profile.schemas?.[channel]?.json_input));
    if (profile.in_channels.includes('generation_request')) $<HTMLSelectElement>('session-input').value = 'generation_request';
    $<HTMLInputElement>('custom-intervention').checked = false;
    if (!$<HTMLSelectElement>('output-channel').options.length && !profile.calibration) mode = 'session';
    sessionExample(false);
  }
  if (profile && !restoredForm) { restoredForm = true; restoreRequest(state.run?.request); }
  document.querySelectorAll<HTMLButtonElement>('[data-mode]').forEach(button => {
    button.setAttribute('aria-selected', String(button.dataset.mode === mode));
    button.disabled = busy || (button.dataset.mode === 'calibration' && !profile?.calibration);
  });
  for (const item of ['benchmark', 'stimuli', 'session', 'calibration']) $(`${item}-editor`).hidden = item !== mode;
  $('calibration-settings').hidden = mode !== 'calibration';
  $('behavior-settings').hidden = requestedOutput() !== 'behavior' || mode !== 'stimuli';
  $('instruction-settings').hidden = mode !== 'stimuli' || !['behavior', 'response_trace'].includes(requestedOutput());
  const registered = $<HTMLSelectElement>('stimulus-format').value === 'registered';
  $('registered-stimulus-editor').hidden = !registered;
  $('stimulus-text-editor').hidden = registered;
  $('intervention-editor').hidden = mode === 'benchmark' || mode === 'calibration';
  const canIntervene = Boolean(profile?.out_channels.includes('perturbation') && profile.in_channels.some(channel => ['lesion', 'stimulation'].includes(channel)));
  $<HTMLInputElement>('custom-intervention').disabled = busy || !canIntervene;
  $<HTMLSelectElement>('model-id').disabled = busy || state.phase === 'loading' || !catalogLoaded;
  $<HTMLInputElement>('model-search').disabled = busy || state.phase === 'loading' || !catalogLoaded;
  $<HTMLButtonElement>('model-search-clear').disabled = busy || state.phase === 'loading' || !catalogLoaded;
  $<HTMLSelectElement>('model-id').disabled ||= matchingModelCount === 0;
  $<HTMLButtonElement>('load-model').disabled = busy || state.phase === 'loading' || !catalogLoaded || !$<HTMLSelectElement>('model-id').value;
  $<HTMLButtonElement>('load-calibration').disabled = busy || state.phase === 'loading';
  document.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>('.experiment-editor input,.experiment-editor select,.experiment-editor textarea').forEach(input => { input.disabled = busy; });
  $<HTMLInputElement>('custom-intervention').disabled = busy || !canIntervene;
  const channel = $<HTMLSelectElement>('session-input').value;
  $('channel-schema').textContent = profile?.schemas?.[channel]?.payload ?? 'JSON-native request channels are supported here. Typed environments and custom schemas use their benchmark or a Python harness.';
  $('workflow-help').textContent = mode === 'benchmark' ? 'The registered benchmark owns its experiment. Exploratory tools are disabled here; its score and provenance are saved.'
    : mode === 'session' ? 'Send events through interact(session). JSON requests use declared channels. Environments and non-JSON payloads require a typed harness.'
      : mode === 'stimuli' ? 'Choose response_trace for generated text, neural:<region> for activations, or behavior for labeled task responses. Generated text is kept verbatim.'
        : 'Calibration checks the tools. Load a registered subject to run your own science.';
}

export function handleTool(name: string, enabled: boolean): void {
  if (mode === 'calibration') return;
  if (name === 'activity' && enabled) {
    const channel = callbacks.state().profile?.out_channels.find(channel => channel.startsWith('neural:'));
    if (channel) { mode = 'stimuli'; $<HTMLSelectElement>('output-channel').value = channel; }
  }
  if (name === 'ablation') $<HTMLInputElement>('custom-intervention').checked = enabled;
}

export function buildRequest(record: boolean): object {
  const intervention = customIntervention() ? {
    target: $<HTMLInputElement>('intervention-target').value.trim(),
    indices: $<HTMLInputElement>('intervention-indices').value.trim() ? $<HTMLInputElement>('intervention-indices').value.split(',').map(value => Number(value.trim())) : null,
    operation: $<HTMLSelectElement>('intervention-operation').value,
    value: Number($<HTMLInputElement>('intervention-value').value),
  } : null;
  if (mode === 'benchmark') return { mode, benchmark: $<HTMLInputElement>('benchmark-id').value.trim() };
  if (mode === 'session') return { mode, output: $<HTMLSelectElement>('session-output').value, events: JSON.parse($<HTMLTextAreaElement>('session-events').value), record, intervention };
  const format = $<HTMLSelectElement>('stimulus-format').value;
  return { mode, output: requestedOutput(), record, intervention,
    rows: format === 'registered' ? [] : format === 'csv' ? parseCSV($<HTMLTextAreaElement>('stimulus-text').value)
      : $<HTMLTextAreaElement>('stimulus-text').value.split('\n').filter(line => line.trim()).map(sentence => ({ sentence })),
    stimulus_set: format === 'registered' ? $<HTMLInputElement>('stimulus-set-id').value.trim() : '',
    instruction: $<HTMLTextAreaElement>('instruction').value,
    labels: $<HTMLInputElement>('labels').value.split(',').map(label => label.trim()).filter(Boolean),
    prefer_path: $<HTMLSelectElement>('behavior-path').value,
    fitting_rows: $<HTMLTextAreaElement>('fitting-text').value.trim() ? parseCSV($<HTMLTextAreaElement>('fitting-text').value) : [],
  };
}

export function initializeWorkbench(api: Callbacks): void {
  callbacks = api;
  document.querySelectorAll<HTMLButtonElement>('[data-mode]').forEach(button => { button.onclick = () => chooseMode(button.dataset.mode as Mode); });
  const load = async (identifier: string) => {
    try { callbacks.error(''); await callbacks.post('/api/model', { identifier }); callbacks.loading(); }
    catch (problem) { callbacks.error(String(problem)); }
  };
  $('load-model').onclick = () => void load($<HTMLSelectElement>('model-id').value);
  $<HTMLSelectElement>('model-id').onchange = callbacks.changed;
  $<HTMLInputElement>('model-search').oninput = () => { filterModelChoices(); callbacks.changed(); };
  $('model-search-clear').onclick = () => {
    $<HTMLInputElement>('model-search').value = ''; filterModelChoices(); callbacks.changed();
    $<HTMLInputElement>('model-search').focus();
  };
  $<HTMLInputElement>('model-search').onkeydown = event => {
    if ((event.key === 'ArrowDown' || event.key === 'Enter') && matchingModelCount > 0) {
      event.preventDefault(); $<HTMLSelectElement>('model-id').focus();
    } else if (event.key === 'Escape') {
      $<HTMLInputElement>('model-search').value = ''; filterModelChoices(); callbacks.changed();
    }
  };
  $('load-calibration').onclick = () => { mode = 'calibration'; void load('__calibration__'); };
  document.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>('.experiment-editor input,.experiment-editor select').forEach(input => { input.onchange = callbacks.changed; });
  $<HTMLSelectElement>('session-input').onchange = () => sessionExample();
  $<HTMLInputElement>('csv-file').onchange = async event => {
    try {
      const file = (event.target as HTMLInputElement).files?.[0]; if (!file) return;
      if (file.size > 1024 * 1024) throw new Error('CSV import limit is 1 MiB; use a registered stimulus set for larger datasets');
      const source = await file.text(); const rows = parseCSV(source);
      $<HTMLTextAreaElement>('stimulus-text').value = source; $<HTMLSelectElement>('stimulus-format').value = 'csv';
      $('upload-status').textContent = `${rows.length} stimulus rows imported`; callbacks.changed();
    } catch (problem) { callbacks.error(String(problem)); }
  };
  $<HTMLInputElement>('media-files').onchange = async event => {
    uploading = true; callbacks.changed();
    try {
      const files = [...((event.target as HTMLInputElement).files ?? [])];
      const rows: Record<string, string>[] = [];
      for (const file of files) {
        if (file.size > 128 * 1024 * 1024) throw new Error('Media upload limit is 128 MiB per file');
        $('upload-status').textContent = `Uploading ${file.name}...`;
        const response = await fetch('/api/uploads', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Lab-Token': callbacks.state().token, 'X-Lab-Filename': encodeURIComponent(file.name) }, body: file });
        const result = await response.json(); if (!response.ok) throw new Error(result.error);
        const suffix = file.name.split('.').pop()?.toLowerCase() ?? '';
        const column = ['wav', 'mp3', 'flac', 'ogg'].includes(suffix) ? 'audio_path' : ['mp4', 'mov', 'webm'].includes(suffix) ? 'video_path' : 'image_path';
        rows.push({ stimulus_id: file.name, [column]: result.asset });
      }
      const headers = [...new Set(rows.flatMap(row => Object.keys(row)))];
      const quote = (text: string) => `"${text.replaceAll('"', '""')}"`;
      $<HTMLTextAreaElement>('stimulus-text').value = [headers.join(','), ...rows.map(row => headers.map(header => quote(row[header] ?? '')).join(','))].join('\n');
      $<HTMLSelectElement>('stimulus-format').value = 'csv'; $('upload-status').textContent = `${rows.length} files ready as stimulus rows`;
    } catch (problem) { callbacks.error(String(problem)); }
    finally { uploading = false; callbacks.changed(); }
  };
}
