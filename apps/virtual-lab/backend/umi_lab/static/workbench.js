import { initializeProtocol, updateProtocol, protocolRequest, restoreProtocol, protocolSpec, protocolTool } from './protocol.js';
import { renderDownload } from './downloads.js';
const $ = (id) => document.getElementById(id);
let mode = 'protocol';
let subjectID = '';
let catalogLoaded = false;
let catalogPending = false;
let callbacks;
let uploading = false;
let downloadPolicyPending = false;
let restoredForm = false;
let modelEntries = [];
let matchingModelCount = 0;
let previousFormat = 'lines';
const stimulusDrafts = {};
export const currentMode = () => mode;
export const isUploading = () => uploading;
export const isUpdatingSettings = () => downloadPolicyPending;
export const customIntervention = () => mode !== 'benchmark' && mode !== 'calibration' && $('custom-intervention').checked;
export const requestedOutput = () => $(mode === 'session' ? 'session-output' : 'output-channel').value;
function chooseMode(value) {
    mode = value;
    callbacks.changed();
}
function options(id, values, previous = '') {
    const select = $(id);
    select.replaceChildren();
    for (const value of values) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        select.append(option);
    }
    if (values.includes(previous))
        select.value = previous;
}
export function parseCSV(source) {
    const rows = [];
    let row = [], cell = '', quoted = false;
    for (let i = 0; i < source.length; i++) {
        const char = source[i];
        if (char === '"') {
            if (quoted && source[i + 1] === '"') {
                cell += '"';
                i++;
            }
            else
                quoted = !quoted;
        }
        else if (char === ',' && !quoted) {
            row.push(cell);
            cell = '';
        }
        else if (char === '\n' && !quoted) {
            row.push(cell.replace(/\r$/, ''));
            rows.push(row);
            row = [];
            cell = '';
        }
        else
            cell += char;
    }
    if (quoted)
        throw new Error('CSV contains an unclosed quoted field');
    if (cell || row.length) {
        row.push(cell.replace(/\r$/, ''));
        rows.push(row);
    }
    const headers = rows.shift()?.map(header => header.trim()) ?? [];
    if (!headers.length || headers.some(header => !header) || new Set(headers).size !== headers.length)
        throw new Error('CSV needs distinct, nonempty column headers');
    return rows.filter(row => row.some(cell => cell.trim())).map(row => {
        if (row.length !== headers.length)
            throw new Error('Each CSV row must have the same number of columns as its header');
        return Object.fromEntries(headers.map((header, i) => [header, row[i]]));
    });
}
function filterModelChoices(selected = $('model-id').value) {
    const search = $('model-search').value.trim();
    const normalize = (value) => value.toLowerCase().replace(/[^a-z0-9]/g, '');
    const words = search.toLowerCase().split(/\s+/).map(normalize).filter(Boolean);
    const matching = modelEntries.filter(entry => words.every(word => normalize(`${entry.identifier} ${entry.source}`).includes(word)));
    matchingModelCount = matching.length;
    const select = $('model-id');
    select.replaceChildren();
    if (!search)
        select.append(new Option('Select a registered model', ''));
    if (!matching.length)
        select.append(new Option('No matching models', ''));
    const sources = [...new Set(matching.map(entry => entry.source))];
    sources.sort((a, b) => a === 'unified' ? -1 : b === 'unified' ? 1 : a.localeCompare(b));
    for (const source of sources) {
        const group = document.createElement('optgroup');
        group.label = `${source[0].toUpperCase()}${source.slice(1)}`;
        const entries = matching.filter(entry => entry.source === source);
        entries.sort((a, b) => {
            const prefixA = words.length > 0 && normalize(a.identifier).startsWith(words[0]);
            const prefixB = words.length > 0 && normalize(b.identifier).startsWith(words[0]);
            return Number(prefixB) - Number(prefixA) || a.identifier.localeCompare(b.identifier);
        });
        for (const entry of entries)
            group.append(new Option(entry.identifier, entry.identifier));
        select.append(group);
    }
    select.size = search ? Math.min(8, Math.max(3, matching.length + sources.length)) : 1;
    if (matching.some(entry => entry.identifier === selected))
        select.value = selected;
    else
        select.value = search && matching.length ? select.options[0].value : '';
    $('model-search-clear').hidden = !search;
    $('model-search-status').textContent = search
        ? matching.length ? `${matching.length} matching model${matching.length === 1 ? '' : 's'} / ${modelEntries.length}. Select one below.` : 'No matching models. Try a shorter name or clear the search.'
        : `${modelEntries.length} models. Type part of a name to filter instantly.`;
}
async function readCatalog() {
    catalogPending = true;
    try {
        const response = await fetch('/api/catalog');
        if (!response.ok)
            throw new Error('Registry catalog unavailable');
        const result = await response.json();
        const select = $('model-id');
        const selected = select.value || (callbacks.state().profile?.calibration ? '' : callbacks.state().profile?.identifier ?? '');
        modelEntries = result.models;
        if (selected && !modelEntries.some(entry => entry.identifier === selected))
            modelEntries.push({ identifier: selected, source: 'loaded subject' });
        filterModelChoices(selected);
        const benchmarkList = $('benchmark-options');
        benchmarkList.replaceChildren();
        for (const entry of result.benchmarks) {
            const option = document.createElement('option');
            option.value = entry.identifier;
            option.label = entry.source;
            benchmarkList.append(option);
        }
        const stimuli = $('stimulus-options');
        stimuli.replaceChildren();
        for (const name of result.stimulus_sets) {
            const option = document.createElement('option');
            option.value = name;
            stimuli.append(option);
        }
        $('catalog-count').textContent = `${result.models.length} model IDs / ${result.benchmarks.length} benchmark IDs. Loading may require weights or data.`;
        catalogLoaded = true;
        callbacks.changed();
    }
    catch (problem) {
        callbacks.error(String(problem));
    }
    finally {
        catalogPending = false;
    }
}
async function uploadMediaFiles(files, progress) {
    uploading = true;
    callbacks.changed();
    try {
        const rows = [];
        for (const file of files) {
            if (file.size > 128 * 1024 * 1024)
                throw new Error('Media upload limit is 128 MiB per file');
            progress(file.name);
            const response = await fetch('/api/uploads', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Lab-Token': callbacks.state().token, 'X-Lab-Filename': encodeURIComponent(file.name) }, body: file });
            const result = await response.json();
            if (!response.ok)
                throw new Error(result.error);
            const suffix = file.name.split('.').pop()?.toLowerCase() ?? '';
            const column = ['wav', 'mp3', 'flac', 'ogg'].includes(suffix) ? 'audio_path' : ['mp4', 'mov', 'webm'].includes(suffix) ? 'video_path' : 'image_path';
            const row = { stimulus_id: file.name, [column]: result.asset };
            if (column === 'video_path' && $('extract-video-audio').checked && callbacks.state().profile?.in_channels.includes('audio')) {
                progress(`Extract audio: ${file.name}`);
                await attachAudio(row);
            }
            rows.push(row);
        }
        return rows;
    }
    finally {
        uploading = false;
        callbacks.changed();
    }
}
async function attachAudio(row) {
    const result = await callbacks.post('/api/extract-audio', { asset: row.video_path });
    if (!result.audio_asset) {
        row.audio_extraction = 'No audio stream in the video';
        callbacks.error('This video has no audio stream. Supply a separate audio file for a subject that requires audio.');
        return;
    }
    row.audio_path = result.audio_asset;
    row.audio_source = row.video_path;
    row.audio_preparation = result.preparation;
}
function sessionExample(notify = true) {
    const channel = $('session-input').value;
    const payload = channel === 'generation_request' ? { messages: [{ role: 'user', content: 'Describe what an experiment measures.' }], temperature: 0, max_tokens: 64 }
        : channel === 'conditional_logprob_request' ? { contexts: ['The capital of France is', 'The capital of France is'], targets: [' Paris', ' London'], reduction: 'sum' }
            : 'A stimulus to present to the subject.';
    $('session-events').value = JSON.stringify([{ channel, payload, t_ms: null }], null, 2);
    if (notify)
        callbacks.changed();
}
function toCSV(rows) {
    const headers = [...new Set(rows.flatMap(row => Object.keys(row)))];
    const quote = (value) => `"${String(value ?? '').replaceAll('"', '""')}"`;
    return [headers.join(','), ...rows.map(row => headers.map(header => quote(row[header])).join(','))].join('\n');
}
function restoreRequest(value) {
    const saved = value;
    if (!saved || !['protocol', 'benchmark', 'stimuli', 'session'].includes(saved.mode))
        return;
    mode = saved.mode;
    $('benchmark-id').value = saved.benchmark ?? '';
    $('output-channel').value = saved.output;
    $('session-output').value = saved.output;
    if (saved.stimulus_set) {
        $('stimulus-format').value = 'registered';
        $('stimulus-set-id').value = saved.stimulus_set;
    }
    else if (saved.rows?.length) {
        const lines = saved.rows.every(row => Object.keys(row).length === 1 && typeof row.sentence === 'string');
        const media = saved.rows.some(row => ['image_path', 'audio_path', 'video_path'].some(column => column in row));
        $('stimulus-format').value = lines ? 'lines' : media ? 'media' : 'csv';
        $('stimulus-text').value = lines ? saved.rows.map(row => String(row.sentence)).join('\n') : toCSV(saved.rows);
    }
    previousFormat = $('stimulus-format').value;
    $('instruction').value = saved.instruction ?? '';
    $('labels').value = saved.labels?.join(',') ?? '';
    $('behavior-path').value = saved.prefer_path ?? 'auto';
    $('fitting-text').value = saved.fitting_rows?.length ? toCSV(saved.fitting_rows) : '';
    if (saved.mode === 'protocol' && saved.protocol) {
        restoreProtocol(saved.protocol);
        $('protocol-fitting').value = saved.fitting_rows?.length ? toCSV(saved.fitting_rows) : '';
    }
    else if (saved.mode === 'stimuli' && saved.output?.startsWith('neural:')) {
        mode = 'protocol';
        const draft = protocolSpec();
        draft.measurements.regions = [saved.output.slice(7)];
        restoreProtocol(draft);
    }
    if (saved.events?.length)
        $('session-events').value = JSON.stringify(saved.events, null, 2);
    $('custom-intervention').checked = Boolean(saved.intervention);
    if (saved.intervention) {
        $('intervention-target').value = saved.intervention.target;
        $('intervention-indices').value = saved.intervention.indices?.join(',') ?? '';
        $('intervention-operation').value = saved.intervention.operation ?? 'zero';
        $('intervention-value').value = String(saved.intervention.value ?? 0);
    }
}
export function updateWorkbench(state, busy) {
    renderDownload(state.download, state.phase);
    const downloads = $('allow-model-downloads');
    if (!downloadPolicyPending)
        downloads.checked = state.allow_model_downloads ?? true;
    downloads.disabled = busy || state.phase !== 'ready' && state.phase !== 'load_failed';
    if (state.catalog_ready && !catalogLoaded && !catalogPending)
        void readCatalog();
    const profile = state.profile;
    if (profile && profile.identifier !== subjectID) {
        subjectID = profile.identifier;
        if (mode === 'calibration' && !profile.calibration)
            mode = 'protocol';
        const modelSelect = $('model-id');
        if (!profile.calibration && ![...modelSelect.options].some(option => option.value === profile.identifier))
            modelSelect.append(new Option(profile.identifier, profile.identifier));
        modelSelect.value = profile.calibration ? '' : profile.identifier;
        const outputs = profile.out_channels.filter(channel => channel === 'behavior' || channel.startsWith('neural:') || channel === 'response_trace' && profile.in_channels.includes('generation_request'));
        if (outputs.includes('response_trace')) {
            outputs.splice(outputs.indexOf('response_trace'), 1);
            outputs.unshift('response_trace');
        }
        options('output-channel', outputs, requestedOutput());
        options('session-output', profile.out_channels, profile.out_channels.includes('response_trace') ? 'response_trace' : '');
        options('session-input', profile.in_channels.filter(channel => profile.schemas?.[channel]?.json_input));
        if (profile.in_channels.includes('generation_request'))
            $('session-input').value = 'generation_request';
        $('custom-intervention').checked = false;
        if (mode !== 'protocol' && !$('output-channel').options.length && !profile.calibration)
            mode = 'session';
        sessionExample(false);
    }
    if (profile && !restoredForm) {
        restoredForm = true;
        restoreRequest(state.run?.request);
    }
    document.querySelectorAll('[data-mode]').forEach(button => {
        button.setAttribute('aria-selected', String(button.dataset.mode === mode));
        button.disabled = busy || (button.dataset.mode === 'calibration' && !profile?.calibration);
    });
    for (const item of ['protocol', 'benchmark', 'stimuli', 'session', 'calibration'])
        $(`${item}-editor`).hidden = item === 'stimuli' ? !['protocol', 'stimuli'].includes(mode) : item !== mode;
    const modelHost = $(mode === 'protocol' ? 'protocol-model-host' : 'legacy-model-host');
    if ($('model-loader').parentElement !== modelHost)
        modelHost.append($('model-loader'));
    const stimuliHost = $(mode === 'protocol' ? 'protocol-stimuli-host' : 'legacy-stimuli-host');
    if ($('stimuli-editor').parentElement !== stimuliHost)
        stimuliHost.append($('stimuli-editor'));
    $('protocol-nav').hidden = mode !== 'protocol';
    $('legacy-toolbox').hidden = mode === 'protocol';
    $('legacy-output-field').hidden = mode === 'protocol';
    $('stimulus-source-grid').classList.toggle('protocol-source', mode === 'protocol');
    $('calibration-settings').hidden = mode !== 'calibration';
    $('behavior-settings').hidden = requestedOutput() !== 'behavior' || mode !== 'stimuli';
    $('instruction-settings').hidden = mode !== 'stimuli' || !['behavior', 'response_trace'].includes(requestedOutput());
    const format = $('stimulus-format').value;
    $('video-audio-tools').hidden = !['csv', 'media'].includes(format);
    const registered = format === 'registered';
    $('registered-stimulus-editor').hidden = !registered;
    $('media-editor').hidden = format !== 'media';
    $('stimulus-actions').hidden = registered || format === 'media';
    $('stimulus-text-editor').hidden = registered || format === 'media' && !$('stimulus-text').value.trim();
    $('stimulus-text-label').textContent = format === 'media' ? 'Uploaded stimulus rows (CSV; editable)' : format === 'csv' ? 'Stimulus table (CSV)' : 'Stimuli';
    $('upload-status').hidden = registered || !$('upload-status').textContent;
    $('choose-media').disabled = busy;
    $('intervention-editor').hidden = mode === 'benchmark' || mode === 'calibration' || mode === 'protocol';
    const canIntervene = Boolean(profile?.out_channels.includes('perturbation') && profile.in_channels.some(channel => ['lesion', 'stimulation'].includes(channel)));
    $('custom-intervention').disabled = busy || !canIntervene;
    $('model-id').disabled = busy || state.phase === 'loading' || !catalogLoaded;
    $('model-search').disabled = busy || state.phase === 'loading' || !catalogLoaded;
    $('model-search-clear').disabled = busy || state.phase === 'loading' || !catalogLoaded;
    $('model-id').disabled ||= matchingModelCount === 0;
    $('load-model').disabled = busy || state.phase === 'loading' || !catalogLoaded || !$('model-id').value;
    $('load-calibration').disabled = busy || state.phase === 'loading';
    $('extract-existing-audio').disabled = busy || !['csv', 'media'].includes($('stimulus-format').value);
    document.querySelectorAll('.experiment-editor input,.experiment-editor select,.experiment-editor textarea').forEach(input => { if (!input.closest('#model-loader'))
        input.disabled = busy; });
    $('custom-intervention').disabled = busy || !canIntervene;
    const channel = $('session-input').value;
    $('channel-schema').textContent = profile?.schemas?.[channel]?.payload ?? 'JSON-native request channels are supported here. Typed environments and custom schemas use their benchmark or a Python harness.';
    $('workflow-help').textContent = mode === 'benchmark' ? 'The registered benchmark owns its experiment. Exploratory tools are disabled here; its score and provenance are saved.'
        : mode === 'session' ? 'Send JSON events through the subject’s declared request channels. Environments and non-JSON payloads require a typed harness.'
            : mode === 'stimuli' ? 'Choose response_trace for generated text, neural:<region> for activations, or behavior for labeled task responses. Generated text is kept verbatim.'
                : 'Calibration checks the tools. Load a registered subject to run your own science.';
    $('workflow-help').hidden = mode === 'protocol';
    $('editor-title').textContent = mode === 'protocol' ? 'Protocol' : mode === 'benchmark' ? 'Benchmark' : 'Experiment';
    if (mode === 'protocol')
        updateProtocol(profile, busy, state.validation);
}
export function handleTool(name, enabled) {
    if (mode === 'protocol') {
        protocolTool(name, enabled);
        return;
    }
    if (mode === 'calibration')
        return;
    if (name === 'activity' && enabled) {
        const channel = callbacks.state().profile?.out_channels.find(channel => channel.startsWith('neural:'));
        if (channel) {
            mode = 'stimuli';
            $('output-channel').value = channel;
        }
    }
    if (name === 'ablation')
        $('custom-intervention').checked = enabled;
}
export function buildRequest(record) {
    if (mode === 'protocol')
        return protocolRequest(stimulusInput());
    const intervention = customIntervention() ? {
        target: $('intervention-target').value.trim(),
        indices: $('intervention-indices').value.trim() ? $('intervention-indices').value.split(',').map(value => Number(value.trim())) : null,
        operation: $('intervention-operation').value,
        value: Number($('intervention-value').value),
    } : null;
    if (mode === 'benchmark')
        return { mode, benchmark: $('benchmark-id').value.trim() };
    if (mode === 'session')
        return { mode, output: $('session-output').value, events: JSON.parse($('session-events').value), record, intervention };
    const format = $('stimulus-format').value;
    if (format === 'media' && !$('stimulus-text').value.trim())
        throw new Error('Choose image, audio or video files first');
    return { mode, output: requestedOutput(), record, intervention,
        rows: format === 'registered' ? [] : ['csv', 'media'].includes(format) ? parseCSV($('stimulus-text').value)
            : $('stimulus-text').value.split('\n').filter(line => line.trim()).map(sentence => ({ sentence })),
        stimulus_set: format === 'registered' ? $('stimulus-set-id').value.trim() : '',
        instruction: $('instruction').value,
        labels: $('labels').value.split(',').map(label => label.trim()).filter(Boolean),
        prefer_path: $('behavior-path').value,
        fitting_rows: $('fitting-text').value.trim() ? parseCSV($('fitting-text').value) : [],
    };
}
function stimulusInput() {
    const format = $('stimulus-format').value;
    return { rows: format === 'registered' ? [] : ['csv', 'media'].includes(format)
            ? ($('stimulus-text').value.trim() ? parseCSV($('stimulus-text').value) : [])
            : $('stimulus-text').value.split('\n').filter(line => line.trim()).map(sentence => ({ sentence })),
        stimulus_set: format === 'registered' ? $('stimulus-set-id').value.trim() : '' };
}
export function initializeWorkbench(api) {
    callbacks = api;
    $('extract-existing-audio').onclick = async () => {
        uploading = true;
        callbacks.changed();
        try {
            const { rows } = stimulusInput();
            const videos = rows.filter(row => row.video_path && !row.audio_path);
            if (!videos.length)
                throw new Error('No video rows without audio are available. Upload a video or supply its uploaded asset in the table.');
            $('upload-status').hidden = false;
            for (const row of videos) {
                $('upload-status').textContent = `Extract audio: ${row.stimulus_id ?? row.video_path}`;
                await attachAudio(row);
            }
            $('stimulus-text').value = toCSV(rows);
            $('upload-status').textContent = `${videos.filter(row => row.audio_path).length} video rows now have an audio input`;
        }
        catch (problem) {
            callbacks.error(String(problem));
        }
        finally {
            uploading = false;
            callbacks.changed();
        }
    };
    $('allow-model-downloads').onchange = async () => {
        const allow = $('allow-model-downloads').checked;
        downloadPolicyPending = true;
        callbacks.changed();
        try {
            const result = await callbacks.post('/api/settings', { allow_model_downloads: allow });
            callbacks.state().allow_model_downloads = result.allow_model_downloads;
        }
        catch (problem) {
            callbacks.error(String(problem));
        }
        finally {
            downloadPolicyPending = false;
            callbacks.changed();
        }
    };
    document.querySelectorAll('[data-mode]').forEach(button => { button.onclick = () => chooseMode(button.dataset.mode); });
    const load = async (identifier) => {
        try {
            callbacks.error('');
            await callbacks.post('/api/model', { identifier });
            callbacks.loading();
        }
        catch (problem) {
            callbacks.error(String(problem));
        }
    };
    $('load-model').onclick = () => void load($('model-id').value);
    $('model-id').onchange = callbacks.changed;
    $('model-search').oninput = () => { filterModelChoices(); callbacks.changed(); };
    $('model-search-clear').onclick = () => {
        $('model-search').value = '';
        filterModelChoices();
        callbacks.changed();
        $('model-search').focus();
    };
    $('model-search').onkeydown = event => {
        if ((event.key === 'ArrowDown' || event.key === 'Enter') && matchingModelCount > 0) {
            event.preventDefault();
            $('model-id').focus();
        }
        else if (event.key === 'Escape') {
            $('model-search').value = '';
            filterModelChoices();
            callbacks.changed();
        }
    };
    $('load-calibration').onclick = () => { mode = 'calibration'; void load('__calibration__'); };
    document.querySelectorAll('.experiment-editor input,.experiment-editor select').forEach(input => { input.onchange = callbacks.changed; });
    $('stimulus-format').onchange = () => {
        stimulusDrafts[previousFormat] = $('stimulus-text').value;
        previousFormat = $('stimulus-format').value;
        $('stimulus-text').value = stimulusDrafts[previousFormat] ?? '';
        $('upload-status').textContent = '';
        callbacks.changed();
    };
    $('choose-media').onclick = () => {
        $('stimulus-format').value = 'media';
        $('stimulus-format').dispatchEvent(new Event('change'));
        $('media-files').focus();
    };
    $('session-input').onchange = () => sessionExample();
    $('stimulus-text').oninput = callbacks.changed;
    $('stimulus-set-id').oninput = callbacks.changed;
    $('csv-file').onchange = async (event) => {
        try {
            const file = event.target.files?.[0];
            if (!file)
                return;
            if (file.size > 1024 * 1024)
                throw new Error('CSV import limit is 1 MiB; use a registered stimulus set for larger datasets');
            const source = await file.text();
            const rows = parseCSV(source);
            stimulusDrafts[previousFormat] = $('stimulus-text').value;
            $('stimulus-text').value = source;
            $('stimulus-format').value = 'csv';
            previousFormat = 'csv';
            $('upload-status').textContent = `${rows.length} stimulus rows imported`;
            callbacks.changed();
        }
        catch (problem) {
            callbacks.error(String(problem));
        }
    };
    $('media-files').onchange = async (event) => {
        const files = [...(event.target.files ?? [])];
        if (!files.length)
            return;
        try {
            const rows = await uploadMediaFiles(files, name => { $('upload-status').textContent = `Uploading ${name}...`; });
            const headers = [...new Set(rows.flatMap(row => Object.keys(row)))];
            const quote = (text) => `"${text.replaceAll('"', '""')}"`;
            $('stimulus-text').value = [headers.join(','), ...rows.map(row => headers.map(header => quote(row[header] ?? '')).join(','))].join('\n');
            $('stimulus-format').value = 'media';
            previousFormat = 'media';
            $('upload-status').textContent = `${rows.length} files ready as stimulus rows`;
        }
        catch (problem) {
            callbacks.error(String(problem));
        }
        finally {
            uploading = false;
            callbacks.changed();
        }
    };
    initializeProtocol({ ...api, stimuli: stimulusInput, parseCSV, toCSV, uploadMediaFiles });
}
