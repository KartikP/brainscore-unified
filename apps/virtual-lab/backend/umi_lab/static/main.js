import { drawScene, drawAvatar } from './scene.js';
import { currentMode, initializeWorkbench, updateWorkbench, handleTool, buildRequest, customIntervention, requestedOutput, isUploading, isUpdatingSettings } from './workbench.js';
import { addResponsePlot, addCalibrationPlots, activationExtent, HeatmapScale } from './plots.js';
import { conditionNames, protocolSpec, protocolMeasurements, protocolHasIntervention, protocolRunReady, protocolStep, protocolDraftMatches } from './protocol.js';
import { renderProtocolResults } from './protocol_results.js';
import { renderProgress } from './progress.js';
import { replayView } from './replay.js';
const $ = (id) => document.getElementById(id);
const plan = { target: 'population', fraction: 25, activity: false, behavior: true, ablation: false, seed: 1729, repeats: 1 };
let state = { phase: 'loading', token: '', profile: null, run: null };
let profileID = '';
let hydrated = false;
let shownResult = '';
let frame = 0;
let requestPending = false;
let animation;
let lastScene = '';
let connectionLost = false;
let replay;
let replayID = '';
const activePhases = new Set(['queued', 'running', 'trial_finished', 'cancelling']);
const toolLabels = { behavior: 'Behavior log', activity: 'Activation probe', ablation: 'Ablation tool' };
const conditionLabels = { baseline: 'Baseline', ablation: 'Ablation', intervention: 'Intervention', random_control: 'Random control', restored: 'Restored', benchmark: 'Benchmark', loading_benchmark: 'Loading benchmark' };
const running = () => Boolean(state.run && activePhases.has(state.run.phase));
const busy = () => running() || requestPending || isUploading() || isUpdatingSettings() || state.phase === 'validating';
const currentProtocolResult = () => state.run?.report?.subject === state.profile?.identifier && protocolDraftMatches(state.run?.request);
function reviewTab(name) {
    const showReplay = name === 'replay' && !$('replay-tab').disabled;
    $('result-content').hidden = showReplay;
    $('replay-content').hidden = !showReplay;
    for (const [id, selected] of [['results-tab', !showReplay], ['replay-tab', showReplay]]) {
        $(id).setAttribute('aria-selected', String(selected));
        $(id).tabIndex = selected ? 0 : -1;
    }
    if (showReplay && !replay && replayID) {
        replay = replayView(replayID);
        $('replay-content').replaceChildren(replay.element);
    }
    if (!showReplay)
        replay?.pause();
}
$('results-tab').onclick = () => reviewTab('results');
$('replay-tab').onclick = () => reviewTab('replay');
document.querySelector('.review-tabs').addEventListener('keydown', event => {
    const key = event.key;
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(key))
        return;
    event.preventDefault();
    const target = key === 'Home' || key === 'ArrowLeft' ? 'results' : 'replay';
    if (target === 'replay' && $('replay-tab').disabled)
        return;
    reviewTab(target);
    $(`${target}-tab`).focus();
});
function scene() {
    const sceneState = { ...plan, running: running(), ready: state.phase === 'ready', condition: state.run?.condition ?? '', frame, seed: state.profile?.avatar_seed };
    const key = JSON.stringify(sceneState);
    if (key === lastScene)
        return;
    lastScene = key;
    drawScene($('lab-canvas'), sceneState);
}
function syncAnimation() {
    if (running() && !document.hidden && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
        if (animation === undefined)
            animation = window.setInterval(() => { frame++; scene(); }, 250);
    }
    else if (animation !== undefined) {
        clearInterval(animation);
        animation = undefined;
    }
}
document.addEventListener('visibilitychange', syncAnimation);
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
function renderTools() {
    const attached = $('attached-tools');
    attached.replaceChildren();
    for (const name of Object.keys(toolLabels)) {
        const button = document.querySelector(`[data-tool="${name}"]`);
        button.setAttribute('aria-pressed', String(plan[name]));
        button.querySelector('.tool-add').textContent = plan[name] ? '−' : '+';
        const unsupported = currentMode() === 'benchmark' || (name !== 'behavior' && state.profile !== null && (currentMode() === 'calibration' ? state.profile.targets.length === 0 : name === 'activity' ? !state.profile.out_channels.some(channel => channel.startsWith('neural:')) : !state.profile.in_channels.includes('lesion') || !state.profile.out_channels.includes('perturbation')));
        button.disabled = busy() || unsupported;
        button.title = unsupported ? 'This profile does not expose instrumentation targets' : `${plan[name] ? 'Remove' : 'Attach'} ${toolLabels[name]}`;
        if (!plan[name] || currentMode() === 'benchmark')
            continue;
        const chip = document.createElement('span');
        chip.className = `attached-chip ${name === 'activity' ? 'purple' : name === 'ablation' ? 'amber' : ''}`;
        chip.append(document.createTextNode(toolLabels[name]));
        const remove = document.createElement('button');
        remove.textContent = '×';
        remove.setAttribute('aria-label', `Remove ${toolLabels[name]}`);
        remove.disabled = busy();
        remove.onclick = () => toggleTool(name);
        chip.append(remove);
        attached.append(chip);
    }
    $('drop-hint').hidden = plan.activity && plan.behavior && plan.ablation;
    $('target').disabled = busy() || !state.profile?.targets.length;
    $('fraction').disabled = busy() || !plan.ablation;
    $('dose-value').textContent = `${plan.fraction}%`;
    const target = state.profile?.targets.find(target => target.name === plan.target);
    if (target) {
        const count = Math.max(1, Math.floor(target.units * plan.fraction / 100));
        $('selection-hint').textContent = `First ${count} of ${target.units} units; ${count * 2 <= target.units ? 'disjoint, size-matched random control.' : 'size-matched control may overlap.'}`;
    }
    renderConditions();
    scene();
}
function toggleTool(name) {
    if (busy())
        return;
    plan[name] = !plan[name];
    handleTool(name, plan[name]);
    renderState();
}
function renderConditions() {
    const names = currentMode() === 'protocol' ? conditionNames() : currentMode() === 'benchmark' ? ['benchmark'] : currentMode() !== 'calibration' ? customIntervention() ? ['baseline', 'intervention', 'restored'] : ['baseline'] : plan.ablation ? ['baseline', 'ablation', 'random_control', 'restored'] : ['baseline'];
    const container = $('conditions');
    container.replaceChildren();
    const current = names.indexOf(state.run?.condition ?? '');
    names.forEach((name, index) => {
        const div = document.createElement('div');
        div.className = 'condition';
        if (state.run?.phase === 'complete' && shownResult === state.run.id && (currentMode() === 'protocol' ? currentProtocolResult() : JSON.stringify(plan) === JSON.stringify(state.run.plan)))
            div.classList.add('done');
        else if (running() && index === current)
            div.classList.add('active');
        else if (running() && current > index)
            div.classList.add('done');
        div.append(document.createTextNode(conditionLabels[name] ?? name));
        const small = document.createElement('small');
        small.textContent = name === 'ablation' ? `${plan.fraction}% ZEROED` : name === 'random_control' ? 'SAME DOSE' : name === 'restored' ? 'CLEANUP CHECK' : 'UNCHANGED';
        if (currentMode() === 'protocol') {
            const change = protocolSpec().conditions[index]?.intervention;
            small.textContent = change ? `${change.operation} · ${change.indices?.length ?? 'all'} units` : 'Intact / no change';
        }
        div.append(small);
        container.append(div);
    });
}
function renderState() {
    updateWorkbench(state, busy());
    if (currentMode() !== 'calibration' && currentMode() !== 'protocol') {
        plan.ablation = customIntervention();
        plan.activity = currentMode() === 'stimuli' && requestedOutput().startsWith('neural:');
    }
    const profile = state.profile;
    if (profile && !hydrated) {
        if (state.run?.plan)
            Object.assign(plan, state.run.plan);
        if (typeof state.run?.request?.record === 'boolean')
            plan.behavior = state.run.request.record;
        $('fraction').value = String(plan.fraction);
        hydrated = true;
    }
    if (currentMode() === 'protocol') {
        const measurements = protocolMeasurements();
        plan.behavior = measurements.behavior;
        plan.activity = measurements.regions.length > 0;
        plan.ablation = protocolHasIntervention();
    }
    if (profile && profileID !== profile.identifier) {
        profileID = profile.identifier;
        shownResult = '';
        drawAvatar($('subject-avatar'), profile.avatar_seed ?? '0');
        $('subject-title').textContent = profile.title;
        $('subject-id').textContent = profile.identifier;
        $('subject-id').hidden = profile.title === profile.identifier;
        $('subject-description').textContent = profile.description;
        $('task-title').textContent = profile.task;
        $('subject-notice').textContent = profile.notice ?? '';
        $('subject-notice').hidden = !profile.notice;
        $('profile-note').textContent = profile.provenance.kind === 'engineered calibration' ? 'Engineered calibration circuit. Load a registered subject above to run benchmarks or supply your own stimuli.' : 'The avatar is generated from the subject identifier. Scientific abilities come from declared input and output channels.';
        const channels = $('channels');
        channels.replaceChildren();
        for (const channel of [...profile.in_channels.map(x => `in: ${x}`), ...profile.out_channels.map(x => `out: ${x}`)]) {
            const tag = document.createElement('span');
            tag.className = 'channel';
            tag.textContent = channel;
            channels.append(tag);
        }
        const select = $('target');
        select.replaceChildren();
        profile.targets.forEach(target => { const option = document.createElement('option'); option.value = target.name; option.textContent = `${target.name} / ${target.units} units`; select.append(option); });
        if (!profile.targets.some(target => target.name === plan.target))
            plan.target = profile.targets[0]?.name ?? 'none';
        select.value = plan.target;
    }
    const run = state.run;
    $('replay-tab').disabled = run?.phase !== 'complete';
    if (run?.phase !== 'complete')
        reviewTab('results');
    renderProgress(run, state.phase, !connectionLost);
    if (currentMode() !== 'calibration')
        $('task-title').textContent = currentMode() === 'protocol' ? protocolSpec().identifier || 'New protocol' : currentMode() === 'benchmark' ? 'Registered benchmark' : currentMode() === 'session' ? 'Declared event session' : `Custom stimuli / ${requestedOutput() || 'choose an output'}`;
    const status = $('run-status');
    status.replaceChildren(document.createElement('i'));
    const label = state.phase === 'failed' || state.phase === 'load_failed' ? 'Unavailable' : state.phase === 'loading' ? 'Loading subject' : state.phase === 'validating' ? 'Validating draft' : running() ? run?.phase === 'cancelling' ? 'Stopping' : 'Experiment running' : currentMode() === 'protocol' && !currentProtocolResult() ? 'Protocol draft' : run?.phase === 'complete' ? 'Run complete' : run?.phase === 'failed' ? 'Run failed' : 'Ready to experiment';
    status.append(document.createTextNode(label));
    status.classList.toggle('running', running());
    $('subject-status').textContent = state.phase === 'ready' ? 'Loaded' : state.phase.replaceAll('_', ' ');
    $('scene-state').textContent = running() ? conditionLabels[run?.condition ?? ''] ?? 'Preparing run' : state.phase === 'ready' ? 'Subject ready' : 'Awaiting subject';
    const blocked = state.phase !== 'ready' || busy() || currentMode() === 'protocol' && !protocolRunReady();
    $('run').disabled = blocked;
    $('run-editor').disabled = blocked;
    $('run-editor').hidden = currentMode() === 'protocol' && protocolStep() !== 5;
    $('run').textContent = $('run-editor').textContent = currentMode() === 'protocol' ? 'Run protocol' : 'Run experiment';
    $('stop').hidden = !running();
    $('stop').disabled = run?.phase === 'cancelling';
    $('stop').textContent = currentMode() === 'protocol' ? 'Stop after repetition' : currentMode() === 'benchmark' ? 'Stop after benchmark' : 'Stop after trial';
    $('run-help').textContent = running() ? 'Stopping waits for the current repetition or whole benchmark and its cleanup.' : currentMode() === 'protocol' ? 'Each stimulus is shown once per condition and repetition. Validate the complete setup in step 6 before running.' : currentMode() === 'benchmark' ? 'Runs the existing benchmark and its pre-flight checks.' : currentMode() !== 'calibration' ? 'Executes your inputs through the subject session interface.' : plan.ablation ? 'Four conditions. Same stimuli. Independent trials, with automatic cleanup.' : 'Start with a baseline, or attach an ablation tool to compare conditions.';
    if (state.error)
        error(state.error);
    else if (run?.phase === 'failed' || run?.phase === 'cancelled') {
        error(run.error || 'Experiment stopped; the plugin supplied no additional message');
        $('result-time').textContent = run.phase === 'failed' ? 'FAILED' : 'STOPPED';
    }
    if (run?.phase === 'complete' && shownResult !== run.id)
        renderResults(run);
    renderTools();
    syncAnimation();
}
function renderResults(run) {
    shownResult = run.id;
    replay?.dispose();
    replay = undefined;
    replayID = run.id;
    $('replay-content').replaceChildren();
    reviewTab('results');
    const content = $('result-content');
    content.replaceChildren();
    const report = run.report;
    if (report?.kind === 'protocol') {
        renderProtocolResults(report, content, run.id, (observations, target, scale) => renderWorkbenchReport(observations, target, run, scale));
    }
    else if (report?.conditions) {
        const grid = document.createElement('div');
        grid.className = 'metric-grid';
        for (const row of report.conditions) {
            const card = document.createElement('div');
            card.className = `metric-card ${row.condition}`;
            const label = document.createElement('span');
            label.className = 'metric-label';
            label.textContent = conditionLabels[row.condition] ?? row.condition;
            const value = document.createElement('strong');
            value.textContent = `${(row.accuracy * 100).toFixed(0)}%`;
            const bar = document.createElement('div');
            bar.className = 'metric-bar';
            const visual = document.createElement('canvas');
            visual.width = 200;
            visual.height = 4;
            const ctx = visual.getContext('2d');
            ctx.fillStyle = row.condition === 'ablation' ? '#c7ad7d' : '#9cb987';
            ctx.fillRect(0, 0, 200 * row.accuracy, 4);
            bar.append(visual);
            const detail = document.createElement('small');
            detail.textContent = `${row.stimuli} STIMULI / ACCURACY`;
            card.append(label, value, bar, detail);
            grid.append(card);
        }
        content.append(grid);
        addCalibrationPlots(report, content);
    }
    else if (report?.kind) {
        renderWorkbenchReport(report, content, run);
    }
    else {
        const summary = document.createElement('pre');
        summary.textContent = JSON.stringify(report, null, 2);
        content.append(summary);
    }
    if (report?.restored_exact !== null && report?.restored_exact !== undefined) {
        const restored = document.createElement('p');
        restored.className = `restoration ${report.restored_exact ? '' : 'failed'}`;
        restored.textContent = report.restored_exact ? 'Restored: baseline and restored outputs match exactly.' : 'Check required: restored outputs differ from baseline.';
        content.append(restored);
    }
    if (run.selection?.overlap) {
        const warning = document.createElement('p');
        warning.className = 'result-note';
        warning.textContent = `The random control overlaps the selected units at this dose (${run.selection.overlap} units).`;
        content.append(warning);
    }
    if (report?.note) {
        const note = document.createElement('p');
        note.className = 'result-note';
        note.textContent = report.note;
        content.append(note);
    }
    $('result-time').textContent = `${run.elapsed_s?.toFixed(2)} s`;
    $('downloads').hidden = false;
    $('download-python').href = `/api/runs/${run.id}/experiment.py`;
    $('download-plan').href = `/api/runs/${run.id}/plan.json`;
    $('download-report').href = `/api/runs/${run.id}/summary.json`;
    $('download-artifacts').href = `/api/runs/${run.id}/artifacts.zip`;
}
function renderWorkbenchReport(report, container, run, sharedScale) {
    const title = document.createElement('p');
    title.className = 'response-meta';
    title.textContent = `${String(report.subject ?? '')} / ${String(report.benchmark ?? report.output ?? '')}`;
    container.append(title);
    if (report.kind === 'benchmark') {
        const value = document.createElement('div');
        value.className = 'benchmark-number';
        const score = report.score;
        value.textContent = report.status === 'N/A' ? 'N/A' : typeof score?.sample?.[0] === 'number' ? score.sample[0].toFixed(6) : 'Score saved';
        container.append(value);
        if (report.reason) {
            const reason = document.createElement('p');
            reason.className = 'editor-note';
            reason.textContent = String(report.reason);
            container.append(reason);
        }
        if (report.score) {
            const detail = document.createElement('pre');
            detail.textContent = JSON.stringify(report.score, null, 2);
            container.append(detail);
        }
        return;
    }
    const outputs = report.outputs;
    const extent = activationExtent((outputs ?? []).flatMap(output => output.response.activation_preview ? [output.response.activation_preview] : []));
    const scale = sharedScale ?? new HeatmapScale();
    for (const [index, output] of (outputs ?? []).entries()) {
        const card = document.createElement('div');
        card.className = 'response-card';
        const heading = document.createElement('h3');
        heading.textContent = `${conditionLabels[output.condition] ?? output.condition}${output.channel ? ` / ${output.channel}` : ''}${output.repetition !== undefined ? ` / repetition ${output.repetition + 1}` : ''}`;
        card.append(heading);
        const response = output.response;
        if (Array.isArray(response.shape)) {
            const meta = document.createElement('p');
            meta.className = 'response-meta';
            meta.textContent = `${String(response.type)} / shape [${response.shape.join(', ')}] / ${response.dims.join(' × ')}`;
            card.append(meta);
            if (response.measure) {
                const measure = document.createElement('p');
                measure.className = 'response-meta';
                const details = response.measure;
                measure.textContent = typeof response.measure === 'string' ? response.measure : `${details.kind ?? 'Recorded'} measure${details.source ? ` / ${details.source}` : ''}`;
                card.append(measure);
            }
            if (response.recording && typeof response.recording === 'object') {
                const recording = response.recording;
                const note = document.createElement('p');
                note.className = 'response-meta';
                note.textContent = `Recording: ${[recording.task_type, recording.path, recording.forward_pass, recording.readout_fit].filter(value => value !== undefined).join(' / ')}`;
                card.append(note);
            }
        }
        const rows = response.responses;
        if (rows?.length) {
            const table = document.createElement('table');
            table.className = 'response-table';
            const header = document.createElement('tr');
            const labels = response.choice_labels;
            for (const label of ['Stimulus', 'Model preference', `Choice values${labels?.length ? ` (${labels.join(', ')})` : ''}`]) {
                const th = document.createElement('th');
                th.textContent = label;
                header.append(th);
            }
            table.append(header);
            for (const row of rows) {
                const preference = row.selection_status === 'tie' ? `No preference (${row.tied_choices?.join(' / ')})` : row.selection_status === 'invalid' ? 'Invalid choice values' : row.answer ?? 'No preference';
                const tr = document.createElement('tr');
                const stimulus = row.stimulus_text ? `${row.stimulus}: ${row.stimulus_text}` : row.stimulus;
                for (const text of [stimulus, preference, row.scores.map(value => value === null ? 'nonfinite' : value.toFixed(4)).join(', ')]) {
                    const td = document.createElement('td');
                    td.textContent = text;
                    tr.append(td);
                }
                table.append(tr);
            }
            card.append(table);
        }
        else if (Array.isArray(response.scalar_rows)) {
            const table = document.createElement('table');
            table.className = 'response-table';
            for (const row of response.scalar_rows) {
                const tr = document.createElement('tr');
                for (const text of [row.stimulus, row.value === null ? 'Missing / invalid' : String(row.value)]) {
                    const td = document.createElement('td');
                    td.textContent = text;
                    tr.append(td);
                }
                table.append(tr);
            }
            card.append(table);
        }
        else if (Array.isArray(response.sample)) {
            const values = document.createElement('div');
            values.className = 'sample-values';
            for (const item of response.sample) {
                const value = document.createElement('span');
                value.textContent = typeof item === 'number' ? item.toPrecision(4) : String(item);
                values.append(value);
            }
            card.append(values);
            const note = document.createElement('p');
            note.className = 'subtle';
            note.textContent = 'First 16 values shown. The complete labeled assembly is saved in the artifacts.';
            card.append(note);
        }
        else if (Array.isArray(response.preview) && response.preview.some(item => typeof item?.raw_answer === 'string')) {
            const table = document.createElement('table');
            table.className = 'response-table';
            const header = document.createElement('tr');
            for (const title of ['Stimulus', 'Generated answer (verbatim)', 'Record status']) {
                const th = document.createElement('th');
                th.textContent = title;
                header.append(th);
            }
            table.append(header);
            for (const item of response.preview) {
                const tr = document.createElement('tr');
                for (const text of [String(item.stimulus_text ?? item.stimulus ?? ''), String(item.raw_answer ?? ''), item.valid === false ? `Parse error: ${item.parse_error}` : 'Recorded']) {
                    const td = document.createElement('td');
                    td.textContent = text;
                    tr.append(td);
                }
                table.append(tr);
            }
            card.append(table);
        }
        else {
            const detail = document.createElement('pre');
            detail.textContent = JSON.stringify(response, null, 2);
            card.append(detail);
        }
        const repetitions = Number(run.request?.repeats ?? 1);
        addResponsePlot(card, response, extent, `/api/runs/${run.id}/activation?index=${output.result_index ?? index * repetitions}${output.channel ? `&channel=${encodeURIComponent(output.channel)}` : ''}`, scale);
        container.append(card);
    }
}
async function post(path, data) {
    const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Lab-Token': state.token }, body: JSON.stringify(data) });
    const result = await response.json();
    if (!response.ok)
        throw new Error(result.error);
    return result;
}
async function poll() {
    try {
        const response = await fetch('/api/state');
        if (!response.ok)
            throw new Error('The local lab server is unavailable');
        state = await response.json();
        if (connectionLost)
            error('');
        connectionLost = false;
        renderState();
    }
    catch (problem) {
        connectionLost = true;
        renderProgress(state.run, state.phase, false);
        error(`Connection lost. ${String(problem)}`);
    }
    window.setTimeout(poll, document.hidden ? 5000 : running() || state.phase === 'loading' ? 500 : 2000);
}
$('run').onclick = async () => {
    if (busy())
        return;
    if (currentMode() === 'protocol' && !protocolRunReady()) {
        error('Review and validate this protocol in step 6 before running.');
        return;
    }
    error('');
    requestPending = true;
    renderState();
    try {
        const calibration = currentMode() === 'calibration';
        const response = await post(calibration ? '/api/runs' : '/api/experiments', calibration ? plan : buildRequest(plan.behavior));
        state.run = { id: response.id, phase: 'queued', ...(calibration ? { plan: { ...plan } } : {}) };
        error('');
        $('downloads').hidden = true;
        $('result-time').textContent = 'RUNNING';
    }
    catch (problem) {
        error(String(problem));
    }
    finally {
        requestPending = false;
        renderState();
    }
};
$('stop').onclick = async () => {
    try {
        await post('/api/cancel', {});
        if (state.run)
            state.run.phase = 'cancelling';
        renderState();
    }
    catch (problem) {
        error(String(problem));
    }
};
$('run-editor').onclick = () => $('run').click();
$('fraction').oninput = event => { plan.fraction = Number(event.target.value); renderTools(); renderState(); };
$('target').onchange = event => { plan.target = event.target.value; renderTools(); };
document.querySelectorAll('[data-tool]').forEach(button => {
    const name = button.dataset.tool;
    button.onclick = () => toggleTool(name);
    button.ondragstart = event => { if (busy()) {
        event.preventDefault();
        return;
    } event.dataTransfer?.setData('application/x-umi-tool', name); };
});
$('bench').ondragover = event => { if (!busy() && event.dataTransfer?.types.includes('application/x-umi-tool')) {
    event.preventDefault();
    $('bench').classList.add('dragover');
} };
$('bench').ondragleave = event => { if (!$('bench').contains(event.relatedTarget))
    $('bench').classList.remove('dragover'); };
$('bench').ondrop = event => {
    event.preventDefault();
    $('bench').classList.remove('dragover');
    if (busy())
        return;
    const name = event.dataTransfer?.getData('application/x-umi-tool');
    if (Object.hasOwn(toolLabels, name) && !document.querySelector(`[data-tool="${name}"]`)?.disabled) {
        plan[name] = true;
        handleTool(name, true);
        renderState();
    }
};
initializeWorkbench({ state: () => state, post, changed: renderState, error,
    loading: () => {
        replay?.dispose();
        replay = undefined;
        replayID = '';
        $('replay-content').replaceChildren();
        reviewTab('results');
        state.phase = 'loading';
        state.error = undefined;
        state.run = null;
        shownResult = '';
        error('');
        const note = document.createElement('p');
        note.className = 'editor-note';
        note.textContent = 'Run an experiment with the loaded subject to inspect its responses.';
        $('result-content').replaceChildren(note);
        $('downloads').hidden = true;
        $('result-time').textContent = 'NO RUN YET';
        renderState();
    } });
renderState();
void poll();
