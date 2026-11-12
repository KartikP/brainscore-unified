const $ = (id) => document.getElementById(id);
const value = (id) => $(id).value.trim();
const checked = (id) => $(id).checked;
const numeric = (id) => value(id) === '' ? NaN : Number(value(id));
const list = (source) => source.split(',').map(part => part.trim()).filter(Boolean);
const labels = ['Participant', 'Stimuli', 'Task', 'Recordings', 'Session', 'Review'];
let api;
let active = 0;
let profile = null;
let profileID = '';
let validation;
let pendingID = '', pendingKey = '', validatedKey = '';
let busy = false, reviewKey = '';
export function responseChannels(task) {
    if (task.kind === 'passive')
        return [];
    if (task.kind === 'legacy')
        return ['behavior'];
    if (task.kind === 'rating')
        return ['behavior:rating'];
    if (task.kind === 'text')
        return ['behavior:text'];
    return ['behavior:choice', ...(task.probabilities ? ['behavior:probabilities'] : []), ...(task.confidence ? ['behavior:confidence'] : [])];
}
export function requestedRecordings(spec) {
    return [...spec.measurements.regions.map(region => `neural:${region}`), ...(spec.measurements.behavior ? responseChannels(spec.task) : [])];
}
function conditionCard(condition) {
    const card = document.createElement('fieldset');
    card.className = 'protocol-condition';
    card.innerHTML = '<legend>Condition</legend>'
        + '<div class="condition-fields"><label>Name<input data-condition="name"></label><label>Change to the participant<select data-condition="operation"><option value="none">None (intact)</option><option value="zero">Ablate / zero</option><option value="scale">Scale activity</option><option value="drive">Drive / add activity</option></select></label><button type="button" class="secondary" data-remove-condition>Remove</button></div>'
        + '<p class="subtle field-hint" data-condition-hint></p>'
        + '<div data-condition-settings>'
        + '<label>Where: target address<input data-condition="target" list="protocol-targets" placeholder="e.g. model.layers.14.mlp or features.12"></label>'
        + '<p class="subtle field-hint">The part of the model to change, written as its module path. Add <code>:input</code> to change what goes into the module. Suggestions appear when the participant declares its targets.</p>'
        + '<div class="editor-grid"><label>Which units (blank = all)<input data-condition="indices" placeholder="e.g. 0,1,2"></label><label>What the unit numbers count<select data-condition="axis"><option value="-1">Last axis (linear and transformer units)</option><option value="1">Axis 1 (convolution channels)</option><option value="null">All units, flattened</option></select></label></div>'
        + '<p class="subtle field-hint">Unit numbers start at 0 and are separated by commas. Leave the field blank to change every unit at the target.</p>'
        + '<div data-condition-strength><label>How much<input data-condition="value" type="number" step="any" value="0"></label><p class="subtle field-hint" data-condition-strength-hint></p></div>'
        + '<p class="subtle field-hint">Tip: add one more condition that changes the same number of different units. It is the control for this one.</p>'
        + '</div>';
    const field = (name) => card.querySelector(`[data-condition="${name}"]`);
    const change = condition.intervention;
    field('name').value = condition.name;
    field('operation').value = change?.operation ?? 'none';
    field('target').value = change?.target ?? '';
    field('indices').value = change?.indices?.join(',') ?? '';
    field('axis').value = change?.axis === null ? 'null' : String(change?.axis ?? -1);
    field('value').value = String(change?.value ?? 0);
    const sync = () => {
        card.querySelector('[data-condition-settings]').hidden = field('operation').value === 'none';
        card.querySelector('[data-condition-strength]').hidden = field('operation').value === 'zero';
        const operation = field('operation').value;
        card.querySelector('[data-condition-hint]').textContent = operation === 'none' ? 'The participant is not changed in this condition. Use it as the baseline.'
            : operation === 'zero' ? 'Ablate: the chosen units are set to zero during this condition. The participant is restored after it.'
                : operation === 'scale' ? 'Scale: the activity of the chosen units is multiplied by a factor during this condition.'
                    : 'Drive: an amount is added to the activity of the chosen units during this condition.';
        card.querySelector('[data-condition-strength-hint]').textContent = operation === 'scale' ? 'Factor: 0 silences the units, 0.5 halves their activity, 2 doubles it.' : 'Amount added to each chosen unit. Start small, for example 1.';
    };
    card.querySelector('[data-remove-condition]').onclick = () => { card.remove(); api.changed(); };
    card.querySelectorAll('input,select').forEach(input => {
        input.oninput = () => { sync(); api.changed(); };
    });
    sync();
    return card;
}
export function conditionNames() {
    return [...$('protocol-conditions').querySelectorAll('[data-condition="name"]')].map(input => input.value.trim() || 'Unnamed condition');
}
export function protocolSpec() {
    const kind = value('protocol-task');
    const task = { kind, instruction: value('protocol-instruction'), labels: list(value('protocol-labels')),
        probabilities: kind === 'choice' && checked('protocol-probabilities'), path: value('protocol-path'),
        readout_fit: value('protocol-fit-policy'), present: list(value('protocol-present')), trial_memory: value('protocol-trial-memory') };
    if (kind === 'rating')
        task.scale = { low: numeric('protocol-scale-low'), high: numeric('protocol-scale-high'), step: value('protocol-scale-step') === '' ? null : numeric('protocol-scale-step') };
    if (kind === 'choice' && checked('protocol-confidence'))
        task.confidence = { low: numeric('protocol-confidence-low'), high: numeric('protocol-confidence-high'), step: 1 };
    const conditions = [...$('protocol-conditions').querySelectorAll('.protocol-condition')].map(card => {
        const field = (name) => card.querySelector(`[data-condition="${name}"]`).value.trim();
        return { name: field('name'), intervention: field('operation') === 'none' ? null : {
                target: field('target'), indices: field('indices') ? list(field('indices')).map(Number) : null,
                axis: field('axis') === 'null' ? null : Number(field('axis')), operation: field('operation'), value: Number(field('value')),
            } };
    });
    return { identifier: value('protocol-name'), task, measurements: { behavior: checked('protocol-record-behavior'), replay: checked('protocol-record-replay'),
            regions: [...$('protocol-regions').querySelectorAll('input:checked')].map(input => input.value) },
        conditions, repetitions: numeric('protocol-repetitions'), required_traits: [] };
}
export function protocolRequest(stimuli = api.stimuli()) {
    return { mode: 'protocol', ...stimuli, fitting_rows: value('protocol-fitting') ? api.parseCSV(value('protocol-fitting')) : [], protocol: protocolSpec() };
}
export function protocolDraftMatches(request) {
    if (request?.mode !== 'protocol')
        return false;
    try {
        const draft = protocolRequest();
        return ['protocol', 'rows', 'fitting_rows', 'stimulus_set'].every(field => JSON.stringify(draft[field]) === JSON.stringify(request[field]));
    }
    catch {
        return false;
    }
}
function key() {
    return JSON.stringify({ subject: profile?.identifier, request: protocolRequest() });
}
function issue(step) {
    let spec;
    try {
        spec = protocolSpec();
    }
    catch (error) {
        return String(error);
    }
    if (step === 0)
        return !spec.identifier ? 'Give the protocol a name.' : profile ? '' : 'Load a participant to see its declared inputs and outputs.';
    if (step === 1) {
        try {
            const data = api.stimuli();
            return data.stimulus_set || data.rows.length ? '' : 'Add at least one stimulus, or choose a registered stimulus set.';
        }
        catch (error) {
            return String(error);
        }
    }
    if (step === 2) {
        const task = spec.task;
        if (['choice', 'legacy'].includes(task.kind) && (task.labels.length < 2 || new Set(task.labels.map(label => label.toLowerCase())).size !== task.labels.length))
            return 'Provide at least two distinct response labels.';
        for (const scale of [task.scale, task.confidence])
            if (scale && (!Number.isFinite(scale.low) || !Number.isFinite(scale.high) || scale.low >= scale.high || (scale.step !== null && (!Number.isFinite(scale.step) || scale.step <= 0))))
                return 'Response scales need a lower bound below the upper bound and a positive step.';
        const missing = responseChannels(task).filter(channel => !profile?.out_channels.includes(channel));
        if (missing.length)
            return `This participant does not declare ${missing.join(', ')}. Choose a compatible response or another participant.`;
        if (task.kind !== 'passive' && task.path === 'readout') {
            try {
                const fitting = value('protocol-fitting') ? api.parseCSV(value('protocol-fitting')) : [];
                if (!fitting.length || !('label' in fitting[0]))
                    return 'A readout needs a separate fitting CSV with a label column.';
                if (fitting.some(row => !row.label?.trim()))
                    return 'Fill the label for every fitting example.';
            }
            catch (error) {
                return String(error);
            }
        }
        try {
            const rows = api.stimuli().rows;
            const columns = new Set(rows.flatMap(row => Object.keys(row)));
            const parts = [columns.has('sentence'), columns.has('image_path'), columns.has('audio_path'), columns.has('video_path')].filter(Boolean).length;
            if (task.kind !== 'passive' && parts > 1 && !task.present.length)
                return 'Specify the trial part order for multimodal stimuli, for example text,image.';
        }
        catch { /* The stimuli step owns input-format errors. */ }
    }
    if (step === 3) {
        const channels = requestedRecordings(spec);
        if (!channels.length)
            return 'Select behavior or at least one neural recording.';
        if (spec.task.kind === 'passive' && spec.measurements.behavior)
            return 'Plain viewing has no behavioral answer. Choose a response, or record neural activity only.';
        const missing = channels.filter(channel => !profile?.out_channels.includes(channel));
        if (missing.length)
            return `The participant does not declare ${missing.join(', ')}.`;
    }
    if (step === 4) {
        if (!spec.identifier)
            return 'Give the protocol a name.';
        if (!spec.conditions.length || spec.conditions.length > 12)
            return 'Provide between one and twelve conditions.';
        if (spec.conditions.some(condition => !condition.name) || new Set(spec.conditions.map(condition => condition.name)).size !== spec.conditions.length)
            return 'Give each condition a distinct, nonempty name.';
        if (!Number.isInteger(spec.repetitions) || spec.repetitions < 1 || spec.repetitions > 100)
            return 'Choose 1–100 independent repetitions.';
        if (spec.task.trial_memory === 'history' && !profile?.traits?.includes('memory:trial_history'))
            return 'This participant does not keep earlier trials, so the trials cannot run with history. Choose that each trial starts fresh.';
        for (const condition of spec.conditions) {
            const change = condition.intervention;
            if (!change)
                continue;
            const channel = change.operation === 'zero' ? 'lesion' : 'stimulation';
            if (!profile?.in_channels.includes(channel) || !profile.out_channels.includes('perturbation'))
                return `This participant does not declare ${channel} and perturbation channels.`;
            if (!change.target || !Number.isFinite(change.value) || change.indices?.some(index => !Number.isInteger(index) || index < 0))
                return `Complete the intervention address, unit indices and strength for ${condition.name}.`;
        }
    }
    return '';
}
function firstIssue() {
    for (let step = 0; step < 5; step++) {
        const message = issue(step);
        if (message)
            return { step, message };
    }
    return null;
}
let sitesKey = '';
/** Recording sites and lesions drawn on the participant. A site toggles its recording directly. */
function renderSites() {
    const inputs = [...$('protocol-regions').querySelectorAll('input')];
    let lesions = [];
    try {
        lesions = protocolSpec().conditions.flatMap(condition => condition.intervention ? [`${condition.intervention.operation === 'zero' ? 'lesion' : condition.intervention.operation} at ${condition.intervention.target || 'no target'} (${condition.name})`] : []);
    }
    catch { /* shown by the session part */ }
    const key = JSON.stringify([inputs.map(input => [input.value, input.checked]), lesions, busy]);
    if (key === sitesKey)
        return; // keep focus: rebuild only when something changed
    sitesKey = key;
    const container = $('rig-sites');
    container.replaceChildren();
    for (const input of inputs) {
        const site = document.createElement('button');
        site.type = 'button';
        site.className = 'rig-site';
        site.textContent = input.value;
        site.setAttribute('aria-pressed', String(input.checked));
        site.disabled = busy;
        site.title = `${input.checked ? 'Stop recording' : 'Record'} ${input.value}`;
        site.onclick = () => { input.checked = !input.checked; api.changed(); };
        container.append(site);
    }
    for (const text of lesions) {
        const mark = document.createElement('span');
        mark.className = 'rig-lesion';
        mark.textContent = text;
        container.append(mark);
    }
}
/** One short line per part, shown on the rig. */
function stepSummaries() {
    let spec;
    try {
        spec = protocolSpec();
    }
    catch {
        return Array(6).fill('');
    }
    let stimuli = 'Not ready';
    try {
        const source = api.stimuli();
        stimuli = source.stimulus_set || `${source.rows.length} ${source.rows.length === 1 ? 'trial' : 'trials'}`;
    }
    catch { /* an incomplete table is reported by the step's own check */ }
    const task = spec.task;
    const taskText = task.kind === 'passive' ? 'Plain viewing' : task.kind === 'choice' ? `Choice: ${task.labels.join(' / ') || 'no labels'}`
        : task.kind === 'rating' ? `Rating ${task.scale?.low ?? ''} to ${task.scale?.high ?? ''}` : task.kind === 'text' ? 'Written answer' : 'Labeled behavior';
    const recorded = [...spec.measurements.regions, ...(spec.measurements.behavior ? ['answers'] : []), ...(spec.measurements.replay ? ['replay log'] : [])];
    return [profile?.identifier ?? 'No participant', stimuli, taskText, recorded.join(', ') || 'Nothing recorded',
        `${spec.conditions.map(condition => condition.name).join(', ')} \u00d7 ${spec.repetitions}`, protocolRunReady() ? 'Checked' : 'Not checked'];
}
/** The first thing that still needs attention, with the part it belongs to. Empty when the setup is complete. */
export function protocolIssue() {
    const problem = firstIssue();
    return problem ? `${labels[problem.step]}: ${problem.message}` : '';
}
export function protocolValidationFailed() {
    return validation?.id === pendingID && validation.phase === 'invalid' ? validation.error ?? 'Validation failed' : '';
}
export async function validateProtocol() {
    pendingKey = key();
    validatedKey = '';
    const response = await api.post('/api/validate', protocolRequest());
    pendingID = response.id;
    api.changed();
}
/** Back to an empty protocol. The loaded participant is kept. */
export function resetProtocol() {
    restoreProtocol({});
    for (const id of ['protocol-instruction', 'protocol-fitting', 'protocol-present'])
        $(id).value = '';
    active = 0;
    pendingID = '';
    pendingKey = '';
    validatedKey = '';
    reviewKey = '';
}
export function protocolRunReady() {
    try {
        return !firstIssue() && validation?.id === pendingID && validation.phase === 'valid' && validatedKey === key();
    }
    catch {
        return false;
    }
}
export const protocolStep = () => active;
export const protocolMeasurements = () => protocolSpec().measurements;
export const protocolHasIntervention = () => protocolSpec().conditions.some(condition => Boolean(condition.intervention));
export function protocolTool(name, enabled) {
    if (name === 'behavior')
        $('protocol-record-behavior').checked = enabled;
    if (name === 'activity') {
        const boxes = [...$('protocol-regions').querySelectorAll('input')];
        if (!enabled)
            boxes.forEach(box => { box.checked = false; });
        else if (!boxes.some(box => box.checked) && boxes[0])
            boxes[0].checked = true;
    }
    if (name === 'ablation' && enabled && !protocolHasIntervention()) {
        $('protocol-conditions').append(conditionCard({ name: 'lesioned', intervention: { target: '', operation: 'zero', value: 0, indices: null, axis: -1 } }));
        active = 4;
    }
    if (name === 'ablation' && !enabled) {
        $('protocol-conditions').querySelectorAll('.protocol-condition').forEach(card => {
            if (card.querySelector('[data-condition="operation"]').value !== 'none')
                card.remove();
        });
    }
}
function renderRegions(previous) {
    const declared = profile?.out_channels.filter(channel => channel.startsWith('neural:')).map(channel => channel.slice(7)) ?? [];
    const regionList = $('protocol-regions');
    regionList.replaceChildren();
    for (const region of [...new Set([...declared, ...previous])]) {
        const label = document.createElement('label');
        label.className = 'check-label';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.value = region;
        input.checked = previous.includes(region);
        input.onchange = api.changed;
        label.append(input, document.createTextNode(region + (declared.includes(region) ? '' : ' (not declared by this subject)')));
        regionList.append(label);
    }
    if (!declared.length && !previous.length) {
        const note = document.createElement('p');
        note.className = 'subtle';
        note.textContent = 'This participant declares no neural recording outputs.';
        regionList.append(note);
    }
}
function showReview() {
    let fingerprint;
    try {
        fingerprint = key();
    }
    catch {
        fingerprint = 'invalid';
    }
    const stateKey = fingerprint + JSON.stringify(validation);
    if (reviewKey === stateKey)
        return;
    reviewKey = stateKey;
    const target = $('protocol-review');
    target.replaceChildren();
    const problem = firstIssue();
    if (problem) {
        const note = document.createElement('p');
        note.textContent = `${labels[problem.step]}: ${problem.message}`;
        target.append(note);
        return;
    }
    const spec = protocolSpec(), stimuli = api.stimuli();
    const rows = [['Protocol', spec.identifier], ['Participant', profile?.identifier ?? ''],
        ['Stimuli', stimuli.stimulus_set || `${stimuli.rows.length} rows`], ['Task', spec.task.kind === 'passive' ? 'Plain viewing' : `${spec.task.kind} · ${spec.task.path}`],
        ['Response', spec.task.kind === 'choice' || spec.task.kind === 'legacy' ? spec.task.labels.join(', ') : spec.task.kind],
        ['Readout fitting', spec.task.readout_fit], ['Memory across trials', spec.task.trial_memory === 'history' ? 'History: the subject keeps what it took in and answered on earlier trials' : 'None: each trial starts fresh'], ['Recordings', requestedRecordings(spec).join(', ')],
        ['Conditions', spec.conditions.map(condition => condition.name).join(', ')], ['Repetitions', String(spec.repetitions)],
        ['Trial schedule', `${spec.conditions.length} conditions × ${spec.repetitions} repetitions × ${stimuli.stimulus_set ? 'registered stimuli' : `${stimuli.rows.length} stimuli`}`]];
    const table = document.createElement('dl');
    table.className = 'protocol-summary';
    for (const [label, text] of rows) {
        const term = document.createElement('dt'), description = document.createElement('dd');
        term.textContent = label;
        description.textContent = text;
        table.append(term, description);
    }
    target.append(table);
    const details = document.createElement('details'), summary = document.createElement('summary'), code = document.createElement('pre');
    summary.textContent = 'Protocol configuration';
    code.textContent = JSON.stringify(spec, null, 2);
    details.append(summary, code);
    target.append(details);
}
export function updateProtocol(nextProfile, nextBusy, nextValidation) {
    const previous = protocolSpec().measurements.regions;
    profile = nextProfile;
    busy = nextBusy;
    validation = nextValidation;
    const targets = (profile?.targets ?? []).map(target => target.layer).join('\n');
    const suggestions = $('protocol-targets');
    if (suggestions.dataset.key !== targets) {
        suggestions.dataset.key = targets;
        suggestions.replaceChildren(...(profile?.targets ?? []).map(target => { const option = document.createElement('option'); option.value = target.layer; option.label = `${target.name} (${target.units} units)`; return option; }));
    }
    if (profile && profileID !== profile.identifier) {
        const first = !profileID;
        profileID = profile.identifier;
        renderRegions(first && !previous.length ? profile.out_channels.filter(channel => channel.startsWith('neural:')).slice(0, 1).map(channel => channel.slice(7)) : previous);
    }
    const task = value('protocol-task');
    $('protocol-choice-settings').hidden = !['choice', 'legacy'].includes(task);
    $('protocol-modern-choice').hidden = task !== 'choice';
    $('protocol-confidence-settings').hidden = task !== 'choice' || !checked('protocol-confidence');
    $('protocol-rating-settings').hidden = task !== 'rating';
    $('protocol-answer-settings').hidden = task === 'passive';
    $('protocol-record-behavior-hint').textContent = task === 'passive' ? 'Plain viewing has no answer; choose a response in Task to record behavior.' : `Requested answer outputs: ${responseChannels(protocolSpec().task).join(', ')}`;
    const problem = firstIssue();
    const summaries = stepSummaries();
    // The rig has no order: every part can be opened at any time and shows its own to-do.
    document.querySelectorAll('[data-protocol-nav]').forEach(button => {
        const step = Number(button.dataset.protocolNav);
        const todo = step < 5 ? issue(step) : '';
        const summary = button.querySelector('[data-step-summary]');
        if (summary)
            summary.textContent = todo || summaries[step];
        button.toggleAttribute('data-problem', Boolean(todo));
        button.toggleAttribute('data-done', step === 5 && protocolRunReady());
        button.setAttribute('aria-current', step === active ? 'step' : 'false');
        button.disabled = busy;
    });
    renderSites();
    document.querySelectorAll('[data-protocol-step]').forEach(section => { section.hidden = Number(section.dataset.protocolStep) !== active; });
    $('protocol-step-position').textContent = `${active + 1} / ${labels.length} · ${labels[active]}`;
    $('protocol-back').disabled = busy || active === 0;
    $('protocol-next').hidden = active === 5;
    $('protocol-next').disabled = busy || Boolean(issue(active));
    $('protocol-step-help').textContent = issue(active);
    $('protocol-step-help').hidden = !issue(active);
    $('protocol-review-actions').hidden = active !== 5;
    $('validate-protocol').disabled = busy || Boolean(problem);
    $('protocol-save').disabled = busy || Boolean(problem);
    $('protocol-python').disabled = busy || Boolean(problem);
    $('protocol-editor').querySelectorAll('input,select,textarea').forEach(input => { if (!input.closest('#model-loader'))
        input.disabled = busy; });
    $('protocol-conditions').querySelectorAll('button').forEach(button => { button.disabled = busy; });
    $('protocol-add-condition').disabled = busy || protocolSpec().conditions.length >= 12;
    if (validation?.id === pendingID && validation.phase === 'valid' && pendingKey)
        validatedKey = pendingKey;
    const receipt = $('protocol-validation');
    receipt.replaceChildren();
    let text = 'Check the complete setup before running. Validation does not run model inference.';
    if (validation?.id === pendingID && validation.phase === 'pending')
        text = 'Checking the subject, stimuli, task and recorders…';
    else if (validation?.id === pendingID && validation.phase === 'invalid')
        text = validation.error ?? 'Validation failed';
    else if (protocolRunReady())
        text = `Validated · ${validation?.summary?.compatibility?.match ?? 'compatible'} inputs · ${validation?.summary?.presentations} presentations. Model-specific task settings are also checked when the session starts.`;
    else if (validatedKey)
        text = 'The configuration changed. Validate this draft again before running.';
    const message = document.createElement('p');
    message.textContent = text;
    receipt.append(message);
    for (const note of validation?.id === pendingID ? validation.summary?.notes ?? [] : []) {
        const item = document.createElement('p');
        item.className = 'subtle';
        item.textContent = note;
        receipt.append(item);
    }
    if (active === 5)
        showReview();
}
export function restoreProtocol(spec) {
    spec = Object.assign({ identifier: 'my-protocol', repetitions: 1, required_traits: [], conditions: [{ name: 'baseline' }] }, spec, {
        task: Object.assign({ kind: 'passive', instruction: '', labels: [], probabilities: false, path: 'auto', readout_fit: 'current', present: [], trial_memory: 'none' }, spec.task),
        measurements: Object.assign({ behavior: false, regions: [], replay: false }, spec.measurements)
    });
    $('protocol-name').value = spec.identifier;
    $('protocol-task').value = spec.task.kind;
    $('protocol-instruction').value = spec.task.instruction;
    $('protocol-labels').value = spec.task.labels.join(',');
    $('protocol-probabilities').checked = spec.task.probabilities;
    $('protocol-confidence').checked = Boolean(spec.task.confidence);
    $('protocol-path').value = spec.task.path;
    $('protocol-fit-policy').value = spec.task.readout_fit;
    $('protocol-present').value = spec.task.present.join(',');
    for (const [kind, bounds] of [['scale', spec.task.scale], ['confidence', spec.task.confidence]])
        if (bounds) {
            $(`protocol-${kind}-low`).value = String(bounds.low);
            $(`protocol-${kind}-high`).value = String(bounds.high);
            if (kind === 'scale')
                $('protocol-scale-step').value = bounds.step === null ? '' : String(bounds.step);
        }
    $('protocol-record-behavior').checked = spec.measurements.behavior;
    $('protocol-record-replay').checked = spec.measurements.replay;
    renderRegions(spec.measurements.regions);
    $('protocol-repetitions').value = String(spec.repetitions);
    $('protocol-trial-memory').value = spec.task.trial_memory;
    $('protocol-conditions').replaceChildren(...spec.conditions.map(conditionCard));
}
export function initializeProtocol(callbacks) {
    api = callbacks;
    $('protocol-conditions').append(conditionCard({ name: 'baseline' }));
    document.querySelectorAll('[data-protocol-nav]').forEach(button => { button.onclick = () => { active = Number(button.dataset.protocolNav); api.changed(); }; });
    $('protocol-next').onclick = () => { if (!issue(active)) {
        active = Math.min(5, active + 1);
        api.changed();
    } };
    $('protocol-back').onclick = () => { active = Math.max(0, active - 1); api.changed(); };
    $('protocol-editor').querySelectorAll('input,select,textarea').forEach(input => { if (!input.closest('.protocol-condition'))
        input.oninput = api.changed; });
    $('protocol-task').onchange = () => { $('protocol-record-behavior').checked = value('protocol-task') !== 'passive'; api.changed(); };
    $('protocol-add-condition').onclick = () => { $('protocol-conditions').append(conditionCard({ name: `condition-${protocolSpec().conditions.length + 1}` })); api.changed(); };
    $('protocol-fitting-csv').onchange = async (event) => {
        try {
            const file = event.target.files?.[0];
            if (!file)
                return;
            if (file.size > 1024 * 1024)
                throw new Error('Fitting CSV limit is 1 MiB');
            const source = await file.text(), rows = api.parseCSV(source);
            $('protocol-fitting').value = source;
            $('protocol-fitting-status').textContent = `${rows.length} fitting rows imported`;
            api.changed();
        }
        catch (error) {
            api.error(String(error));
        }
    };
    $('protocol-fitting-media').onchange = async (event) => {
        try {
            const files = [...(event.target.files ?? [])];
            if (!files.length)
                return;
            const rows = await api.uploadMediaFiles(files, name => { $('protocol-fitting-status').textContent = `Uploading ${name}…`; });
            $('protocol-fitting').value = api.toCSV(rows.map(row => ({ ...row, stimulus_id: `fitting-${row.stimulus_id}`, label: '' })));
            $('protocol-fitting-status').textContent = `${rows.length} fitting files uploaded. Fill their label column before validation.`;
            api.changed();
        }
        catch (error) {
            api.error(String(error));
        }
    };
    $('validate-protocol').onclick = async () => {
        try {
            await validateProtocol();
        }
        catch (error) {
            api.error(String(error));
        }
    };
    $('protocol-save').onclick = () => {
        try {
            const file = { kind: 'workbench', subject: profile?.identifier, model_source: profile?.calibration ? 'profile' : 'registered', request: protocolRequest() };
            const url = URL.createObjectURL(new Blob([JSON.stringify(file, null, 2)], { type: 'application/json' }));
            const link = document.createElement('a');
            link.href = url;
            link.download = `${protocolSpec().identifier.replace(/[^a-z0-9_-]/gi, '-')}.json`;
            link.click();
            URL.revokeObjectURL(url);
        }
        catch (error) {
            api.error(String(error));
        }
    };
    $('protocol-python').onclick = async () => {
        try {
            const saved = { kind: 'workbench', subject: profile?.identifier, model_source: profile?.calibration ? 'profile' : 'registered', request: protocolRequest() };
            const result = await api.post('/api/export-python', saved);
            const url = URL.createObjectURL(new Blob([result.source], { type: 'text/x-python' }));
            const link = document.createElement('a');
            link.href = url;
            link.download = result.filename;
            link.click();
            URL.revokeObjectURL(url);
        }
        catch (error) {
            api.error(String(error));
        }
    };
}
