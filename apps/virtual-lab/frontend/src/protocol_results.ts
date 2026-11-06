import { HeatmapScale } from './full_heatmap.js';

export function renderProtocolResults(report: Record<string, unknown>, container: HTMLElement, runID: string,
    render: (observations: Record<string, unknown>, target: HTMLElement, scale: HeatmapScale) => void): void {
  const conditions = report.conditions as string[];
  const channels = report.channels as string[];
  const repetitions = Number(report.repetitions);
  const title = document.createElement('p'); title.className = 'response-meta';
  title.textContent = `${report.subject} / ${report.identifier} · ${report.stimuli} stimuli × ${conditions.length} conditions × ${repetitions} repetitions`;
  const controls = document.createElement('div'); controls.className = 'protocol-result-controls';
  const conditionLabel = document.createElement('label'); conditionLabel.textContent = 'Condition';
  const condition = document.createElement('select'); condition.setAttribute('aria-label', 'Recorded condition');
  conditions.forEach(name => condition.append(new Option(name, name))); conditionLabel.append(condition);
  const repetitionLabel = document.createElement('label'); repetitionLabel.textContent = 'Repetition';
  const repetition = document.createElement('input'); repetition.type = 'number'; repetition.min = '1'; repetition.max = String(repetitions); repetition.value = '1';
  repetition.setAttribute('aria-label', 'Recorded repetition'); repetitionLabel.append(repetition);
  const channelLabel = document.createElement('label'); channelLabel.textContent = 'Measurement';
  const channel = document.createElement('select'); channel.setAttribute('aria-label', 'Recorded measurement');
  channels.forEach(name => channel.append(new Option(name, name))); channelLabel.append(channel);
  controls.append(conditionLabel, repetitionLabel, channelLabel);
  const content = document.createElement('div');
  const status = document.createElement('p'); status.className = 'subtle'; status.setAttribute('role', 'status');
  container.append(title, controls, status, content);
  const scales = new Map<string, HeatmapScale>();
  let request = 0;
  const load = async () => {
    const token = ++request;
    const rep = Number(repetition.value) - 1;
    if (!Number.isInteger(rep) || rep < 0 || rep >= repetitions) { status.textContent = `Choose repetition 1–${repetitions}.`; return; }
    const index = conditions.indexOf(condition.value) * repetitions + rep;
    status.textContent = 'Reading saved measurements…';
    try {
      const response = await fetch(`/api/runs/${runID}/observations?index=${index}&channel=${encodeURIComponent(channel.value)}`);
      const observations = await response.json();
      if (!response.ok) throw new Error(observations.error);
      if (token !== request || !container.isConnected) return;
      content.replaceChildren();
      observations.outputs.forEach((output: Record<string, unknown>) => { output.condition = condition.value; output.repetition = rep; });
      const scaleKey = channel.value;
      if (!scales.has(scaleKey)) scales.set(scaleKey, new HeatmapScale());
      render({ kind: 'session', subject: report.subject, output: channel.value, outputs: observations.outputs }, content, scales.get(scaleKey)!);
      status.textContent = 'Saved data only; inspecting this result does not run the subject. The neural scale is held across inspected conditions and expands if a larger value is found.';
    } catch (error) { if (token === request) status.textContent = String(error); }
  };
  condition.onchange = channel.onchange = repetition.onchange = () => void load();
  for (const note of report.notes as string[] ?? []) { const paragraph = document.createElement('p'); paragraph.className = 'result-note'; paragraph.textContent = note; container.append(paragraph); }
  queueMicrotask(() => void load());
}
