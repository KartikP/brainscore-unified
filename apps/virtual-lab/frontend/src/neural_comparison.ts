import { fullHeatmap, HeatmapScale } from './full_heatmap.js';
import { difference, largestChanges, variableUnits, distance, loadComparisons, type DistanceMetric } from './neural_math.js';

const node = <K extends keyof HTMLElementTagNameMap>(tag: K, text = '', className = '') => {
  const result = document.createElement(tag); result.textContent = text; result.className = className; return result;
};
const number = (value: number) => Number.isFinite(value) ? value.toPrecision(4) : 'missing';
function color(value: number, extent: number): string {
  if (!Number.isFinite(value)) return 'rgb(220,220,220)';
  const target = value < 0 ? [72, 111, 157] : [181, 119, 58];
  return `rgb(${target.map(component => Math.round(250 + (component - 250) * Math.min(1, Math.abs(value) / extent))).join(',')})`;
}
function extent(values: Iterable<number>): number {
  let high = 0;
  for (const value of values) if (Number.isFinite(value)) high = Math.max(high, Math.abs(value));
  return high || 1;
}
function select(label: string, choices: { value: string; text: string }[]) {
  const control = node('label', label), input = node('select');
  input.setAttribute('aria-label', label);
  for (const choice of choices) input.append(new Option(choice.text, choice.value));
  control.append(input); return { control, input };
}
function legend(high: number, description: string): HTMLElement {
  const result = node('div', '', 'activation-legend');
  result.append(node('span', `−${number(high)}`), node('span', '', 'activation-ramp'), node('span', `+${number(high)}`), node('span', description));
  return result;
}

function differencePlot(a: Float64Array, b: Float64Array, label: string): HTMLElement {
  const delta = difference(a, b), high = extent(delta);
  const panel = node('div');
  panel.append(node('h4', label));
  const canvas = node('canvas'); canvas.tabIndex = 0; canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', `Difference for every recorded unit: ${label}. Click or use arrow keys to inspect changes.`);
  const columns = Math.ceil(Math.sqrt(delta.length)), rows = Math.ceil(delta.length / columns);
  canvas.width = columns; canvas.height = rows;
  const cell = Math.max(1, Math.min(10, Math.floor(420 / rows)));
  canvas.style.width = `${columns * cell}px`; canvas.style.height = `${rows * cell}px`;
  const context = canvas.getContext('2d')!, pixels = context.createImageData(columns, rows);
  for (let index = 0; index < delta.length; index++) {
    const target = delta[index] < 0 ? [72, 111, 157] : [181, 119, 58];
    const strength = Number.isFinite(delta[index]) ? Math.min(1, Math.abs(delta[index]) / high) : 0;
    for (let channel = 0; channel < 3; channel++) pixels.data[index * 4 + channel] = Number.isFinite(delta[index])
      ? Math.round(250 + (target[channel] - 250) * strength) : 220;
    pixels.data[index * 4 + 3] = 255;
  }
  context.putImageData(pixels, 0, 0);
  const scroll = node('div', '', 'full-heatmap-scroll comparison-difference'); scroll.append(canvas);
  const detail = node('p', 'Click a cell, or use arrow keys, to inspect its unit index and signed change.', 'plot-selection');
  detail.setAttribute('aria-live', 'polite');
  let selected = 0;
  const inspect = (index: number) => {
    selected = Math.max(0, Math.min(delta.length - 1, index));
    detail.textContent = `Unit ${selected}: A ${number(a[selected])} − B ${number(b[selected])} = ${number(delta[selected])}`;
  };
  canvas.onclick = event => {
    const column = Math.floor(event.offsetX / cell), row = Math.floor(event.offsetY / cell), index = row * columns + column;
    if (index < delta.length) inspect(index);
  };
  canvas.onkeydown = event => {
    const offsets: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -columns, ArrowDown: columns, Home: -selected, End: delta.length - 1 - selected };
    if (event.key in offsets) { event.preventDefault(); inspect(selected + offsets[event.key]); }
  };
  panel.append(scroll, legend(high, 'A − B · 0 = no change · gray = missing'), detail,
    node('p', `All ${delta.length.toLocaleString()} units in recorded order. Signed raw differences; this pair has its own color scale. No anatomical layout or significance test.`, 'subtle'),
    node('h4', 'Largest absolute changes'));
  const ranked = node('div', '', 'bar-chart');
  for (const unit of largestChanges(delta)) {
    const row = node('div', '', 'bar-item'), track = node('div', '', 'bar-track');
    const zero = node('span', '', 'bar-zero'); zero.style.left = '50%';
    const fill = node('span', '', `bar-fill${delta[unit] < 0 ? ' negative' : ''}`);
    const position = delta[unit] / high * 50;
    fill.style.left = `${50 + Math.min(0, position)}%`; fill.style.width = `${Math.abs(position)}%`;
    track.append(zero, fill); track.setAttribute('aria-hidden', 'true');
    row.append(node('span', `Unit ${unit}`, 'bar-label'), track, node('span', number(delta[unit]), 'bar-value'));
    ranked.append(row);
  }
  panel.append(ranked, node('p', 'Up to 20 units ranked by absolute change for this pair. Large changes do not establish causal importance.', 'subtle'));
  return panel;
}

function variationPlot(rows: Float64Array[], labels: string[]): HTMLElement {
  const { indices, means } = variableUnits(rows);
  const high = extent(rows.flatMap(row => indices.map(unit => row[unit] - means[unit])));
  const panel = node('div'), scroll = node('div', '', 'heatmap-scroll'), table = node('table', '', 'activation-grid');
  table.append(node('caption', 'Stimuli × most variable units: each unit’s mean across these stimuli has been subtracted'));
  const header = node('tr'); header.append(node('th', 'Stimulus'));
  for (const unit of indices) { const cell = node('th', String(unit)); cell.scope = 'col'; header.append(cell); }
  table.append(header);
  const detail = node('p', 'Select a cell to inspect the raw activation and its deviation from that unit’s mean.', 'plot-selection');
  detail.setAttribute('aria-live', 'polite');
  rows.forEach((row, index) => {
    const tr = node('tr'), label = node('th', labels[index], 'heatmap-row'); label.scope = 'row'; label.title = labels[index]; tr.append(label);
    for (const unit of indices) {
      const value = row[unit], centered = value - means[unit], td = node('td'), button = node('button', '', 'activation-cell'); button.type = 'button';
      const description = `${labels[index]}, unit ${unit}: raw ${number(value)}, mean ${number(means[unit])}, deviation ${number(centered)}`;
      button.style.background = color(centered, high); button.title = description; button.setAttribute('aria-label', description);
      button.onclick = () => { detail.textContent = description; };
      td.append(button); tr.append(td);
    }
    table.append(tr);
  });
  scroll.append(table); panel.append(scroll, legend(high, 'Activation − unit mean across these stimuli'), detail,
    node('p', `Showing ${indices.length} of ${rows[0].length.toLocaleString()} units, ranked by max − min across these ${rows.length} stimuli. Means use finite values; missing cells remain missing. Values are centered, not standardized or divided by variance. The raw and difference views retain every unit.`, 'subtle'));
  return panel;
}

function distancePlot(rows: Float64Array[], labels: string[], metric: DistanceMetric): HTMLElement {
  const panel = node('div'), scroll = node('div', '', 'heatmap-scroll'), table = node('table', '', 'distance-table');
  table.append(node('caption', metric === 'rms' ? 'Root mean square activation difference (raw activation units)' : 'Correlation distance (1 − Pearson r)'));
  const header = node('tr'); header.append(node('th', 'Stimulus'));
  labels.forEach(label => { const th = node('th', label); th.scope = 'col'; th.title = label; header.append(th); }); table.append(header);
  const pairs = rows.map(a => rows.map(b => distance(a, b, metric)));
  const high = metric === 'correlation' ? 2 : extent(pairs.flatMap(row => row.map(pair => pair.value ?? NaN)));
  const detail = node('p', 'Select a pair to inspect its distance and number of jointly finite units.', 'plot-selection'); detail.setAttribute('aria-live', 'polite');
  pairs.forEach((row, i) => {
    const tr = node('tr'), th = node('th', labels[i]); th.scope = 'row'; th.title = labels[i]; tr.append(th);
    row.forEach((pair, j) => {
      const td = node('td'), button = node('button', pair.value === null ? 'Missing' : number(pair.value)); button.type = 'button';
      button.style.background = color(pair.value ?? NaN, high);
      const description = `${labels[i]} vs ${labels[j]}: ${pair.value === null ? 'undefined' : number(pair.value)}; ${pair.count} of ${rows[0].length} jointly finite units`;
      button.title = description; button.setAttribute('aria-label', description); button.onclick = () => { detail.textContent = description; };
      td.append(button); tr.append(td);
    }); table.append(tr);
  });
  scroll.append(table); panel.append(scroll, detail, node('p', metric === 'rms'
    ? '0 = identical. RMS = √mean((A − B)²), over jointly finite units. Shared offsets cancel; units with large changes can still dominate. Compare distances within this recording, not absolute scales across regions.'
    : '0 = perfect correlation; 1 = no linear correlation; 2 = perfect anticorrelation. Constant vectors are undefined. Shared outlier dimensions can keep correlations high despite image differences.', 'subtle'));
  return panel;
}

export function neuralViews(source: string, scale: HeatmapScale, stimulusLabels: string[]): HTMLElement {
  const container = node('div');
  const view = select('Neural view', [
    { value: 'raw', text: 'Raw heatmap — all units' }, { value: 'difference', text: 'Difference between two stimuli' },
    { value: 'variation', text: 'Stimulus variation' }, { value: 'distances', text: 'Pairwise distances' },
  ]);
  const raw = fullHeatmap(source, scale, stimulusLabels), comparison = node('div'); comparison.hidden = true;
  container.append(view.control, raw, comparison);
  let saved: Awaited<ReturnType<typeof loadComparisons>> | undefined, controller: AbortController | undefined, sequence = 0;
  const render = async () => {
    const current = ++sequence;
    controller?.abort();
    const kind = view.input.value;
    raw.hidden = kind !== 'raw'; comparison.hidden = kind === 'raw';
    if (kind === 'raw') return;
    comparison.replaceChildren(node('p', 'Loading saved activations for comparison…', 'subtle'));
    try {
      if (!saved) {
        controller = new AbortController(); const signal = controller.signal;
        saved = await loadComparisons(async (presentation, offset) => {
          const response = await fetch(`${source}&presentation=${presentation}&offset=${offset}&limit=2048`, { signal });
          const result = await response.json();
          if (!response.ok) throw new Error(result.error ?? 'Saved neural recording unavailable');
          return result;
        }, signal);
      }
      if (current !== sequence || !container.isConnected) return;
      const { rows, slice } = saved, labels = saved.labels.map((label, i) => `${i + 1}: ${stimulusLabels[i] ?? label}`);
      const controls = node('div', '', 'comparison-controls'), plot = node('div');
      comparison.replaceChildren(controls, plot);
      if (kind === 'difference') {
        const choices = labels.map((text, index) => ({ value: String(index), text }));
        const a = select('Stimulus A', choices), b = select('Stimulus B', choices); b.input.value = '1';
        controls.append(a.control, b.control);
        const update = () => plot.replaceChildren(differencePlot(rows[Number(a.input.value)], rows[Number(b.input.value)], `${labels[Number(a.input.value)]} − ${labels[Number(b.input.value)]}`));
        a.input.onchange = b.input.onchange = update; update();
      } else if (kind === 'variation') plot.append(variationPlot(rows, labels));
      else {
        const metric = select('Distance measure', [{ value: 'rms', text: 'RMS difference' }, { value: 'correlation', text: 'Correlation distance (1 − r)' }]);
        controls.append(metric.control);
        const update = () => plot.replaceChildren(distancePlot(rows, labels, metric.input.value as DistanceMetric));
        metric.input.onchange = update; update();
      }
      const fixed = Object.entries(slice).map(([key, value]) => `${key}=${value}`).join(', ');
      comparison.append(node('p', `Uses this channel, condition and repetition only.${fixed ? ` Fixed slice: ${fixed}.` : ''} Descriptive comparisons; more stimuli and suitable controls are needed to infer category selectivity. Computed from saved arrays without running the model.`, 'subtle'));
    } catch (error) {
      if (current === sequence) comparison.replaceChildren(node('p', String(error), 'error'));
    }
  };
  view.input.onchange = () => void render();
  return container;
}
